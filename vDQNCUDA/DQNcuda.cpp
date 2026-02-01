#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <random>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>  // 必须添加这个才能用 cudaDeviceSynchronize

using namespace torch::indexing;

struct HyperParams {
    double gamma = 0.99;
    double eps_start = 1.0;
    double eps_end = 0.01;
    double lr = 5e-4;
    int batch_size = 64;
    int total_episodes = 3000;
    int memory_capacity = 20000;
    std::string filename = "dqn_results_cuda.txt";
};

// 结果保存函数：将训练指标写入磁盘
void save_results_to_txt(const std::vector<int>& step_records, const std::vector<double>& episode_cumulative_times, std::vector<double>& episode_local_times, const HyperParams& p) {
    std::cout << "\nSaving results to " << p.filename << "..." << std::endl;
    std::ofstream f(p.filename);
    if (f.is_open()) {
        f << "# --- Hyperparameters ---\n";
        f << "# GAMMA: " << p.gamma << "\n# EPS_START: " << p.eps_start << "\n# EPS_END: " << p.eps_end << "\n";
        f << "# BATCH_SIZE: " << p.batch_size << "\n# LR: " << p.lr << "\n# Total Episodes: " << p.total_episodes << "\n";
        f << "# -----------------------\n\n";
        f << "Episode\tCurrent_Lifespan\tElapsed_Time_perStep(s)\tTotal_Elapsed_Time(s)\n";
        for (size_t i = 0; i < step_records.size(); ++i) {
            f << std::left << std::setw(10) << (i + 1) << "\t"
                << std::left << std::setw(10) << step_records[i] << "\t"
                << std::left << std::setw(10) << std::fixed << std::setprecision(4) << episode_local_times[i] << "\t"
                << std::left << std::setw(10) << episode_cumulative_times[i] << "\n";
        }
        f.close();
        std::cout << "Saving complete." << std::endl;
    }
}

// 1. 神经网络定义：确保模型可以根据设备初始化
struct DQNNetImpl : torch::nn::Module {
    torch::nn::Linear l1{ nullptr }, l2{ nullptr }, l3{ nullptr };

    DQNNetImpl() {
        l1 = register_module("l1", torch::nn::Linear(4, 128));
        l2 = register_module("l2", torch::nn::Linear(128, 128));
        l3 = register_module("l3", torch::nn::Linear(128, 2));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(l1(x));
        x = torch::relu(l2(x));
        return l3(x);
    }
};
TORCH_MODULE(DQNNet);

class CartPole {
public:
    double x = 0, x_dot = 0, theta = 0, theta_dot = 0;
    const double g = 9.8, mc = 1.0, mp = 0.1, L = 0.5, force = 10.0, dt = 0.02;

    void reset() {
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> d(-0.05, 0.05);
        x = d(gen); x_dot = d(gen); theta = d(gen); theta_dot = d(gen);
    }

    bool step(int action) {
        double f = (action == 1) ? force : -force;
        double cost = std::cos(theta), sint = std::sin(theta);
        double temp = (f + mp * L * theta_dot * theta_dot * sint) / (mc + mp);
        double theta_acc = (g * sint - cost * temp) / (L * (4.0 / 3.0 - mp * cost * cost / (mc + mp)));
        double x_acc = temp - mp * L * theta_acc * cost / (mc + mp);
        x += dt * x_dot; x_dot += dt * x_acc;
        theta += dt * theta_dot; theta_dot += dt * theta_acc;
        return std::abs(x) > 4.8 || std::abs(theta) > 0.418;
    }

    torch::Tensor state() { return torch::tensor({ (float)x, (float)x_dot, (float)theta, (float)theta_dot }); }
};

struct Transition { torch::Tensor s, ns; int a; float r; bool d; };

class DQNAgent {
    DQNNet model, target;
    torch::optim::Adam opt;
    std::deque<Transition> memory;
    std::mt19937 rd_gen{ std::random_device{}() };
    torch::Device device; // 【新增】设备成员变量

public:
    int64_t steps = 0;
    // 【修改】构造函数初始化设备并移动模型
    DQNAgent(const HyperParams& hp) :
        device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        model(DQNNet()), target(DQNNet()),
        opt(model->parameters(), torch::optim::AdamOptions(hp.lr)) 
    {
        std::cout << "Using Device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
        model->to(device);  // 将模型移至 GPU
        target->to(device); // 将目标网络移至 GPU
        update_target(1.0);
    }

    int act(torch::Tensor s, const HyperParams& hp) {
        double eps = hp.eps_end + (hp.eps_start - hp.eps_end) * std::exp(-1. * steps++ / 2000.);
        if (std::uniform_real_distribution<>(0, 1)(rd_gen) < eps) return rd_gen() % 2;

        torch::NoGradGuard no_grad;
        // 【修改】将输入张量 s 移至设备，并添加 batch 维度
        auto s_cuda = s.to(device).unsqueeze(0);
        return model->forward(s_cuda).argmax(1).item<int>();
    }

    void store(Transition t, int cap) {
        memory.push_back(t);
        if (memory.size() > cap) memory.pop_front();
    }

    void update_target(double tau) {
        torch::NoGradGuard no_grad;
        auto p = model->parameters(), tp = target->parameters();
        for (size_t i = 0; i < p.size(); ++i) tp[i].copy_(tau * p[i] + (1 - tau) * tp[i]);
    }

    void train(int sz, double gamma) {

        if (memory.size() < sz) return;

        std::vector<torch::Tensor> bs, bns, ba, br, bd;
        for (int i = 0; i < sz; ++i) {
            auto& m = memory[rd_gen() % memory.size()];
            bs.push_back(m.s);
            bns.push_back(m.ns);
            ba.push_back(torch::full({ 1 }, (long)m.a, torch::kLong));
            br.push_back(torch::full({ 1 }, (float)m.r, torch::kFloat32));
            bd.push_back(torch::full({ 1 }, (float)(m.d ? 1.0f : 0.0f), torch::kFloat32));
        }

        // 【修改】将拼接后的整个 Batch 移至设备
        auto s_batch = torch::stack(bs).to(device);
        auto ns_batch = torch::stack(bns).to(device);
        auto a_batch = torch::stack(ba).to(device);
        auto r_batch = torch::stack(br).to(device);
        auto d_batch = torch::stack(bd).to(device);

        auto q = model->forward(s_batch).gather(1, a_batch);

        torch::Tensor next_q;
        {
            torch::NoGradGuard no_grad;
            next_q = std::get<0>(target->forward(ns_batch).max(1, true));
        }
        auto target_q = r_batch + (1 - d_batch) * gamma * next_q;

        auto loss = torch::smooth_l1_loss(q, target_q);
        opt.zero_grad(); 
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        opt.step();

        update_target(0.005);
    }
};

// 5. 主循环：协调环境与 Agent 的交互
int main() {

    HyperParams hp;
    CartPole env;
    DQNAgent agent(hp);
    // cv::Mat canvas = cv::Mat::zeros(400, 600, CV_8UC3); // OpenCV 画布

    // 初始化统计数据收集器
    std::vector<int> step_records;
    std::vector<double> episode_cumulative_times;
    std::vector<double> episode_local_times;
    auto global_start_time = std::chrono::high_resolution_clock::now();

    for (int ep = 0; ep < hp.total_episodes; ++ep) {
        env.reset();
        int step_cnt = 0;
        auto local_start_time = std::chrono::high_resolution_clock::now(); // 记录单局起始时刻

        while (step_cnt < 500) {
            auto s = env.state();        // 获取当前状态
            int a = agent.act(s, hp);    // 智能体做决策
            bool done = env.step(a);     // 环境执行动作
            auto ns = env.state();       // 获取新状态

            // 奖励逻辑
            float r = done ? -10.0f : 1.0f;
            agent.store({ s, ns, a, r, done }, hp.memory_capacity); // 存入经验

            // 训练触发：每 4 个 Step 训练一次网络
            if (agent.steps % 4 == 0) agent.train(hp.batch_size, hp.gamma);

            step_cnt++;
            if (done) break;

            // 每 5 局展示一次可视化画面
            // if (ep % 5 == 0) {
            //     canvas.setTo(0); // 清空画布
            //     int x_pix = (int)(env.x * 100 + 300); // 物理坐标映射到像素坐标
            //     cv::rectangle(canvas, { x_pix - 20, 280, 40, 20 }, { 0,255,0 }, -1); // 画车
            //     cv::line(canvas, { x_pix, 280 }, { x_pix + (int)(sin(env.theta) * 80), 280 - (int)(cos(env.theta) * 80) }, { 0,0,255 }, 3); // 画杆
            //     cv::imshow("DQN", canvas);
            //     if (cv::waitKey(1) == 27) return 0; // 按 ESC 退出
            // }
        }

        // --- 单局结束，统计数据 ---
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cumulative_diff = now - global_start_time; // 计算总运行时间
        std::chrono::duration<double> local_diff = now - local_start_time;      // 计算单局耗时

        step_records.push_back(step_cnt);
        episode_cumulative_times.push_back(cumulative_diff.count());
        episode_local_times.push_back(local_diff.count());

        // 控制台日志输出
        if (ep % 10 == 0)
            printf("Ep: %d | Steps: %d | Time: %.2fs | Total: %.1fs\n",
                ep, step_cnt, local_diff.count(), cumulative_diff.count());

        // 定期自动存盘
        if (ep > 0 && ep % 50 == 0) {
            save_results_to_txt(step_records, episode_cumulative_times, episode_local_times, hp);
        }
    }

    // 训练全部结束后的最后一次保存
    save_results_to_txt(step_records, episode_cumulative_times, episode_local_times, hp);

    return 0;
}