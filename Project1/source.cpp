#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <random>
#include <fstream>
#include <chrono>

using namespace torch::indexing; // 方便张量切片，类似于 Python 中的 [:, 1] 操作

// 1. 配置管理：使用结构体统一管理超参数
struct HyperParams {
    double gamma = 0.99;            // 折扣因子：对未来奖励的重视程度
    double eps_start = 1.0;         // 探索率起点：初始时 100% 随机尝试动作
    double eps_end = 0.01;          // 探索率终点：最小保留 1% 的随机性以持续探索
    double lr = 5e-4;               // 学习率：决定神经网络参数更新的步长
    int batch_size = 64;            // 批处理大小：每次从经验池中抽取多少条样本训练
    int total_episodes = 3000;      // 总训练局数
    int memory_capacity = 20000;    // 经验回放池的最大容量
    std::string filename = "dqn_results.txt"; // 数据保存路径
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

// 2. 神经网络定义：DQN 的大脑
struct DQNNetImpl : torch::nn::Module {
    // 定义三层全连接层指针
    torch::nn::Linear l1{ nullptr }, l2{ nullptr }, l3{ nullptr };

    DQNNetImpl() {
        // 在构造函数中注册模块，这样 optimizer 才能识别到这些参数
        l1 = register_module("l1", torch::nn::Linear(4, 128)); // 输入层：4个状态量 (x, x_dot, theta, theta_dot)
        l2 = register_module("l2", torch::nn::Linear(128, 128));
        l3 = register_module("l3", torch::nn::Linear(128, 2));   // 输出层：2个动作的 Q 值
    }

    // 前向传播逻辑
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(l1(x)); // 隐藏层 1 使用 ReLU 激活
        x = torch::relu(l2(x)); // 隐藏层 2 使用 ReLU 激活
        return l3(x);           // 输出层输出原始 Q 值
    }
};
// 使用宏生成包装类 DQNNet (智能指针形式)，简化内存管理
TORCH_MODULE(DQNNet);

// 3. 环境模拟：CartPole 物理引擎
class CartPole {
public:
    double x = 0, x_dot = 0, theta = 0, theta_dot = 0;
    // 物理常数：g-重力，mc-车重，mp-杆重，L-杆半长，force-推力大小，dt-时间步长
    const double g = 9.8, mc = 1.0, mp = 0.1, L = 0.5, force = 10.0, dt = 0.02;

    // 重置环境：每局开始时赋予微小的随机扰动
    void reset() {
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> d(-0.05, 0.05);
        x = d(gen); x_dot = d(gen); theta = d(gen); theta_dot = d(gen);
    }

    // 环境步进逻辑：根据物理公式更新状态
    bool step(int action) {
        double f = (action == 1) ? force : -force; // 根据 action 决定推力方向
        double cost = std::cos(theta), sint = std::sin(theta);

        // 计算物理加速度 (核心微分方程简化版)
        double temp = (f + mp * L * theta_dot * theta_dot * sint) / (mc + mp);
        double theta_acc = (g * sint - cost * temp) / (L * (4.0 / 3.0 - mp * cost * cost / (mc + mp)));
        double x_acc = temp - mp * L * theta_acc * cost / (mc + mp);

        // 欧拉积分更新状态
        x += dt * x_dot; x_dot += dt * x_acc;
        theta += dt * theta_dot; theta_dot += dt * theta_acc;

        // 死亡检测：车子跑太远或杆子偏角太大
        return std::abs(x) > 2.4 || std::abs(theta) > 0.209;
    }

    // 获取当前物理状态对应的 Tensor
    torch::Tensor state() { return torch::tensor({ (float)x, (float)x_dot, (float)theta, (float)theta_dot }); }
};

// 4. DQN 智能体：算法核心逻辑
struct Transition { torch::Tensor s, ns; int a; float r; bool d; };

class DQNAgent {
    DQNNet model, target;            // 两个网络：评估网络(model)和目标网络(target)
    torch::optim::Adam opt;          // Adam 优化器
    std::deque<Transition> memory;   // 经验池容器
    std::mt19937 rd_gen{ std::random_device{}() }; // 随机数生成器

public:
    int64_t steps = 0;               // 记录总步数，用于 epsilon 衰减
    DQNAgent(const HyperParams& hp) :
        model(DQNNet()), target(DQNNet()),
        opt(model->parameters(), torch::optim::AdamOptions(hp.lr)) {
        update_target(1.0); // 初始时让目标网络同步评估网络
    }

    // 动作选择：Epsilon-Greedy 策略
    int act(torch::Tensor s, const HyperParams& hp) {
        // 指数级衰减 epsilon：前期多探索，后期多利用
        double eps = hp.eps_end + (hp.eps_start - hp.eps_end) * std::exp(-1. * steps++ / 2000.);
        if (std::uniform_real_distribution<>(0, 1)(rd_gen) < eps) return rd_gen() % 2; // 随机动作

        torch::NoGradGuard no_grad; // 预测阶段不计梯度，加速
        return model->forward(s.unsqueeze(0)).argmax(1).item<int>(); // 选择 Q 值最大的动作
    }

    // 存储经验到池中
    void store(Transition t, int cap) {
        memory.push_back(t);
        if (memory.size() > cap) memory.pop_front(); // 池满后踢出最旧数据
    }

    // 同步目标网络：tau=1.0 为硬同步，tau=0.005 为软更新
    void update_target(double tau) {
        torch::NoGradGuard no_grad;
        auto p = model->parameters(), tp = target->parameters();
        for (size_t i = 0; i < p.size(); ++i) tp[i].copy_(tau * p[i] + (1 - tau) * tp[i]);
    }

    // 训练逻辑：从回放池采样并更新模型
    void train(int sz, double gamma) {
        if (memory.size() < sz) return; // 样本不足则跳过

        // 随机批量采样
        std::vector<torch::Tensor> bs, bns, ba, br, bd;
        for (int i = 0; i < sz; ++i) {
            auto& m = memory[rd_gen() % memory.size()];
            bs.push_back(m.s);
            bns.push_back(m.ns);

            // 使用 torch::full 显式指定 Tensor 形状和类型，规避 C2664 错误
            ba.push_back(torch::full({ 1 }, (long)m.a, torch::kLong));
            br.push_back(torch::full({ 1 }, (float)m.r, torch::kFloat32));
            bd.push_back(torch::full({ 1 }, (float)(m.d ? 1.0f : 0.0f), torch::kFloat32));
        }

        // 拼接成 Batch Tensor
        auto s_batch = torch::stack(bs), ns_batch = torch::stack(bns);
        auto a_batch = torch::stack(ba), r_batch = torch::stack(br), d_batch = torch::stack(bd);

        // 计算当前 Q 值 (Predicted Q)
        auto q = model->forward(s_batch).gather(1, a_batch);

        // 计算目标 Q 值 (Target Q)
        torch::Tensor next_q;
        {
            torch::NoGradGuard no_grad; // 计算下一步 Q 值时不计梯度
            next_q = std::get<0>(target->forward(ns_batch).max(1, true));
        }
        auto target_q = r_batch + (1 - d_batch) * gamma * next_q;

        // 计算 Loss 并反向传播
        auto loss = torch::smooth_l1_loss(q, target_q);
        opt.zero_grad(); loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0); // 梯度裁剪防止爆炸
        opt.step();

        update_target(0.005); // 软更新目标网络权重
    }
};

// 5. 主循环：协调环境与 Agent 的交互
int main() {
    HyperParams hp;
    CartPole env;
    DQNAgent agent(hp);
    cv::Mat canvas = cv::Mat::zeros(400, 600, CV_8UC3); // OpenCV 画布

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
            if (ep % 5 == 0) {
                canvas.setTo(0); // 清空画布
                int x_pix = (int)(env.x * 100 + 300); // 物理坐标映射到像素坐标
                cv::rectangle(canvas, { x_pix - 20, 280, 40, 20 }, { 0,255,0 }, -1); // 画车
                cv::line(canvas, { x_pix, 280 }, { x_pix + (int)(sin(env.theta) * 80), 280 - (int)(cos(env.theta) * 80) }, { 0,0,255 }, 3); // 画杆
                cv::imshow("DQN", canvas);
                if (cv::waitKey(1) == 27) return 0; // 按 ESC 退出
            }
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