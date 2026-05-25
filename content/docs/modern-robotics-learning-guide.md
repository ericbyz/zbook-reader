# 现代机器人学学习指南

> 本文档综合 Northwestern、MIT、Stanford、Michigan、CMU 等顶尖院校课程体系及主流教材，梳理现代机器人学的核心知识模块与学习路径。

---

## 目录

- [一、核心知识体系总览](#一核心知识体系总览)
- [二、数学基础](#二数学基础)
- [三、运动学与动力学](#三运动学和动力学)
- [四、感知](#四感知)
- [五、规划](#五规划)
- [六、控制](#六控制)
- [七、学习与智能](#七学习与智能)
- [八、系统工程与实践](#八系统工程与实践)
- [九、前沿专题](#九前沿专题)
- [十、推荐学习路径](#十推荐学习路径)
- [十一、推荐教材与课程](#十一推荐教材与课程)
- [十二、关键软件工具](#十二关键软件工具)

---

## 一、核心知识体系总览

现代机器人学可归纳为 **七大模块**：

| 模块 | 核心内容 |
|------|----------|
| 数学基础 | 线性代数、概率论、优化、李群李代数 |
| 运动学与动力学 | 正/逆运动学、雅可比、拉格朗日/牛顿-欧拉动力学 |
| 感知 | 计算机视觉、3D 感知、深度学习感知、传感器融合、SLAM |
| 规划 | 运动规划（RRT/PRM）、轨迹优化、抓取规划、TAMP |
| 控制 | PID、LQR、MPC、力/阻抗控制、非线性最优控制 |
| 学习与智能 | 概率机器人、深度强化学习、模仿学习、基础模型、具身 AI |
| 系统工程 | ROS 2、仿真、嵌入式系统、人机交互、安全与伦理 |

---

## 二、数学基础

### 2.1 线性代数

- 向量空间、线性变换、正交基
- 矩阵分解：SVD、QR、Cholesky
- 最小二乘与伪逆
- 特征值与特征向量

### 2.2 概率论与统计学

- 贝叶斯推断、先验/后验分布
- 高斯分布与多元高斯
- 蒙特卡洛方法
- 随机过程基础

### 2.3 优化理论

- 凸优化基础（凸函数、凸集）
- 线性规划与二次规划
- 非线性优化（梯度下降、牛顿法）
- 约束优化与 KKT 条件

### 2.4 微分几何与李群

- 旋转表示（旋转矩阵、欧拉角、四元数、指数坐标）
- SO(3) 与 SE(3)：特殊正交群与特殊欧氏群
- 螺旋理论（Screw Theory）：twist、wrench
- 李群与李代数在机器人中的应用

### 2.5 微分方程

- 常微分方程（ODE）数值求解
- 拉格朗日力学中的运动方程

> **推荐课程：** Michigan ROB 501 — Mathematics for Robotics

---

## 三、运动学和动力学

### 3.1 运动学

#### 正运动学 (Forward Kinematics)

- DH 参数法 (Denavit-Hartenberg)
- 指数积公式 (Product of Exponentials, PoE) — Lynch & Park 的核心方法
- 构型空间 (Configuration Space)

#### 逆运动学 (Inverse Kinematics)

- 解析解与数值解
- Jacobian 迭代法
- 多解问题与工作空间分析

#### 速度运动学与雅可比矩阵

- 空间 Jacobian 与物体 Jacobian
- 奇异点分析 (Singularity)
- 可操作度 (Manipulability)
- 静力学：力与力矩映射

### 3.2 动力学

#### 建模方法

- **拉格朗日力学**：能量方法，适合分析
- **牛顿-欧拉方法**：递归方法，适合计算
- **空间向量代数** (Featherstone)：高效递归算法

#### 核心概念

- 质量矩阵、科里奥利力、重力项
- 正向动力学与逆向动力学
- 任务空间动力学

#### 接触力学

- 接触建模（点接触、面接触）
- 摩擦锥与接触力分析
- 抓取分析 (Grasp Analysis)、形封闭与力封闭

> **推荐课程：** Northwestern Modern Robotics 课程 1-3 | Stanford CS223A | CMU 16-745

---

## 四、感知

### 4.1 传统计算机视觉

- 相机模型（针孔模型、畸变校正）
- 特征提取与匹配（SIFT、ORB、SuperPoint）
- 立体视觉与深度估计
- 图像处理基础

### 4.2 3D 感知

- 点云处理（滤波、分割、配准）
- ICP (Iterative Closest Point) 算法
- 3D 物体检测与分割
- 深度传感器（RGB-D、LiDAR）

### 4.3 深度学习感知

- 目标检测（YOLO、DETR）
- 语义分割与实例分割
- 6-DoF 位姿估计
- 自监督与无监督学习

### 4.4 传感器融合

- 多传感器标定（相机-IMU、相机-LiDAR）
- 卡尔曼滤波（KF、EKF、UKF）
- 粒子滤波
- 多模态融合策略

### 4.5 SLAM（同步定位与建图）

- 视觉 SLAM（ORB-SLAM、VINS）
- 激光 SLAM（Cartographer、LOAM）
- 图优化与位姿图
- EKF-SLAM、FastSLAM、GraphSLAM

> **推荐课程：** MIT 6.4210 | CMU 16-385/720 | Michigan DeepRob (deeprob.org)

---

## 五、规划

### 5.1 运动规划

#### 基于采样的方法

- RRT (Rapidly-exploring Random Trees) 及其变体
- PRM (Probabilistic Roadmaps)
- BIT*、Informed RRT*

#### 基于搜索的方法

- A*、Dijkstra、D* Lite
- 构型空间中的搜索

### 5.2 轨迹优化

- 多项式轨迹插值
- 时间最优轨迹
- 直接配点法 (Direct Collocation)
- 直接射击法 (Direct Shooting)
- 轨迹优化中的约束处理

### 5.3 抓取规划

- 抓取稳定性分析
- 抓取质量度量
- 基于学习的抓取（GraspNet、Contact-GraspNet）

### 5.4 任务与运动规划 (TAMP)

- 符号规划与运动规划的结合
- 任务级推理
- 接触丰富的操作规划

### 5.5 不确定性下的规划

- POMDP (Partially Observable MDP)
- 鲁棒运动规划
- 机会约束规划

> **推荐课程：** Michigan ROB 520 | CMU 16-833/864 | Northwestern Modern Robotics 课程 4

---

## 六、控制

### 6.1 经典控制

- PID 控制：设计、调参、抗积分饱和
- 频域分析：Bode 图、奈奎斯特稳定性
- 根轨迹法

### 6.2 现代控制

- 状态空间模型
- LQR (Linear Quadratic Regulator)
- 可控性与可观测性
- 状态观测器设计

### 6.3 最优控制

- 动态规划与 Bellman 方程
- Pontryagin 极小值原理
- 模型预测控制 (MPC)：线性与非线性
- 微分动态规划 (DDP)、iLQR

### 6.4 力与阻抗控制

- 阻抗控制与导纳控制
- 混合位置/力控制
- 接触丰富的控制策略
- 柔顺控制

### 6.5 非线性控制

- 反馈线性化
- Lyapunov 稳定性分析
- 自适应控制
- 鲁棒控制（H-infinity）
- 计算力矩控制

### 6.6 欠驱动系统控制

- 欠驱动机械臂
- 足式运动控制（ZMP、倒立摆模型）
- 动态行走与跑步

> **推荐课程：** MIT 6.8210 Underactuated Robotics | CMU 16-745 Nonlinear Optimal Control

---

## 七、学习与智能

### 7.1 概率机器人学

- 贝叶斯滤波器框架
- 卡尔曼滤波族（KF、EKF、UKF）
- 粒子滤波与蒙特卡洛方法
- 高斯过程回归
- 状态估计与概率推理

> **教材：** Thrun, Burgard, Fox — *Probabilistic Robotics*

### 7.2 深度强化学习

#### 核心算法

- **模型无关方法：** PPO、SAC、TD3、DDPG
- **模型基方法：** MBPO、Dreamer、Dyna
- **策略梯度与 Actor-Critic 架构**

#### 在机器人中的应用

- 灵巧操作（Dexterous Manipulation）
- 足式运动控制（Legged Locomotion）
- 自主导航与避障
- 多机器人协作

### 7.3 模仿学习与行为克隆

- 行为克隆 (Behavioral Cloning)
- DAgger 算法
- 逆强化学习 (Inverse RL)
- 从人类演示中学习

### 7.4 Sim-to-Real 迁移

- **域随机化 (Domain Randomization)：** 随机化物理参数和视觉外观
- **域适配 (Domain Adaptation)：** 缩小仿真与真实差距
- **系统辨识 (System Identification)：** 从真实数据校准模型
- **Real-to-Sim-to-Real：** 扫描真实环境进入仿真再返回真实
- **渐进网络与潜在适应**

### 7.5 机器人基础模型

#### 核心方向

| 方向 | 描述 |
|------|------|
| VLA 模型 (Vision-Language-Action) | 视觉-语言-动作模型，LLM 能力延伸到物理执行 |
| 世界模型 (World Models) | 理解和预测环境动态变化的基础模型 |
| 通用操作策略 | 大规模操作基础模型 |
| 语言指令驱动控制 | 自然语言作为机器人交互接口 |

#### 代表性工作

- **RT-2** (Google DeepMind)：视觉-语言-动作模型
- **Octo**：开源通用机器人策略
- **pi0/pi1** (Physical Intelligence)：通用操作基础模型
- **OpenVLA**：开源视觉-语言-动作模型

### 7.6 具身智能 (Embodied AI)

- 具身感知与场景理解
- 具身导航 (Embodied Navigation)
- 具身问答 (Embodied QA)
- 具身操作 (Embodied Manipulation)
- 多模态推理与交互
- 仿真环境：Habitat、AI2-THOR、ThreeDWorld

> **推荐课程：** MIT 6.4210 | Princeton 机器人课程 | Michigan DeepRob

---

## 八、系统工程与实践

### 8.1 ROS 2 开发

- ROS 2 架构（节点、话题、服务、动作）
- 导航栈 (Nav2)
- MoveIt 2 操作框架
- 自定义消息与服务
- Launch 文件与参数管理
- QoS 与实时通信

### 8.2 仿真环境

- **Gazebo (Harmonic+)**：通用物理仿真（新版）
- **Drake**：多体动力学与优化
- **MuJoCo**：高性能物理仿真
- **NVIDIA Isaac Sim**：GPU 加速、大规模并行
- **PyBullet**：轻量级快速原型

### 8.3 嵌入式系统与实时计算

- 实时操作系统 (RTOS)
- 传感器接口与驱动
- 嵌入式编程（C/C++、micro-ROS）
- 硬件抽象层设计

### 8.4 人机交互

- 物理人机交互 (pHRI)
- 安全交互策略
- 共享自主 (Shared Autonomy)
- 社交机器人

### 8.5 机器人安全与伦理

- 功能安全 (ISO 10218、ISO/TS 15066)
- 风险评估与缓解
- AI 安全与可解释性
- 伦理框架与社会影响

---

## 九、前沿专题

### 9.1 人形机器人

- 全身运动控制
- 动态行走与跑步
- 双臂协调操作
- 代表项目：Atlas (Hyundai)、Optimus (Tesla)、Figure、Unitree

### 9.2 软体机器人

- 软体执行器设计
- 连续体机器人运动学
- 柔顺操作
- 变刚度控制

### 9.3 集群机器人

- 多机器人协调与通信
- 群体智能算法
- 分布式任务分配
- 编队控制

### 9.4 手术机器人

- 远程操作 (Teleoperation)
- 精密运动控制
- 医学图像引导
- 触觉反馈

### 9.5 农业与野外机器人

- 非结构化环境导航
- 田间自主操作
- 环境感知与监测

---

## 十、推荐学习路径

### 路径 A：机械/控制方向

```
数学基础 → 运动学 → 动力学 → 经典控制 → 现代控制 → 最优控制 → 力控制
```

适合：机械工程、控制工程背景，目标为运动控制、足式机器人等。

### 路径 B：感知/AI 方向

```
数学基础 → 计算机视觉 → 3D 感知 → 深度学习 → SLAM → 强化学习 → 基础模型
```

适合：计算机科学、AI 背景，目标为自主导航、具身智能等。

### 路径 C：系统/工程方向

```
数学基础 → ROS 2 → 传感器融合 → 运动规划 → 控制集成 → 系统部署
```

适合：嵌入式系统、软件工程背景，目标为机器人系统集成与产品化。

### 路径 D：操作/Manipulation 方向

```
运动学 → 动力学 → 接触力学 → 抓取规划 → 力控制 → 操作学习 → Sim-to-Real
```

适合：目标为机械臂操作、灵巧手、工业自动化等。

---

## 十一、推荐教材与课程

### 核心教材

| 教材 | 作者 | 侧重点 |
|------|------|--------|
| *Modern Robotics: Mechanics, Planning, and Control* | Lynch & Park | 螺旋理论统一框架，运动学/动力学/规划/控制 |
| *Robot Modeling and Control* | Spong, Hutchinson & Vidyasagar | 机械建模与控制入门经典 |
| *Probabilistic Robotics* | Thrun, Burgard, Fox | 概率方法处理不确定性 |
| *Robotics: Modelling, Planning and Control* | Siciliano et al. | 全面覆盖建模、规划、控制、视觉伺服 |
| *Underactuated Robotics* (在线) | Russ Tedrake | 欠驱动系统、最优控制、强化学习 |
| *Robotic Manipulation* (在线) | Russ Tedrake | 操作中的感知、规划与控制 |
| *Rigid Body Dynamics Algorithms* | Roy Featherstone | 正/逆动力学、空间向量代数 |
| *Springer Handbook of Robotics* (2nd Ed.) | Siciliano (Ed.) | 综合参考手册 |

### 推荐在线课程

| 课程 | 院校 | 平台 | 内容 |
|------|------|------|------|
| Modern Robotics (6 门课) | Northwestern | Coursera / YouTube | 完整机器人学体系 |
| 6.4210 Robotic Manipulation | MIT | OCW / 在线教材 | 操作感知规划控制 |
| 6.8210 Underactuated Robotics | MIT | 在线教材 | 欠驱动与最优控制 |
| CS223A Introduction to Robotics | Stanford | YouTube / SEE | 机器人学入门经典 |
| ROB 511 AutoRob | Michigan | autorob.org | 自主移动机器人 |
| 16-745 Nonlinear Optimal Control | CMU | — | 非线性最优控制 |

---

## 十二、关键软件工具

### 中间件与框架

| 工具 | 用途 | 说明 |
|------|------|------|
| **ROS 2** | 机器人操作系统 | 行业标准中间件 |
| **MoveIt 2** | 机械臂运动规划 | ROS 2 下操作标准框架 |
| **Nav2** | 自主导航 | ROS 2 导航栈 |

### 仿真环境

| 工具 | 特点 |
|------|------|
| **Gazebo (Harmonic+)** | 通用物理仿真，与 ROS 2 深度集成 |
| **Drake** | 多体动力学、优化规划，MIT 开发 |
| **MuJoCo** | 高性能物理仿真，GPU 加速训练 |
| **NVIDIA Isaac Sim** | GPU 并行仿真，大规模 RL 训练 |
| **PyBullet** | 轻量级仿真，快速原型验证 |

### 编程与 AI 工具

| 工具 | 用途 |
|------|------|
| **Python** (NumPy, SciPy, OpenCV) | 原型开发、算法实现 |
| **C++** | ROS 2 开发、实时控制 |
| **PyTorch / JAX** | 深度学习与机器人学习 |
| **MATLAB/Simulink** | 控制系统设计与仿真 |
| **Gymnasium** | 强化学习标准接口 |

---

## 参考资源

- [Northwestern Modern Robotics](https://modernrobotics.northwestern.edu) — 课程、教材、代码
- [MIT 6.4210 Robotic Manipulation](http://manipulation.csail.mit.edu) — 在线教材与 Drake 仿真
- [MIT Underactuated Robotics](http://underactuated.mit.edu) — 在线教材
- [Stanford CS223A](https://cs.stanford.edu/groups/manips/teaching/cs223a/) — 经典入门课程
- [Michigan Robotics Courses](https://robotics.umich.edu/academics/courses/complete-course-list/) — 完整课程列表
- [CMU Robotics Institute](https://www.ri.cmu.edu) — 全球最大机器人学院
- [Coursera Modern Robotics Specialization](https://www.coursera.org/specializations/modernrobotics) — 免费视频课程
