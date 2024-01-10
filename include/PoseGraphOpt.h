//
// Created by zjl1 on 2023/9/25.
//

#ifndef POSE_GRAPH_OPT_H_
#define POSE_GRAPH_OPT_H_

#include "open3d/Open3D.h"
#include "iostream"
#include <open3d/geometry/PointCloud.h>

// 命名空间PoseGraphOpt，用于封装与姿态图优化相关的类和函数
namespace PoseGraphOpt
{
// 定义一种数据类型CorrespondenceSet，表示一组对应点，每个对应点由两个整数（向量）表示
typedef std::vector<Eigen::Vector2i> CorrespondenceSet;

// 类PoseGraphNode表示姿态图中的一个节点
class PoseGraphNode
{
public:
    // 构造函数，初始化节点的姿态为单位矩阵（默认姿态）
    PoseGraphNode(const Eigen::Matrix4d &pose = Eigen::Matrix4d::Identity());
    // 析构函数
    ~PoseGraphNode();
public:
    // 节点的姿态，用4x4矩阵表示
    Eigen::Matrix4d_u pose_;
};

// 类PoseGraphEdge表示姿态图中的一条边，用于连接两个节点
class PoseGraphEdge
{
public:
    // 构造函数，初始化边的属性，包括源节点和目标节点的ID，变换矩阵，信息矩阵等
    PoseGraphEdge(
        int source_node_id = -1,
        int target_node_id = -1,
        const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity(),
        const Eigen::Matrix6d &information = Eigen::Matrix6d::Identity(),
        bool uncertain = false,
        double confidence = 1.0);
    // 析构函数
    ~PoseGraphEdge();

public:
    // 源节点ID
    int source_node_id_;
    // 目标节点ID
    int target_node_id_;
    // 边上的变换矩阵，表示从源节点到目标节点的变换
    Eigen::Matrix4d_u transformation_;
    // 信息矩阵，用于表示这条边的精确度或不确定性
    Eigen::Matrix6d_u information_;
    // 标记这条边是否不确定
    bool uncertain_;
    // 这条边的置信度
    double confidence_;
};
// OptOption类用于定义姿态图优化的参数选项
class OptOption
{
public:
    // 构造函数，初始化各种优化参数
    OptOption(double max_correspondence_distance = 0.075,
              double edge_prune_threshold = 0.25,
              double preference_loop_closure = 1.0,
              int reference_node = -1);
    // 析构函数
    ~OptOption();

public:
    // 最大对应距离，用于决定节点间对应点的最大距离
    double max_correspondence_distance_;
    // 边剪枝阈值，用于移除信息量较小的边
    double edge_prune_threshold_;
    // 环路闭合的偏好权重
    double preference_loop_closure_;
    // 参考节点ID，用于相对优化
    int reference_node_;
};

// OptConvergenceCriteria类定义了姿态图优化的收敛标准
class OptConvergenceCriteria{
public:
    // 构造函数，初始化优化的收敛条件
    OptConvergenceCriteria(
        int max_iteration = 100,
        double min_relative_increment = 1e-6,
        double min_relative_residual_increment = 1e-6,
        double min_right_term = 1e-6,
        double min_residual = 1e-6,
        int max_iteration_lm = 20,
        double upper_scale_factor = 2. / 3.,
        double lower_scale_factor = 1. / 3.);
    // 析构函数
    ~OptConvergenceCriteria();

public:
    // 最大迭代次数
    int max_iteration_;
    // 最小相对增量，用于判断是否停止迭代
    double min_relative_increment_;
    // 最小相对残差增量
    double min_relative_residual_increment_;
    // 最小右项，用于Levenberg-Marquardt算法
    double min_right_term_;
    // 最小残差
    double min_residual_;
    // Levenberg-Marquardt算法的最大迭代次数
    int max_iteration_lm_;
    // 上调因子，用于调整Levenberg-Marquardt算法的步长
    double upper_scale_factor_;
    // 下调因子
    double lower_scale_factor_;
};

// RegistrationResult类表示姿态图优化后的结果
class RegistrationResult
{
public:
    // 构造函数，初始化变换矩阵为单位矩阵
    RegistrationResult(
        const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity());
    // 析构函数
    ~RegistrationResult();
public:
    // 变换矩阵，表示从源点云到目标点云的变换
    Eigen::Matrix4d_u transformation_;
    // 对应点集合
    CorrespondenceSet correspondence_set_;
    // 内点的均方根误差
    double inlier_rmse_;
    // 适应度，表示匹配的好坏
    double fitness_;
};

// GetRegistrationResultAndCorrespondencesMy函数用于获取点云之间的配准结果和对应点集
static RegistrationResult GetRegistrationResultAndCorrespondencesMy(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const open3d::geometry::KDTreeFlann &target_kdtree,
    double max_correspondence_distance,
    const Eigen::Matrix4d &transformation);

// GetInformationMatrixFromPointCloudsMy函数用于根据两个点云和它们之间的变换计算信息矩阵
Eigen::Matrix6d GetInformationMatrixFromPointCloudsMy(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    double max_correspondence_distance,
    const Eigen::Matrix4d &transformation,
    double weight_mu) ;

// PoseGraph类用于表示姿态图，包含多个节点和边
class PoseGraph
{
public:
    // 构造函数，初始化姿态图
    PoseGraph();
    // 析构函数
    ~PoseGraph();

public:
    // 存储姿态图的所有节点
    std::vector<PoseGraphNode> nodes_;
    // 存储姿态图的所有边
    std::vector<PoseGraphEdge> edges_;
};

// CreateCirculatePoseGraph函数用于创建循环姿态图
std::shared_ptr<PoseGraph> CreateCirculatePoseGraph(
    const PoseGraph &pose_graph, const OptOption &option);

// GlobalLMOptimization类用于进行全局Levenberg-Marquardt优化
class GlobalLMOptimization
{
public:
    // 构造函数
    GlobalLMOptimization() {}
    // 析构函数
    ~GlobalLMOptimization()  {}

public:
    // OptimizePoseGraph函数用于优化姿态图
    void
    OptimizePoseGraph(
        PoseGraph &pose_graph,
        const OptConvergenceCriteria &criteria,
        const OptOption &option) ;
};

// GlobalOptimizationT函数用于全局优化姿态图，使用Levenberg-Marquardt方法
void GlobalOptimizationT(
    PoseGraph &pose_graph,
    GlobalLMOptimization &method,
    const OptConvergenceCriteria &criteria =
    OptConvergenceCriteria(),
    const OptOption &option = OptOption());

// GlobalOptimizationIRLS函数用于全局优化姿态图，使用迭代重加权最小二乘法（Iteratively Reweighted Least Squares）
void
GlobalOptimizationIRLS(PoseGraph &pose_graph,
                       GlobalLMOptimization &method,
                       const OptConvergenceCriteria &criteria =
                       OptConvergenceCriteria(),
                       const OptOption &option = OptOption());
}

#endif //POSE_GRAPH_OPT_H_
