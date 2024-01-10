//
// Created by zjl1 on 2023/9/25.
//

#include "PoseGraphOpt.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tuple>
#include <vector>
#include "open3d/pipelines/registration/GlobalOptimizationMethod.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Timer.h"
#include <Eigen/Cholesky>


namespace PoseGraphOpt
{


PoseGraphNode::PoseGraphNode(const Eigen::Matrix4d &pose )
    : pose_(pose) {}
PoseGraphNode::~PoseGraphNode(){};


PoseGraphEdge::PoseGraphEdge(
    int source_node_id ,
    int target_node_id ,
    const Eigen::Matrix4d &transformation,
    const Eigen::Matrix6d &information,
    bool uncertain,
    double confidence)
    : source_node_id_(source_node_id),
      target_node_id_(target_node_id),
      transformation_(transformation),
      information_(information),
      uncertain_(uncertain),
      confidence_(confidence) {}
PoseGraphEdge::~PoseGraphEdge() { };

PoseGraph::PoseGraph() {};
PoseGraph::~PoseGraph() {};

OptOption::OptOption(double max_correspondence_distance,
                     double edge_prune_threshold,
                     double preference_loop_closure,
                     int reference_node)
    : max_correspondence_distance_(max_correspondence_distance),
      edge_prune_threshold_(edge_prune_threshold),
      preference_loop_closure_(preference_loop_closure),
      reference_node_(reference_node)
{
    max_correspondence_distance_ = max_correspondence_distance < 0.0
                                   ? 0.075
                                   : max_correspondence_distance;
    edge_prune_threshold_ =
        edge_prune_threshold < 0.0 || edge_prune_threshold > 1.0
        ? 0.25
        : edge_prune_threshold;
    preference_loop_closure_ =
        preference_loop_closure < 0.0 ? 1.0 : preference_loop_closure;
};
OptOption::~OptOption() {}

OptConvergenceCriteria::OptConvergenceCriteria(
    int max_iteration,
    double min_relative_increment,
    double min_relative_residual_increment,
    double min_right_term,
    double min_residual,
    int max_iteration_lm,
    double upper_scale_factor,
    double lower_scale_factor)
    : max_iteration_(max_iteration),
      min_relative_increment_(min_relative_increment),
      min_relative_residual_increment_(min_relative_residual_increment),
      min_right_term_(min_right_term),
      min_residual_(min_residual),
      max_iteration_lm_(max_iteration_lm),
      upper_scale_factor_(upper_scale_factor),
      lower_scale_factor_(lower_scale_factor)
{
    upper_scale_factor_ =
        upper_scale_factor < 0.0 || upper_scale_factor > 1.0
        ? 2. / 3.
        : upper_scale_factor;
    lower_scale_factor_ =
        lower_scale_factor < 0.0 || lower_scale_factor > 1.0
        ? 1. / 3.
        : lower_scale_factor;
};
OptConvergenceCriteria::~OptConvergenceCriteria() {}



RegistrationResult::RegistrationResult(
    const Eigen::Matrix4d &transformation )
    : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}
RegistrationResult::~RegistrationResult() {}


namespace multipath_registration
{

/// Function to solve Ax=b
static std::tuple<bool, Eigen::VectorXd>
MySolveLinearSystemPSD(
    const Eigen::MatrixXd &A,
    const Eigen::VectorXd &b,
    bool prefer_sparse /* = false */,
    bool check_symmetric /* = false */,
    bool check_det /* = false */,
    bool check_psd /* = false */)
{
    // PSD implies symmetric
    check_symmetric = check_symmetric || check_psd;
    if (check_symmetric && !A.isApprox(A.transpose()))
    {
        return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
    }

    if (check_det)
    {
        double det = A.determinant();
        if (fabs(det) < 1e-6 || std::isnan(det) || std::isinf(det))
        {
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    // Check PSD: https://stackoverflow.com/a/54569657/1255535
    if (check_psd)
    {
        Eigen::LLT<Eigen::MatrixXd> A_llt(A);
        if (A_llt.info() == Eigen::NumericalIssue)
        {
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    Eigen::VectorXd x(b.size());

    if (prefer_sparse)
    {
        Eigen::SparseMatrix<double> A_sparse = A.sparseView();
        // TODO: avoid deprecated API SimplicialCholesky
        Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> A_chol;
        A_chol.compute(A_sparse);
        if (A_chol.info() == Eigen::Success)
        {
            x = A_chol.solve(b);
            if (A_chol.info() == Eigen::Success)
            {
                // Both decompose and solve are successful
                return std::make_tuple(true, std::move(x));
            }
            else
            {
            }
        }
        else
        {
        }
    }


//    auto LL = A.llt();
    x = A.llt().solve(b);
//    x = A.ldlt().solve(b);
//    x = A.lu().solve(b);

    return std::make_tuple(true, std::move(x));
}

static const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
    jacobian_operator = {
    // for alpha
    (Eigen::Matrix4d() <<
                       0, 0, 0, 0,
        0, 0, -1, 0,
        0, 1, 0, 0,
        0, 0, 0, 0).finished(),
    // for beta
    (Eigen::Matrix4d() << 0, 0, 1, 0,
        0, 0, 0, 0,
        -1, 0, 0, 0,
        0, 0, 0, 0).finished(),
    // for gamma
    (Eigen::Matrix4d() << 0, -1, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0).finished(),
    // for a
    (Eigen::Matrix4d() << 0, 0, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0).finished(),
    // for b
    (Eigen::Matrix4d() << 0, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0).finished(),
    // for c
    (Eigen::Matrix4d() << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 0, 0, 0).finished()};
// clang-format on

/// This function is intended for linearized form of SE(3).
/// It is an approximate form. See [Choi et al 2015] for derivation.
/// Alternatively, explicit representation that uses quaternion can be used
/// here to replace this function. Refer to linearizeOplus() in
/// https://github.com/RainerKuemmerle/g2o/blob/master/g2o/types/slam3d/edge_se3.cpp
static inline Eigen::Vector6d
GetLinearized6DVector(
    const Eigen::Matrix4d &input)
{
    Eigen::Vector6d output;
    output(0) = (-input(1, 2) + input(2, 1)) / 2.0;
    output(1) = (-input(2, 0) + input(0, 2)) / 2.0;
    output(2) = (-input(0, 1) + input(1, 0)) / 2.0;
    output.block<3, 1>(3, 0) = input.block<3, 1>(0, 3);
    return output;
}

static inline Eigen::Vector6d
GetMisalignmentVector(
    const Eigen::Matrix4d &X_inv,
    const Eigen::Matrix4d &Ts,
    const Eigen::Matrix4d &Tt_inv)
{
    Eigen::Matrix4d temp;
    temp.noalias() = X_inv * Tt_inv * Ts;
    return GetLinearized6DVector(temp);
}

static inline std::tuple<Eigen::Matrix4d, Eigen::Matrix4d, Eigen::Matrix4d>
GetRelativePoses(const PoseGraph &pose_graph, int edge_id)
{
    const PoseGraphEdge &te = pose_graph.edges_[edge_id];
    const PoseGraphNode &ts = pose_graph.nodes_[te.source_node_id_];
    const PoseGraphNode &tt = pose_graph.nodes_[te.target_node_id_];
    Eigen::Matrix4d X_inv = te.transformation_.inverse();
    Eigen::Matrix4d Ts = ts.pose_;
    Eigen::Matrix4d Tt_inv = tt.pose_.inverse();
    return std::make_tuple(std::move(X_inv), std::move(Ts), std::move(Tt_inv));
}

static std::tuple<Eigen::Matrix6d, Eigen::Matrix6d>
GetJacobian(
    const Eigen::Matrix4d &X_inv,
    const Eigen::Matrix4d &Ts,
    const Eigen::Matrix4d &Tt_inv)
{
    Eigen::Matrix6d Js = Eigen::Matrix6d::Zero();
    for (int i = 0; i < 6; i++)
    {
        Eigen::Matrix4d temp = X_inv * Tt_inv * jacobian_operator[i] * Ts;
        Js.block<6, 1>(0, i) = GetLinearized6DVector(temp);
    }
    Eigen::Matrix6d Jt = Eigen::Matrix6d::Zero();
    for (int i = 0; i < 6; i++)
    {
        Eigen::Matrix4d temp = X_inv * Tt_inv * -jacobian_operator[i] * Ts;
        Jt.block<6, 1>(0, i) = GetLinearized6DVector(temp);
    }
    return std::make_tuple(std::move(Js), std::move(Jt));
}

/// Function to update line_process value defined in [Choi et al 2015]
/// See Eq (2). temp2 value in this function is derived from dE/dl = 0
static int
UpdateConfidence(PoseGraph &pose_graph,
                 const Eigen::VectorXd &zeta,
                 const double line_process_weight,
                 const OptOption &option)
{
    int n_edges = (int) pose_graph.edges_.size();
    int valid_edges_num = 0;
    for (int iter_edge = 0; iter_edge < n_edges; iter_edge++)
    {
        PoseGraphEdge &t = pose_graph.edges_[iter_edge];
        if (t.uncertain_)
        {
            Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);
            double residual_square = e.transpose() * t.information_ * e;
            double temp = line_process_weight / (line_process_weight + residual_square);
            double temp2 = temp * temp;
            t.confidence_ = temp2;
            if (temp2 > option.edge_prune_threshold_)
                valid_edges_num++;
        }
    }
    return valid_edges_num;
}

/// Function to compute residual defined in [Choi et al 2015] See Eq (9).
static double
ComputeResidual(const PoseGraph &pose_graph,
                const Eigen::VectorXd &zeta,
                const double line_process_weight,
                const OptOption &option)
{
    int n_edges = (int) pose_graph.edges_.size();
    double residual = 0.0;
    for (int iter_edge = 0; iter_edge < n_edges; iter_edge++)
    {
        const PoseGraphEdge &te = pose_graph.edges_[iter_edge];
        double line_process_iter = te.confidence_;
        Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);
        residual += line_process_iter * e.transpose() * te.information_ * e +
            line_process_weight * pow(sqrt(line_process_iter) - 1, 2.0);
    }
    return residual;
}

/// Function to compute residual defined in [Choi et al 2015] See Eq (6).
static Eigen::VectorXd
ComputeZeta(const PoseGraph &pose_graph)
{
    int n_edges = (int) pose_graph.edges_.size();
    Eigen::VectorXd output(n_edges * 6);
    for (int iter_edge = 0; iter_edge < n_edges; iter_edge++)
    {
        Eigen::Matrix4d X_inv, Ts, Tt_inv;
        std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);
        Eigen::Vector6d e = GetMisalignmentVector(X_inv, Ts, Tt_inv);
        output.block<6, 1>(iter_edge * 6, 0) = e;
    }
    return output;
}


static std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
ComputeLinearSystem(
    const PoseGraph &pose_graph, const Eigen::VectorXd &zeta)
{
    int n_nodes = (int) pose_graph.nodes_.size();
    int n_edges = (int) pose_graph.edges_.size();
    Eigen::MatrixXd H(n_nodes * 6, n_nodes * 6);
    Eigen::VectorXd b(n_nodes * 6);
    H.setZero();
    b.setZero();

    for (int iter_edge = 0; iter_edge < n_edges; iter_edge++)
    {
        const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
        Eigen::Vector6d e = zeta.block<6, 1>(iter_edge * 6, 0);

        Eigen::Matrix4d X_inv, Ts, Tt_inv;
        std::tie(X_inv, Ts, Tt_inv) = GetRelativePoses(pose_graph, iter_edge);

        Eigen::Matrix6d Js, Jt;
        std::tie(Js, Jt) = GetJacobian(X_inv, Ts, Tt_inv);
        Eigen::Matrix6d JsT_Info = Js.transpose() * t.information_;
        Eigen::Matrix6d JtT_Info = Jt.transpose() * t.information_;
        Eigen::Vector6d eT_Info = e.transpose() * t.information_;
        double line_process_iter = t.confidence_;

        int id_i = t.source_node_id_ * 6;
        int id_j = t.target_node_id_ * 6;
        H.block<6, 6>(id_i, id_i).noalias() +=
            line_process_iter * JsT_Info * Js;
        H.block<6, 6>(id_i, id_j).noalias() +=
            line_process_iter * JsT_Info * Jt;
        H.block<6, 6>(id_j, id_i).noalias() +=
            line_process_iter * JtT_Info * Js;
        H.block<6, 6>(id_j, id_j).noalias() +=
            line_process_iter * JtT_Info * Jt;
        b.block<6, 1>(id_i, 0).noalias() -=
            line_process_iter * eT_Info.transpose() * Js;
        b.block<6, 1>(id_j, 0).noalias() -=
            line_process_iter * eT_Info.transpose() * Jt;
    }
    return std::make_tuple(std::move(H), std::move(b));
}

static Eigen::VectorXd
UpdatePoseVector(const PoseGraph &pose_graph)
{
    int n_nodes = (int) pose_graph.nodes_.size();
    Eigen::VectorXd output(n_nodes * 6);
    for (int iter_node = 0; iter_node < n_nodes; iter_node++)
    {
        Eigen::Vector6d output_iter = open3d::utility::TransformMatrix4dToVector6d(
            pose_graph.nodes_[iter_node].pose_);
        output.block<6, 1>(iter_node * 6, 0) = output_iter;
    }
    return output;
}

static std::shared_ptr<PoseGraph>
UpdatePoseGraph(const PoseGraph &pose_graph,
                const Eigen::VectorXd delta)
{
    std::shared_ptr<PoseGraph> pose_graph_updated =
        std::make_shared<PoseGraph>();
    *pose_graph_updated = pose_graph;
    int n_nodes = (int) pose_graph.nodes_.size();
    for (int iter_node = 0; iter_node < n_nodes; iter_node++)
    {
        Eigen::Vector6d delta_iter = delta.block<6, 1>(iter_node * 6, 0);
        pose_graph_updated->nodes_[iter_node].pose_ =
            open3d::utility::TransformVector6dToMatrix4d(delta_iter) *
                pose_graph_updated->nodes_[iter_node].pose_;
    }
    return pose_graph_updated;
}

static bool
CheckRightTerm(
    const Eigen::VectorXd &right_term,
    const OptConvergenceCriteria &criteria)
{
    if (right_term.maxCoeff() < criteria.min_right_term_)
    {
        //utility::LogDebug("Maximum coefficient of right term < {:e}",
//                          criteria.min_right_term_);
        return true;
    }
    return false;
}

static bool
CheckRelativeIncrement(
    const Eigen::VectorXd &delta,
    const Eigen::VectorXd &x,
    const OptConvergenceCriteria &criteria)
{
    if (delta.norm() < criteria.min_relative_increment_ *
        (x.norm() + criteria.min_relative_increment_))
    {
        //utility::LogDebug("Delta.norm() < {:e} * (x.norm() + {:e})",
//                          criteria.min_relative_increment_,
//                          criteria.min_relative_increment_);
        return true;
    }
    return false;
}

static bool
CheckRelativeResidualIncrement(
    double current_residual,
    double new_residual,
    const OptConvergenceCriteria &criteria)
{
    if (current_residual - new_residual <
        criteria.min_relative_residual_increment_ * current_residual)
    {
        //utility::LogDebug(
//            "Current_residual - new_residual < {:e} * current_residual",
//            criteria.min_relative_residual_increment_);
        return true;
    }
    return false;
}

static bool
CheckResidual(
    double residual,
    const OptConvergenceCriteria &criteria)
{
    if (residual < criteria.min_residual_)
    {
        //utility::LogDebug("Current_residual < {:e}", criteria.min_residual_);
        return true;
    }
    return false;
}

static bool
CheckMaxIteration(
    int iteration, const OptConvergenceCriteria &criteria)
{
    if (iteration >= criteria.max_iteration_)
    {
        //utility::LogDebug("Reached maximum number of iterations ({:d})",
//                          criteria.max_iteration_);
        return true;
    }
    return false;
}

static bool
CheckMaxIterationLM(
    int iteration, const OptConvergenceCriteria &criteria)
{
    if (iteration >= criteria.max_iteration_lm_)
    {
        //utility::LogDebug("Reached maximum number of iterations ({:d})",
//                          criteria.max_iteration_lm_);
        return true;
    }
    return false;
}

static double
ComputeLineProcessWeight(const PoseGraph &pose_graph,
                         const OptOption &option)
{
    int n_edges = (int) pose_graph.edges_.size();
    double average_number_of_correspondences = 0.0;

// 一个edge就是一个约束，其中有相邻的帧的约束也有回环约束
// 先计算平均有多少个匹配点对
    for (int iter_edge = 0; iter_edge < n_edges; iter_edge++)
    {
        double number_of_correspondences =
            pose_graph.edges_[iter_edge].information_(5, 5);
        average_number_of_correspondences += number_of_correspondences;
    }


    if (n_edges > 0)
    {
        // see Section 5 in [Choi et al 2015]
        average_number_of_correspondences /= (double) n_edges;
        double line_process_weight =
            option.preference_loop_closure_ *
                pow(option.max_correspondence_distance_, 2) *
                average_number_of_correspondences;  //    1 * 20mm * n_avg_ptx
        return line_process_weight;
    }
    else
    {
        return 0.0;
    }
}

static void
CompensateReferencePoseGraphNode(PoseGraph &pose_graph_new,
                                 const PoseGraph &pose_graph_orig,
                                 int reference_node)
{
    //utility::LogDebug("CompensateReferencePoseGraphNode : reference : {:d}",
//                      reference_node);
    int n_nodes = (int) pose_graph_new.nodes_.size();
    if (reference_node < 0 || reference_node >= n_nodes)
    {
        return;
    }
    else
    {
        Eigen::Matrix4d compensation =
            pose_graph_orig.nodes_[reference_node].pose_ *
                pose_graph_new.nodes_[reference_node].pose_.inverse();
        for (int i = 0; i < n_nodes; i++)
        {
            pose_graph_new.nodes_[i].pose_ =
                compensation * pose_graph_new.nodes_[i].pose_;
        }
    }
}

static bool
ValidatePoseGraphConnectivity(const PoseGraph &pose_graph,
                              bool ignore_uncertain_edges = false)
{
    size_t n_nodes = pose_graph.nodes_.size();
    size_t n_edges = pose_graph.edges_.size();

    // Test if the connected component containing the first node is the entire
    // graph
    std::vector<int> nodes_to_explore{};
    std::vector<int> component{};
    if (n_nodes > 0)
    {
        nodes_to_explore.push_back(0);
        component.push_back(0);
    }
    while (!nodes_to_explore.empty())
    {
        int i = nodes_to_explore.back();
        nodes_to_explore.pop_back();
        for (size_t j = 0; j < n_edges; j++)
        {
            const PoseGraphEdge &t = pose_graph.edges_[j];
            if (ignore_uncertain_edges && t.uncertain_)
            {
                continue;
            }
            int adjacent_node{-1};
            if (t.source_node_id_ == i)
            {
                adjacent_node = t.target_node_id_;
            }
            else if (t.target_node_id_ == i)
            {
                adjacent_node = t.source_node_id_;
            }
            if (adjacent_node != -1)
            {
                auto find_result = std::find(component.begin(), component.end(),
                                             adjacent_node);
                if (find_result == component.end())
                {
                    nodes_to_explore.push_back(adjacent_node);
                    component.push_back(adjacent_node);
                }
            }
        }
    }
    return component.size() == n_nodes;
}

static bool
ValidatePoseGraph(const PoseGraph &pose_graph)
{
    int n_nodes = (int) pose_graph.nodes_.size();
    int n_edges = (int) pose_graph.edges_.size();

    if (!ValidatePoseGraphConnectivity(pose_graph, false))
    {
        open3d::utility::LogWarning("Invalid PoseGraph - graph is not connected.");
        return false;
    }

    if (!ValidatePoseGraphConnectivity(pose_graph, true))
    {
        open3d::utility::LogWarning(
            "Certain-edge subset of PoseGraph is not connected.");
    }

    for (int j = 0; j < n_edges; j++)
    {
        bool valid = false;
        const PoseGraphEdge &t = pose_graph.edges_[j];
        if (t.source_node_id_ >= 0 && t.source_node_id_ < n_nodes &&
            t.target_node_id_ >= 0 && t.target_node_id_ < n_nodes)
            valid = true;
        if (!valid)
        {
            open3d::utility::LogWarning(
                "Invalid PoseGraph - an edge references an invalid "
                "node.");
            return false;
        }
    }
    for (int j = 0; j < n_edges; j++)
    {
        const PoseGraphEdge &t = pose_graph.edges_[j];
        if (!t.uncertain_ && t.confidence_ != 1.0)
        {
            open3d::utility::LogWarning(
                "Invalid PoseGraph - the certain edge does not have 1.0 as "
                "a confidence.");
            return false;
        }
    }
    //utility::LogDebug("Validating PoseGraph - finished.");
    return true;
}

std::shared_ptr<PoseGraph>
CreatePoseGraphWithoutInvalidEdges(
    const PoseGraph &pose_graph, const OptOption &option)
{
    std::shared_ptr<PoseGraph> pose_graph_pruned =
        std::make_shared<PoseGraph>();

    int n_nodes = (int) pose_graph.nodes_.size();
    for (int iter_node = 0; iter_node < n_nodes; iter_node++)
    {
        const PoseGraphNode &t = pose_graph.nodes_[iter_node];
        pose_graph_pruned->nodes_.push_back(t);
    }
    int n_edges = (int) pose_graph.edges_.size();
    for (int iter_edge = 0; iter_edge < n_edges; iter_edge++)
    {
        const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
        if (t.uncertain_)
        {
            if (t.confidence_ > option.edge_prune_threshold_)
            {
                pose_graph_pruned->edges_.push_back(t);
            }
        }
        else
        {
            pose_graph_pruned->edges_.push_back(t);
        }
    }
    return pose_graph_pruned;
}
}

void
GlobalOptimizationT(PoseGraph &pose_graph,
                    GlobalLMOptimization &method
    /* = GlobalOptimizationLevenbergMarquardt() */,
                    const OptConvergenceCriteria &criteria
    /* = OptConvergenceCriteria() */,
                    const OptOption &option
    /* = OptOption() */)
{
    if (!multipath_registration::ValidatePoseGraph(pose_graph))
        return;
    std::shared_ptr<PoseGraph> pose_graph_pre = std::make_shared<PoseGraph>();

    //    todo stop 1: 先进行初始优化
    *pose_graph_pre = pose_graph;
    method.OptimizePoseGraph(*pose_graph_pre, criteria, option);

    auto pose_graph_pre_pruned_2 =
        CreateCirculatePoseGraph(*pose_graph_pre, option);

    //    todo stop 2： 输出解结果
    multipath_registration::CompensateReferencePoseGraphNode(*pose_graph_pre_pruned_2, pose_graph,
                                                             option.reference_node_);
    pose_graph = *pose_graph_pre_pruned_2;
}

RegistrationResult
GetRegistrationResultAndCorrespondencesMy(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    const open3d::geometry::KDTreeFlann &target_kdtree,
    double max_correspondence_distance,
    const Eigen::Matrix4d &transformation)
{
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0)
    {
        return result;
    }

    double error2 = 0.0;

#pragma omp parallel
    {
        double error2_private = 0.0;
        CorrespondenceSet correspondence_set_private;
#pragma omp for nowait
        for (int i = 0; i < (int) source.points_.size(); i++)
        {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto &point = source.points_[i];
            if (target_kdtree.SearchHybrid(point, max_correspondence_distance,
                                           1, indices, dists) > 0)
            {
                error2_private += dists[0];
                correspondence_set_private.push_back(
                    Eigen::Vector2i(i, indices[0]));
            }
        }
#pragma omp critical(GetRegistrationResultAndCorrespondencesMy)
        {
            for (int i = 0; i < (int) correspondence_set_private.size(); i++)
            {
                result.correspondence_set_.push_back(
                    correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
    }

    if (result.correspondence_set_.empty())
    {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    }
    else
    {
        size_t corres_number = result.correspondence_set_.size();
        result.fitness_ = (double) corres_number / (double) source.points_.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double) corres_number);
    }
    return result;
}

Eigen::Matrix6d
GetInformationMatrixFromPointCloudsMy(
    const open3d::geometry::PointCloud &source,
    const open3d::geometry::PointCloud &target,
    double max_correspondence_distance,
    const Eigen::Matrix4d &transformation,
    double weight_mu)
{
    open3d::geometry::PointCloud pcd = source;
    if (!transformation.isIdentity())
    {
        pcd.Transform(transformation);
    }
    RegistrationResult result;
    open3d::geometry::KDTreeFlann target_kdtree(target);
    result = GetRegistrationResultAndCorrespondencesMy(
        pcd, target, target_kdtree, max_correspondence_distance,
        transformation);

    Eigen::Matrix6d GTG = Eigen::Matrix6d::Zero();
#pragma omp parallel
    {
        Eigen::Matrix6d GTG_private = Eigen::Matrix6d::Zero();
        Eigen::Vector6d G_r_private = Eigen::Vector6d::Zero();
#pragma omp for nowait
        for (int c = 0; c < int(result.correspondence_set_.size()); c++)
        {
            int t = result.correspondence_set_[c](1);
            int s = result.correspondence_set_[c](0);
            double x = target.points_[t](0);
            double y = target.points_[t](1);
            double z = target.points_[t](2);

            double weight = 1;
            if (weight_mu > 0)
            {
                double color_diff = (target.colors_[t] - source.colors_[s]).norm();
                weight = exp(-(color_diff * color_diff) / (2.0 * weight_mu * weight_mu));
            }
//            std::cout<<color_diff<<std::endl;

            G_r_private.setZero();
            G_r_private(1) = z * weight;
            G_r_private(2) = -y * weight;
            G_r_private(3) = 1.0 * weight;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = -z * weight;
            G_r_private(2) = x * weight;
            G_r_private(4) = 1.0 * weight;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = y * weight;
            G_r_private(1) = -x * weight;
            G_r_private(5) = 1.0 * weight;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
        }
#pragma omp critical(GetInformationMatrixFromPointCloudsMy)
        { GTG += GTG_private; }
    }
    return GTG;
}

void
GlobalOptimizationIRLS(PoseGraph &pose_graph,
                       GlobalLMOptimization &method,
                       const OptConvergenceCriteria &criteria,
                       const OptOption &option)
{
    if (!multipath_registration::ValidatePoseGraph(pose_graph))
        return;

    std::shared_ptr<PoseGraph> pose_graph_pre = std::make_shared<PoseGraph>();

    //    todo stop 1: 先进行初始优化
    *pose_graph_pre = pose_graph;
    method.OptimizePoseGraph(*pose_graph_pre, criteria, option);

//    std::cout << "===============================================================" << std::endl;
//    std::cout << "pose_graph_pre->edges_.size():" << pose_graph_pre->edges_.size() << std::endl;
    //    todo stop 2: 修减掉不好的边，继续优化
    auto pose_graph_pre_pruned =
        CreateCirculatePoseGraph(*pose_graph_pre, option);


    method.OptimizePoseGraph(*pose_graph_pre_pruned, criteria, option);

    auto pose_graph_pre_pruned_2 =
        CreateCirculatePoseGraph(*pose_graph_pre_pruned, option);

    //    todo stop 3： 输出解结果
    multipath_registration::CompensateReferencePoseGraphNode(*pose_graph_pre_pruned_2, pose_graph,
                                                             option.reference_node_);

//    std::cout << "pose_graph_pre_pruned_2->edges_.size():" << pose_graph_pre_pruned_2->edges_.size() << std::endl;
//    std::cout << "===============================================================" << std::endl;

    pose_graph = *pose_graph_pre_pruned_2;
}

std::shared_ptr<PoseGraph>
CreateCirculatePoseGraph(
    const PoseGraph &pose_graph, const OptOption &option)
{
    std::shared_ptr<PoseGraph> pose_graph_pruned =
        std::make_shared<PoseGraph>();
//node 直接拷贝
    int n_nodes = (int) pose_graph.nodes_.size();
    for (int iter_node = 0; iter_node < n_nodes; iter_node++)
    {
        const PoseGraphNode &t = pose_graph.nodes_[iter_node];
        pose_graph_pruned->nodes_.push_back(t);
    }
//edge 修剪
    int n_edges = (int) pose_graph.edges_.size();
    for (int iter_edge = 0; iter_edge < n_edges; iter_edge++)
    {
        const PoseGraphEdge &t = pose_graph.edges_[iter_edge];
        if (t.uncertain_)
        {
            if (t.confidence_ > option.edge_prune_threshold_)
            {
                pose_graph_pruned->edges_.push_back(t);
            }
        }
        else
        {
            pose_graph_pruned->edges_.push_back(t);
        }
    }
    return pose_graph_pruned;
}

void
GlobalLMOptimization::OptimizePoseGraph(
    PoseGraph &pose_graph,
    const OptConvergenceCriteria &criteria,
    const OptOption &option)
{
    int n_nodes = (int) pose_graph.nodes_.size();
    int n_edges = (int) pose_graph.edges_.size();
    double line_process_weight = multipath_registration::ComputeLineProcessWeight(pose_graph, option);

    //todo step 1: 计算当前位姿图下的中间变量zeta
    Eigen::VectorXd zeta = multipath_registration::ComputeZeta(pose_graph);
    double current_residual, new_residual;

    //todo step 2: 计算当前位姿图下的残差，更新边的置信度
    new_residual =
        multipath_registration::ComputeResidual(pose_graph, zeta, line_process_weight, option);
    current_residual = new_residual;

    int valid_edges_num =
        multipath_registration::UpdateConfidence(pose_graph, zeta, line_process_weight, option);

    Eigen::MatrixXd H_I = Eigen::MatrixXd::Identity(n_nodes * 6, n_nodes * 6);
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    Eigen::VectorXd x = multipath_registration::UpdatePoseVector(pose_graph);

    //todo step 3: 计算雅可比矩阵，构建第一次线性方程组
    std::tie(H, b) = multipath_registration::ComputeLinearSystem(pose_graph, zeta);

    Eigen::VectorXd H_diag = H.diagonal();
    double tau = 1e-5;
    double current_lambda = tau * H_diag.maxCoeff();
    double ni = 2.0;
    double rho = 0.1;

    bool stop = false;
    stop = stop || multipath_registration::CheckRightTerm(b, criteria);
    if (stop)
        return;

    for (int iter = 0; !stop; iter++)
    {
        open3d::utility::Timer timer_iter;
        timer_iter.Start();
        int lm_count = 0;

        do
        {
            Eigen::MatrixXd H_LM = H + current_lambda * H_I;
            Eigen::VectorXd delta(H_LM.cols());
            bool solver_success = false;

            // Solve H_LM @ delta == b using a sparse solver
            //todo step 1: 计算增量delta
            std::tie(solver_success, delta) = multipath_registration::MySolveLinearSystemPSD(
                H_LM, b, /*prefer_sparse=*/true, /*check_symmetric=*/false,
                /*check_det=*/false, /*check_psd=*/false);

            stop = stop || multipath_registration::CheckRelativeIncrement(delta, x, criteria);
            if (!stop)
            {
                //todo step 2： 更新位姿图
                std::shared_ptr<PoseGraph> pose_graph_new =
                    multipath_registration::UpdatePoseGraph(pose_graph, delta);

                //todo step 3： 计算新位姿图下的zeta，新的残差
                Eigen::VectorXd zeta_new;
                zeta_new = multipath_registration::ComputeZeta(*pose_graph_new);
                new_residual = multipath_registration::ComputeResidual(pose_graph, zeta_new,
                                                                       line_process_weight, option);

                rho = (current_residual - new_residual) /
                    (delta.dot(current_lambda * delta + b) + 1e-3);

                //todo step 4：通过残差，判断是否接受新的位姿图
                if (rho > 0)
                {  //满足，更新位姿
                    stop = stop ||
                        multipath_registration::CheckRelativeResidualIncrement(
                            current_residual, new_residual, criteria);
                    if (stop)
                        break;
                    double alpha = 1. - pow((2 * rho - 1), 3);
                    alpha = (std::min)(alpha, criteria.upper_scale_factor_);
                    double scaleFactor =
                        (std::max)(criteria.lower_scale_factor_, alpha);
                    current_lambda *= scaleFactor;
                    ni = 2;
                    current_residual = new_residual;

                    //todo step 5：更新中间变量zeta，位姿图，位姿向量，边的置信度，雅可比矩阵
                    zeta = zeta_new;
                    pose_graph = *pose_graph_new;
                    x = multipath_registration::UpdatePoseVector(pose_graph);
                    valid_edges_num = multipath_registration::UpdateConfidence(
                        pose_graph, zeta, line_process_weight, option);
                    std::tie(H, b) = multipath_registration::ComputeLinearSystem(pose_graph, zeta);

                    stop = stop || multipath_registration::CheckRightTerm(b, criteria);
                    if (stop)
                        break;
                }
                else
                {
                    current_lambda *= ni;  //不满足，回退
                    ni *= 2;
                }
            }
            lm_count++;
            stop = stop || multipath_registration::CheckMaxIterationLM(lm_count, criteria);
        }
        while (!((rho > 0) || stop));
        timer_iter.Stop();
        if (!stop)
        {
            open3d::utility::LogDebug(
                "[Iteration {:02d}] residual : {:e}, valid edges : {:d}, "
                "time : "
                "{:.3f} sec.",
                iter, current_residual, valid_edges_num,
                timer_iter.GetDurationInSecond());
        }
        stop = stop || multipath_registration::CheckResidual(current_residual, criteria) ||
            multipath_registration::CheckMaxIteration(iter, criteria);
    }
}

}
