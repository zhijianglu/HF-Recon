//
// Created by zjl1 on 2023/7/22.
//

#include <chrono>
#include <unsupported/Eigen/MatrixFunctions>
#include "PairWiseAlign.h"

#include <cstdio>
#include <memory>
#include <open3d/geometry/KDTreeSearchParam.h>
#include <open3d/io/PointCloudIO.h>
#include <string>
#include <memory.h>
#include "file_tool.h"
#include "getfile.h"
#include "tic_toc.h"

#include <iostream>
#include <random>
#include <Eigen/Dense>

#include "open3d/pipelines/registration/GlobalOptimization.h"

#include "open3d/geometry/PointCloud.h"

//添加open3d聚类头文件
#include <open3d/geometry/KDTreeFlann.h>
#include <open3d/geometry/Octree.h>
#include <random>
#include "tools.h"
#include "PoseGraphOpt.h"
#include "O3dVis.h"
#include "regis_funs.h"
#include "meshrecon_funs.h"
#include "global.h"

using namespace std;

void
pre_process(string path, double voxel_size, Matrix4d &prior_pose, Matrix4d &gt_pose, std::shared_ptr<open3d::geometry::PointCloud>&pcd)
{
    pcd.reset(new open3d::geometry::PointCloud);
    cout << "reading point cloud: " << path + "/point_cloud.pcd" << endl;
    if (!open3d::io::ReadPointCloud(path + "/point_cloud.pcd", *pcd))
    {
        std::cout << "无法读取点云文件：" << path + "/point_cloud.pcd" << std::endl;
        return;
    }

    if (CfgParam.is_add_noise)
    {
        // 设置随机数引擎和正态分布对象
        std::mt19937 gen(0); // Mersenne Twister 随机数引擎
        std::normal_distribution<double> distribution(0.0, 1.0); // 均值为0，标准差为1的正态分布

        auto &S = CfgParam.noise_sigma_range;  //y
        auto &D = CfgParam.noise_distance_range;  //x
        double K = (S[1] - S[0]) / (D[1] - D[0]);
        double b = S[0] - K * D[0];

        for (auto &point: pcd->points_)
        {
            // 计算点到相机的距离
            double distance = point.norm();

            // 生成噪声值并乘以距离
            double noise = (K * distance + b) * distribution(gen);
            // 将噪声添加到点的位置（沿相机视角方向）
            point += (noise * point.normalized());
        }
    }

    open3d::io::WritePointCloud(path + "/noise_point_cloud.pcd", *pcd);

    // 打开文本文件
    std::ifstream file(path + "/coarse_trans2next.txt");
    if (!file.is_open())
    {
        std::cout << "无法打开文件" << std::endl;
        prior_pose = Matrix4d::Identity();
    }
    else
    {
        // 从文件中读取矩阵数据并赋值给Matrix4f对象
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                file >> prior_pose(i, j);
            }
        }
    }
    cout << "prior_pose:" << path << endl << prior_pose << endl;


    //读取yaml参数
    std::string yaml_path = path + "/cam_info.yaml";
    cv::FileStorage fsSettings(yaml_path, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }
    cv::Mat depth2base;
    fsSettings["depth2base"] >> depth2base;
    //转换成Eigen
    gt_pose.setIdentity();
    cv::cv2eigen(depth2base, gt_pose);


    pcd = pcd->VoxelDownSample(voxel_size);
    pcd->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * CfgParam.scale_normal_radius,
                                                                   CfgParam.max_n_normal_pt));
    //设置半径及最大k临近点
    pcd->OrientNormalsTowardsCameraLocation();
//        auto pcd_f = open3d::pipelines::registration::ComputeFPFHFeature(
//            *pcd, open3d::geometry::KDTreeSearchParamHybrid(10, 30));
//        pcd->EstimateCovariances(open3d::geometry::KDTreeSearchParamHybrid(10, 30));
    return;
}

double ComputeRMSE(const open3d::geometry::PointCloud& pc1, const open3d::geometry::PointCloud& pc2) {
    if (pc1.points_.size() != pc2.points_.size()) {
        cerr << "Error: Point clouds are not of the same size." << endl;
        return -1;
    }

    double mse = 0.0;
    for (size_t i = 0; i < pc1.points_.size(); ++i) {
        mse += (pc1.points_[i] - pc2.points_[i]).norm();
    }
    mse /= pc1.points_.size();
    return sqrt(mse);
}

double ComputeVariance(const open3d::geometry::PointCloud& pc1, const open3d::geometry::PointCloud& pc2, double mean) {
    if (pc1.points_.size() != pc2.points_.size()) {
        cerr << "Error: Point clouds are not of the same size." << endl;
        return -1;
    }

    double variance = 0.0;
    for (size_t i = 0; i < pc1.points_.size(); ++i) {
        double distance = (pc1.points_[i] - pc2.points_[i]).norm();
        variance += (distance - mean) * (distance - mean);
    }
    variance /= pc1.points_.size();
    return variance;
}


void
record_error_logs(vector<Eigen::Matrix4d> est, vector<Eigen::Matrix4d> gt, vector<std::shared_ptr<open3d::geometry::PointCloud>> &pc, string out_file)
{
    ofstream error_log_(out_file);
    for (int pointID = 1; pointID < est.size(); ++pointID)
    {
        Eigen::Matrix4d T_pose = est[pointID];

        open3d::geometry::PointCloud pc_est = *pc[pointID];
        open3d::geometry::PointCloud pc_gt = *pc[pointID];

        pc_est.Transform(T_pose);
        pc_gt.Transform(gt[pointID]);

        double rmse = ComputeRMSE(pc_est, pc_gt);
        double variance = ComputeVariance(pc_est, pc_gt, rmse);

        //T_pose 转换成四元数
        Eigen::Matrix3d R_pose = T_pose.block(0, 0, 3, 3);
        Eigen::Quaterniond q_pose(R_pose);
        Eigen::Vector3d t_pose = T_pose.block(0, 3, 3, 1);
        Eigen::Vector2f extrinsic_error = evoPose(T_pose, gt[pointID]);  // ( r, t )
        error_log_ << std::fixed << std::setprecision(8) << extrinsic_error.x() << " " << extrinsic_error.y() << " " << rmse << " "<< variance << endl;
    }
    error_log_.close();
}

void
record_pose(vector<Eigen::Matrix4d> est, string out_file)
{
    ofstream pose_log_(out_file);
    for (int pointID = 0; pointID < est.size(); ++pointID)
    {
        Eigen::Matrix4d T_pose = est[pointID];
        //T_pose 转换成四元数
        Eigen::Matrix3d R_pose = T_pose.block(0, 0, 3, 3);
        Eigen::Quaterniond q_pose(R_pose);
        Eigen::Vector3d t_pose = T_pose.block(0, 3, 3, 1);
        pose_log_ << pointID << " " << t_pose.x() << " " << t_pose.y() << " " << t_pose.z() << " " << q_pose.x() << " " << q_pose.y() << " "
                  << q_pose.z() << " " << q_pose.w() << endl;
    }
    pose_log_.close();
}

void
align_all(string test_data_root_path, string out_dir)
{

    //获取数据
    vector<string> v_data_dirs_tmp;
    vector<string> v_data_dirs;
    get_file_list(test_data_root_path, v_data_dirs_tmp, 0, CfgParam.excluded_dir);
    v_data_dirs = vector<string>(v_data_dirs_tmp.begin() + CfgParam.start_idx, v_data_dirs_tmp.end());

    // 创建一个可视化窗口
    O3dVis *o3d_vis;

    //创建点云，初始化变量
    vector<std::shared_ptr<open3d::geometry::PointCloud>> pcs(v_data_dirs.size());
    vector<Eigen::Matrix4d> v_init_trans2next(v_data_dirs.size());
    vector<Eigen::Matrix4d> v_gt_trans2base(v_data_dirs.size());
    vector<Eigen::Matrix4d> v_pairwise_trans2base(v_data_dirs.size());
    vector<Eigen::Matrix4d> v_global_trans2next(v_data_dirs.size());
    vector<vector<Eigen::Matrix4d>> v_iter_poses;
    //to read the transform 2 base or calculate from neight frame pose;

    vector<std::thread> preprocess_threads(v_data_dirs.size());
    for (int i = 0; i < v_data_dirs.size(); ++i)
        preprocess_threads[i] = std::thread(pre_process, v_data_dirs[i], CfgParam.voxel_size, std::ref(v_init_trans2next[i]), std::ref(v_gt_trans2base[i]), std::ref(pcs[i]));
    for (int i = 0; i < preprocess_threads.size(); ++i)
        preprocess_threads[i].join();
    //load pose 2 base
    //------------------global align
    //----- 2. Full Registration -----//
    std::cout << '\n' << "[ 2. Full Registration: ]" << std::endl;

    if (CfgParam.show_pairalign)
        o3d_vis = new O3dVis();
    else
        o3d_vis = nullptr;

//    auto pose_graph = loop_registration_global(pcs, max_correspondence_distance_coarse, max_correspondence_distance_fine, &o3d_vis, pc_poses_to_base, v_data_dirs);
    loop_registration_global_parallel(pcs, CfgParam.show_pairalign ? o3d_vis : nullptr,
                                      v_init_trans2next,
                                      v_pairwise_trans2base,
                                      v_global_trans2next,
                                      v_iter_poses);

    record_pose(v_iter_poses[0], out_dir + "/teaser_initial_pose.txt");
    record_error_logs(v_iter_poses[0], v_gt_trans2base, pcs, out_dir + "/teaser_initial_pose.txt");

    record_pose(v_pairwise_trans2base, out_dir + "/pairwise_poses.txt");
    record_error_logs(v_pairwise_trans2base, v_gt_trans2base, pcs, out_dir + "/error_pairwise_it.txt");

    record_pose(v_global_trans2next, out_dir + "/globopt_poses.txt");
    for (int i = 0; i < CfgParam.global_opt_itertime; ++i)
        record_error_logs(v_iter_poses[i + 2], v_gt_trans2base, pcs, out_dir + "/error_globopt_it" + to_string(i+1) + ".txt");

    //  将点云累加起来
    std::shared_ptr<open3d::geometry::PointCloud> pc_all_glob_opt(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pc_all_pairwise(new open3d::geometry::PointCloud);

    if (CfgParam.final_align_spin != 0)
        o3d_vis = new O3dVis();

    for (int pointID = 0; pointID < pcs.size(); ++pointID)
    {
        Eigen::Matrix4d T_pose_opt = v_global_trans2next[pointID];
        Eigen::Matrix4d T_pose_pairwise = v_pairwise_trans2base[pointID];

        std::shared_ptr<open3d::geometry::PointCloud> pc_show_opt(new open3d::geometry::PointCloud(*pcs[pointID]));
        std::shared_ptr<open3d::geometry::PointCloud> pc_show_pairwise(new open3d::geometry::PointCloud(*pcs[pointID]));

        pc_show_opt->Transform(T_pose_opt);
        pc_show_pairwise->Transform(T_pose_pairwise);

        *pc_all_glob_opt += *pc_show_opt;
        *pc_all_pairwise += *pc_show_pairwise;
        // 可视化点云
        if (CfgParam.final_align_spin > 0)
            o3d_vis->addPointCloudShow(pc_show_opt, CfgParam.final_align_spin, pointID == 0);
    }



    //保存点云
    create_dir(out_dir + "/models", false);
    open3d::io::WritePointCloud(out_dir + "/models/glob_opt_align.ply", *pc_all_glob_opt);
    open3d::io::WritePointCloud(out_dir + "/models/pairwise_align.ply", *pc_all_pairwise);
    std::shared_ptr<open3d::geometry::PointCloud> pc_norm_checked(new open3d::geometry::PointCloud);
    norm_check(pc_all_glob_opt, pc_norm_checked, out_dir);
    open3d::io::WritePointCloud(out_dir + "/models/final_glob_opt_align.ply", *pc_norm_checked);

    // 重建网格
    if (CfgParam.apply_mesh_recon)
    {
        //保存结果数据对比
//        pc_all_glob_opt->colors_.clear();
//        pc_all_pairwise->colors_.clear();
        auto trimmed_mesh = mesh_reconstruction(pc_norm_checked);
        // 你可能还需要重新计算网格的面（这部分代码取决于你具体的需求和数据）
        // 输出修剪后的网格
        if (open3d::io::WriteTriangleMesh(out_dir + "/models/poison_triang.ply", *trimmed_mesh )) {
            std::cout << "Trimmed mesh has been saved." << std::endl;
        } else {
            std::cerr << "Failed to save trimmed mesh." << std::endl;
        }

        load_textures_simdata(v_data_dirs, v_global_trans2next, Eigen::Matrix4d::Identity(), out_dir + "/models");
    }
}




int
main(int argc, char *argv[])
{
    string input_config_file = argv[1];

    //prepare params
    readParameters(input_config_file);
    double noise_scale = (CfgParam.noise_sigma_range[1] + CfgParam.noise_sigma_range[0]) / 2.0;

    if(CfgParam.is_add_noise)
    {
        std::ostringstream streamObj;
        streamObj << std::fixed << std::setprecision(2) << noise_scale;
        std::string str = streamObj.str();
        CfgParam.data_label = CfgParam.data_label + "-noise-" + str;
    }

    TicToc timer_global;
    TicToc timer;
    vector<std::thread> test_threads;
    for (int data_id = 0; data_id < CfgParam.test_data_list.size(); ++data_id)
    {
        string test_data_root_path = CfgParam.data_root_path + "/" + CfgParam.test_data_list[data_id];
        //创建输出文件夹
        create_dir(test_data_root_path + "/results", false);
        string out_dir = test_data_root_path + "/results/" + CfgParam.pairwise_reg_method + "-" + CfgParam.data_label;

        create_dir(out_dir, false);

        string output_config_log = out_dir + "/config_log.yaml";
        string command = "cp " + input_config_file + " " + output_config_log;
        system(command.c_str());
        test_threads.emplace_back(align_all, test_data_root_path, out_dir);
    }

    for (int i = 0; i < test_threads.size(); ++i)
    {
        test_threads[i].join();
    }

    cout << "method " << CfgParam.pairwise_reg_method << " total time cost: " << timer.toc() / 60000.0 << " min" << endl;

    cout << "all test  total time cost: " << timer_global.toc() / 60000.0 << " min" << endl;
    return 0;
}
