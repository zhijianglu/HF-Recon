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
#include "PinholeCam.h"
using namespace std;

void read_pose(string pose_file_path, vector<Eigen::Matrix4d> &poses){
    Eigen::Matrix4d prior_pose = Matrix4d::Identity();
    std::ifstream infile(pose_file_path);
    string s;
    while (!infile.eof())
    {
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        double idx;
        infile >> idx >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >> q.w();
//        cout << idx << ":" << t.transpose() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        prior_pose.setIdentity();
        prior_pose.block(0, 0, 3, 3) = q.toRotationMatrix();
        prior_pose.block(0, 3, 3, 1) = t;
        poses.push_back(prior_pose);
    }
    infile.close();
}

void read_gt_pose(vector<string> v_path, vector<Matrix4d> &v_gt_pose)
{
    for (int i = 0; i < v_path.size(); ++i)
    {
        string yaml_path = v_path[i] + "/cam_info.yaml";
        cv::FileStorage fsSettings(yaml_path, cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            std::cerr << "ERROR: Wrong path to settings" << std::endl;
            return;
        }
        cv::Mat depth2base;
        fsSettings["depth2base"] >> depth2base;
        //转换成Eigen
        Matrix4d gt_pose;
        gt_pose.setIdentity();
        cv::cv2eigen(depth2base, gt_pose);

        v_gt_pose.push_back(gt_pose);
    }
}

void
display_multiview_gt(string test_data_root_path,  O3dVis *o3d_vis)
{
    //获取数据
    vector<string> v_data_dirs_tmp;
    vector<string> v_data_dirs;
    get_file_list(test_data_root_path, v_data_dirs_tmp, 0, CfgParam.excluded_dir);
    v_data_dirs = vector<string>(v_data_dirs_tmp.begin() + CfgParam.start_idx, v_data_dirs_tmp.end());

    //创建点云，初始化变量
    vector<std::shared_ptr<open3d::geometry::PointCloud>> pc_gt(v_data_dirs.size());
    double voxel_size = 2.5;
    vector<Eigen::Matrix4d> gt_poses_to_base;

    string result_dir = test_data_root_path + "/results/" + CfgParam.pairwise_reg_method + "-" + CfgParam.data_label;

    read_gt_pose(v_data_dirs, gt_poses_to_base);

#pragma omp parallel
#pragma omp for
    for (int pointID = 0; pointID < v_data_dirs.size(); ++pointID)
    {
        std::shared_ptr<open3d::geometry::PointCloud> raw_pc(new open3d::geometry::PointCloud);
        open3d::io::ReadPointCloud(v_data_dirs[pointID] + "/point_cloud.pcd", *raw_pc);
        raw_pc = raw_pc->VoxelDownSample(voxel_size);

        raw_pc->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * CfgParam.scale_normal_radius,
                                                                          CfgParam.max_n_normal_pt));
        raw_pc->OrientNormalsTowardsCameraLocation();

        pc_gt[pointID].reset(new open3d::geometry::PointCloud(*raw_pc));
        pc_gt[pointID]->Transform(gt_poses_to_base[pointID]);

    }

    std::shared_ptr<open3d::geometry::PointCloud> pc_integrate_gt(new open3d::geometry::PointCloud());
    for (int pointID = 0; pointID < pc_gt.size(); ++pointID)
    {
        *pc_integrate_gt = *pc_integrate_gt + *pc_gt[pointID];
    }

//    o3d_vis->addPointCloudShow(pc_integrate_glob, 0, true);
//    o3d_vis->addPointCloudShow(pc_integrate_pairwise, 0, true);

    std::shared_ptr<open3d::geometry::PointCloud> pc_norm_checked_gt(new open3d::geometry::PointCloud);
    norm_check(pc_integrate_gt, pc_norm_checked_gt, "");
    open3d::io::WritePointCloud(result_dir + "/models/NonNoise_gt.ply", *pc_norm_checked_gt);

    auto trimmed_mesh_opt = mesh_reconstruction(pc_norm_checked_gt);
    string area = std::to_string(trimmed_mesh_opt->GetSurfaceArea());
    open3d::io::WriteTriangleMesh(result_dir + "/models/poison_triang_gt_"+area+".ply", *trimmed_mesh_opt);
}

void
display_multiview(string test_data_root_path,  O3dVis *o3d_vis)
{
    //获取数据
    vector<string> v_data_dirs_tmp;
    vector<string> v_data_dirs;
    get_file_list(test_data_root_path, v_data_dirs_tmp, 0, CfgParam.excluded_dir);
    v_data_dirs = vector<string>(v_data_dirs_tmp.begin() + CfgParam.start_idx, v_data_dirs_tmp.end());

    //创建点云，初始化变量
    vector<std::shared_ptr<open3d::geometry::PointCloud>> pc_glob(v_data_dirs.size());
    vector<std::shared_ptr<open3d::geometry::PointCloud>> pc_pairwise(v_data_dirs.size());
    double voxel_size = 2.5;
    vector<Eigen::Matrix4d> globopt_poses_to_base;
    vector<Eigen::Matrix4d> pairwise_poses_to_base;

    string result_dir = test_data_root_path + "/results/" + CfgParam.pairwise_reg_method + "-" + CfgParam.data_label;

    //globopt_poses_to_base.push_back(Eigen::Matrix4d::Identity());
    //pairwise_poses_to_base.push_back(Eigen::Matrix4d::Identity());
    read_pose(result_dir + "/globopt_poses.txt", globopt_poses_to_base);
    read_pose(result_dir + "/pairwise_poses.txt", pairwise_poses_to_base);
//    open3d::io::WritePointCloud(result_dir + "/models/glob_opt_align.ply", *pc_all_glob_opt);
//    open3d::io::WritePointCloud(result_dir + "/models/pairwise_align.ply", *pc_all_pairwise);
//    std::shared_ptr<open3d::geometry::PointCloud> pc_norm_checked(new open3d::geometry::PointCloud);
//    norm_check(pc_all_glob_opt, pc_norm_checked, out_dir);
//    open3d::io::WritePointCloud(out_dir + "/models/final_glob_opt_align.ply", *pc_norm_checked);

    // 重建网格
//    if (CfgParam.apply_mesh_recon)
//    {
//        //保存结果数据对比
////        pc_all_glob_opt->colors_.clear();
////        pc_all_pairwise->colors_.clear();
//        auto trimmed_mesh = mesh_reconstruction(pc_norm_checked, out_dir + "/models");
//        load_textures_simdata(v_data_dirs, v_global_trans2next, Eigen::Matrix4d::Identity(), out_dir + "/models");
//    }


#pragma omp parallel
#pragma omp for
    for (int pointID = 0; pointID < v_data_dirs.size(); ++pointID)
    {
        std::shared_ptr<open3d::geometry::PointCloud> raw_pc(new open3d::geometry::PointCloud);
        open3d::io::ReadPointCloud(v_data_dirs[pointID] + "/point_cloud.pcd", *raw_pc);
        raw_pc = raw_pc->VoxelDownSample(voxel_size);

        raw_pc->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * CfgParam.scale_normal_radius,
                                                                       CfgParam.max_n_normal_pt));
        raw_pc->OrientNormalsTowardsCameraLocation();

        pc_glob[pointID].reset(new open3d::geometry::PointCloud(*raw_pc));
        pc_glob[pointID]->Transform(globopt_poses_to_base[pointID]);

        pc_pairwise[pointID].reset(new open3d::geometry::PointCloud(*raw_pc));
        pc_pairwise[pointID]->Transform(pairwise_poses_to_base[pointID]);
//        cout << "reading point cloud: " << v_data_dirs[i] + "/point_cloud.pcd" << endl;
//        pcs[i] = pre_process(v_data_dirs[i], voxel_size, globopt_poses_to_base[i]);
//        open3d::io::WritePointCloud(v_data_dirs[i] + "/object_cloud.pcd", *pcs[i]);
    }

    std::shared_ptr<open3d::geometry::PointCloud> pc_integrate_glob(new open3d::geometry::PointCloud());
    std::shared_ptr<open3d::geometry::PointCloud> pc_integrate_pairwise(new open3d::geometry::PointCloud());
    for (int pointID = 0; pointID < pc_glob.size(); ++pointID)
    {
        *pc_integrate_glob = *pc_integrate_glob + *pc_glob[pointID];
        *pc_integrate_pairwise = *pc_integrate_pairwise + *pc_pairwise[pointID];
    }

//    o3d_vis->addPointCloudShow(pc_integrate_glob, 0, true);
//    o3d_vis->addPointCloudShow(pc_integrate_pairwise, 0, true);

    std::shared_ptr<open3d::geometry::PointCloud> pc_norm_checked_pw(new open3d::geometry::PointCloud);
    norm_check(pc_integrate_pairwise, pc_norm_checked_pw, "");
    open3d::io::WritePointCloud(result_dir + "/models/NonNoise_pw.ply", *pc_norm_checked_pw);

    std::shared_ptr<open3d::geometry::PointCloud> pc_norm_checked_glob(new open3d::geometry::PointCloud);
    norm_check(pc_integrate_glob, pc_norm_checked_glob, "");
    open3d::io::WritePointCloud(result_dir + "/models/NonNoise_glob.ply", *pc_norm_checked_glob);

//
//    //保存点云
//    open3d::io::WritePointCloud(data_root + "/init_integrate.pcd", *pc_integrate);
//
//    o3d_vis->addPointCloudShow(pc_integrate, 0, true);
//
////    string results_output_path = data_root + "/../results_model/";
////    create_dir(results_output_path, false);
//    create_dir(result_dir + "/texture", false);
//    create_dir(result_dir + "/model", false);
//    create_dir(result_dir + "/debug", false);
    auto trimmed_mesh_opt = mesh_reconstruction(pc_norm_checked_glob);
    open3d::io::WriteTriangleMesh(result_dir + "/models/poison_triang_opt.ply", *trimmed_mesh_opt);

    auto trimmed_mesh_pw = mesh_reconstruction(pc_norm_checked_pw);

    open3d::io::WriteTriangleMesh(result_dir + "/models/poison_triang_pw.ply", *trimmed_mesh_pw);

}

int
main(int argc, char *argv[])
{

    string input_config_file = argv[1];
    //prepare params
    readParameters(input_config_file);
    double noise_scale = (CfgParam.noise_sigma_range[1] + CfgParam.noise_sigma_range[0]) / 2.0;
    std::ostringstream streamObj;
    streamObj << std::fixed << std::setprecision(2) << noise_scale;
    std::string str = streamObj.str();
    CfgParam.data_label = CfgParam.data_label + str;

    O3dVis *o3d_vis = new O3dVis;
    for (int method_id = 0; method_id < CfgParam.all_methods.size(); ++method_id)
    {
        CfgParam.pairwise_reg_method = CfgParam.all_methods[method_id];

        for (int data_id = 0; data_id < CfgParam.test_data_list.size(); ++data_id)
        {

            string test_data_root_path = CfgParam.data_root_path + "/" + CfgParam.test_data_list[data_id];
            string out_dir = test_data_root_path + "/results/" + CfgParam.pairwise_reg_method + "-" + CfgParam.data_label;

            cout << "=====================================" << endl;
            cout << test_data_root_path << endl;
            cout << out_dir << endl;
            display_multiview(test_data_root_path, o3d_vis);
//            display_multiview_gt(test_data_root_path, o3d_vis);
            //vector<string> v_rgb_path,
            //              vector<Eigen::Matrix4d> &T_wc,
            //              Eigen::Matrix4d base_trans,
            //              vector<Eigen::Matrix4f> &cam_poses,
            //              vector<string> &cam_imgs,
            //              string results_output_path
        }
    }
    return 0;
}
