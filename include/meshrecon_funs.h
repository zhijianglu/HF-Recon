//
// Created by zjl1 on 2023/10/26.
//

#ifndef MESHRECON_FUNS_H
#define MESHRECON_FUNS_H

#include <chrono>
#include <cstdio>
#include <memory>
#include <open3d/geometry/KDTreeSearchParam.h>
#include <string>
#include <memory.h>
#include "open3d/Open3D.h"
#include "tic_toc.h"

#include <iostream>
#include <memory>
#include <thread>
#include <random>
#include <cstdlib>
#include <chrono>

#include <Eigen/Dense>

#include "open3d/Open3D.h"
#include "open3d/pipelines/registration/GlobalOptimization.h"

#include "iostream"

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"

//添加open3d聚类头文件
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/KDTreeFlann.h>
#include <open3d/geometry/Octree.h>
#include <random>
#include "tools.h"
#include "PoseGraphOpt.h"
#include "global.h"
#include "O3dVis.h"

void norm_check(std::shared_ptr<open3d::geometry::PointCloud>& pcd_in, std::shared_ptr<open3d::geometry::PointCloud>& pcd_out, string out_dir){
    std::shared_ptr<open3d::geometry::PointCloud> pcd_tmp(new open3d::geometry::PointCloud);
    *pcd_tmp = *pcd_in;
    double radius = 15;
    int max_nn = 50;
    pcd_tmp->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(radius, max_nn));//设置半径及最大k临近点
//    pcd_tmp->OrientNormalsTowardsCameraLocation(); //

    //open3d::io::WritePointCloud(out_dir + "/models/non_normcheck_glob_opt_align.ply", *pcd_tmp);


    open3d::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*pcd_in);
    open3d::geometry::KDTreeSearchParamHybrid search_param = open3d::geometry::KDTreeSearchParamHybrid(radius/2.0, max_nn/2.0);
//    std::shared_ptr<open3d::geometry::PointCloud> source_coview(new open3d::geometry::PointCloud);
    vector<size_t> coview_idx;
    for (int i = 0; i < pcd_tmp->points_.size(); ++i)
    {
// 指定要查询的点
        Eigen::Vector3d query_point = pcd_tmp->points_[i];

// 执行最近点搜索，返回查询点最近的点的索引和距离
        std::vector<int> indices;
        std::vector<double> distances;
        kdtree.Search(query_point, search_param, indices, distances);
        //计算法向量junzhi
        Eigen::Vector3d normal_sum(0,0,0);
        for (int j = 0; j < indices.size(); ++j)
            normal_sum += pcd_in->normals_[indices[j]];
        normal_sum /= indices.size();
        if(normal_sum.dot(pcd_tmp->normals_[i])<0)
            pcd_tmp->normals_[i] *= -1;
    }
    pcd_out = pcd_tmp;
}

void
load_textures_realdata(vector<string> v_rgb_path,
                       PinholeCam eRgbCam,
                      vector<Eigen::Matrix4d> &T_wc,
                      Eigen::Matrix4d base_trans,
                      string results_output_path)
{
    create_dir(results_output_path+"/texture",false);
//    cv::FileStorage fsSettings(v_rgb_path[0] + "/cam_info.yaml", cv::FileStorage::READ);
    std::vector<int> StrCamSize;

    std::vector<int> RgbCamSize;
//    cv::Mat eRgbCamK = eRgbCam.cameraMatrix;
//    PinholeCam RGBCam =  PinholeCam(img0.cols, img0.rows, eRgbCamK.at<double>(0, 0), eRgbCamK.at<double>(1, 1), eRgbCamK.at<double>(0, 2), eRgbCamK.at<double>(1, 2), 0.0, 0.0, 0.0, 0.0);

    int n_frame = v_rgb_path.size();
    for (int frame_idx = 0; frame_idx < n_frame; ++frame_idx)
    {
//        模型通过这个可以映射到其对应的rgb图像
        cv::Mat texture_mat = imread(v_rgb_path[frame_idx]+"/rgb.png", IMREAD_UNCHANGED);
        eRgbCam.undistort(texture_mat, texture_mat);

        string name;
        stringstream ss;
        ss << setw(2) << setfill('0') << to_string(frame_idx);
        ss >> name;
        imwrite(results_output_path + "/texture/" + name + ".png", texture_mat);
        T_wc[frame_idx] = base_trans * T_wc[frame_idx];
        Eigen::Matrix4d T_ = T_wc[frame_idx].inverse();
        T_wc[frame_idx] = T_;
//        cam_poses.push_back(T_wc[frame_idx].cast<float>());
        ofstream cam_file(results_output_path + "/texture/" + name + ".cam");
        cam_file << T_wc[frame_idx](0, 3) << " " << T_wc[frame_idx](1, 3) << " " << T_wc[frame_idx](2, 3) << " ";
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                cam_file << T_wc[frame_idx](r, c) << " ";
            }
        }
        cam_file << endl;
        cam_file
            << (eRgbCam.fx + eRgbCam.fy) / (2.0 * eRgbCam.cam_width)
            << " "
            << 0 << " " << 0 << " " << 1 << " "
            << eRgbCam.cx / eRgbCam.cam_width << " "
            << eRgbCam.cy / eRgbCam.cam_height << endl;
        cam_file.close();
    }
}



void
load_textures_simdata(vector<string> v_rgb_path,
                      vector<Eigen::Matrix4d> &T_wc,
                      Eigen::Matrix4d base_trans,
                      string results_output_path)
{
    create_dir(results_output_path+"/texture",false);
    cv::FileStorage fsSettings(v_rgb_path[0] + "/cam_info.yaml", cv::FileStorage::READ);
    std::vector<int> StrCamSize;
    cv::Mat img0 = cv::imread(v_rgb_path[0] + "/img.png");

    cv::Mat RgbCamK;
    std::vector<int> RgbCamSize;
    fsSettings["rgb_camera_K"] >> RgbCamK;
    PinholeCam RGBCam =  PinholeCam(img0.cols, img0.rows, RgbCamK.at<double>(0, 0), RgbCamK.at<double>(1, 1), RgbCamK.at<double>(0, 2), RgbCamK.at<double>(1, 2), 0.0, 0.0, 0.0, 0.0);

    int n_frame = v_rgb_path.size();
    for (int frame_idx = 0; frame_idx < n_frame; ++frame_idx)
    {
//        模型通过这个可以映射到其对应的rgb图像
        cv::Mat texture_mat = imread(v_rgb_path[frame_idx]+"/img.png", IMREAD_UNCHANGED);
        //RGBCam.undistort(texture_mat, texture_mat);

        string name;
        stringstream ss;
        ss << setw(2) << setfill('0') << to_string(frame_idx);
        ss >> name;
        imwrite(results_output_path + "/texture/" + name + ".png", texture_mat);
        T_wc[frame_idx] = base_trans * T_wc[frame_idx];
        Eigen::Matrix4d T_ = T_wc[frame_idx].inverse();
        T_wc[frame_idx] = T_;
//        cam_poses.push_back(T_wc[frame_idx].cast<float>());
        ofstream cam_file(results_output_path + "/texture/" + name + ".cam");
        cam_file << T_wc[frame_idx](0, 3) << " " << T_wc[frame_idx](1, 3) << " " << T_wc[frame_idx](2, 3) << " ";
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                cam_file << T_wc[frame_idx](r, c) << " ";
            }
        }
        cam_file << endl;
        cam_file
            << (RGBCam.fx + RGBCam.fy) / (2.0 * RGBCam.cam_width)
            << " "
            << 0 << " " << 0 << " " << 1 << " "
            << RGBCam.cx / RGBCam.cam_width << " "
            << RGBCam.cy / RGBCam.cam_height << endl;
        cam_file.close();
    }
}

std::shared_ptr<open3d::geometry::TriangleMesh> mesh_reconstruction( std::shared_ptr<open3d::geometry::PointCloud> pc_all_glob_opt){
    // 泊松表面重建
    auto [trimmed_mesh, densities] = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*pc_all_glob_opt, CfgParam.poission_recon_depth);

    // 设置密度阈值
    double density_threshold = CfgParam.mesh_cut_scale; // 你可以根据需要调整这个阈值

    // 创建一个新的三角网格来存储修剪后的结果

    // 收集低于阈值密度的顶点索引
    std::vector<bool> vertices_to_remove(trimmed_mesh->vertices_.size(), false);
//    open3d::geometry::KDTreeFlann kdtree;
//    kdtree.SetGeometry(*pc_all_glob_opt);
//    open3d::geometry::KDTreeSearchParamHybrid feature_search_param = open3d::geometry::KDTreeSearchParamHybrid(CfgParam.mesh_cut_scale*CfgParam.mesh_voxel, 10);
    for (size_t i = 0; i < densities.size(); ++i) {
        if (densities[i] < density_threshold) {
            vertices_to_remove[i] = true;
        }
    }

    // 删除选中的顶点及其相关的三角面
    trimmed_mesh->RemoveVerticesByMask(vertices_to_remove);

    return trimmed_mesh;
}

#endif //MESHRECON_FUNS_H
