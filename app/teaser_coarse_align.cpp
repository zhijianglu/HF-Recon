//
// Created by zjl1 on 2023/7/22.
//

#include <chrono>
#include <memory>
#include <open3d/geometry/KDTreeSearchParam.h>
#include <open3d/io/PointCloudIO.h>
#include <string>
#include "file_tool.h"
#include <iostream>
#include "open3d/geometry/PointCloud.h"

#include <open3d/geometry/KDTreeFlann.h>
#include "PoseGraphOpt.h"
#include "O3dVis.h"
#include "global.h"

#include <teaser/ply_io.h>
#include <teaser/registration.h>

using namespace std;

std::shared_ptr<open3d::geometry::PointCloud> pre_process(string path, double voxel_size, std::shared_ptr<open3d::pipelines::registration::Feature>& pcd_f){
    std::shared_ptr<open3d::geometry::PointCloud> pcd(new open3d::geometry::PointCloud);
    cout << "reading point cloud: " << path + "/point_cloud.pcd" << endl;
    if (!open3d::io::ReadPointCloud(path + "/point_cloud.pcd", *pcd))
    {
        std::cout << "无法读取点云文件：" << path + "/point_cloud.pcd" << std::endl;
        return nullptr;
    }

    for (int i = 0; i < pcd->points_.size(); ++i)
        pcd->points_[i] /= 1000.0;

    pcd = pcd->VoxelDownSample(voxel_size);
    pcd->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2, CfgParam.max_n_normal_pt));

    //设置半径及最大k临近点
    pcd->OrientNormalsTowardsCameraLocation();
    pcd_f = open3d::pipelines::registration::ComputeFPFHFeature( *pcd, open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 4, 30));
    return pcd;
}
std::vector<std::pair<int, int>> ComputeCorrespondencesBetweenPointClouds(
    std::shared_ptr<open3d::pipelines::registration::Feature>& src_f,
    std::shared_ptr<open3d::pipelines::registration::Feature>& tgt_f
    ){
    open3d::pipelines::registration::CorrespondenceSet coresbound_ = open3d::pipelines::registration::CorrespondencesFromFeatures(*src_f, *tgt_f);
    std::vector<std::pair<int, int>> correspondences;
    for (auto& c : coresbound_)
    {
        correspondences.push_back(std::make_pair(c[0], c[1]));
        //cout<<c[0]<<" "<<c[1]<<endl;
    }
    return correspondences;
}

void o3d2teaser(const std::shared_ptr<open3d::geometry::PointCloud>& o3d_cloud, teaser::PointCloud& teaser_cloud) {
    // 清空 TEASER++ 点云，以防它之前已有数据
    teaser_cloud.clear();

    // 确保 Open3D 点云不为空
    if (o3d_cloud && !o3d_cloud->IsEmpty()) {
        // 预留足够的空间以优化性能
        teaser_cloud.reserve(o3d_cloud->points_.size());

        // 遍历 Open3D 点云中的每个点
        for (const auto& point : o3d_cloud->points_) {
            // 创建一个 TEASER++ PointXYZ 对象并添加到点云中
            teaser::PointXYZ p;
            p.x = static_cast<float>(point(0));
            p.y = static_cast<float>(point(1));
            p.z = static_cast<float>(point(2));
            teaser_cloud.push_back(p);
        }
    }
}

int apply_test(string test_data_root_path)
{
    std::cout << "apply testing data:" << test_data_root_path << endl;
    //获取数据
    vector<string> v_data_dirs_tmp;
    vector<string> v_data_dirs;
    get_file_list(test_data_root_path, v_data_dirs_tmp, 0, CfgParam.excluded_dir);
    v_data_dirs = vector<string>(v_data_dirs_tmp.begin() + CfgParam.start_idx, v_data_dirs_tmp.end());
    std::cout << "data size:" << v_data_dirs_tmp.size() << endl;

    // 创建一个可视化窗口
    O3dVis o3d_vis;

    //创建点云，初始化变量
//    vector<std::shared_ptr<open3d::geometry::PointCloud>> pcs(v_data_dirs.size());
    vector<Eigen::Matrix4d> pc_poses_to_next(v_data_dirs.size());
    vector<Eigen::Matrix4d> pc_poses_gt(v_data_dirs.size());
    vector<std::shared_ptr<open3d::pipelines::registration::Feature>> pc_feature(v_data_dirs.size());
    vector<std::shared_ptr<open3d::geometry::PointCloud>> PCDs(v_data_dirs.size());
    vector<teaser::PointCloud> teaser_PCDs(v_data_dirs.size());
//#pragma omp parallel
//#pragma omp for
    for (int i = 0; i < v_data_dirs.size(); ++i)
    {
        PCDs[i] = pre_process(v_data_dirs[i], CfgParam.teaser_align_voxel, pc_feature[i]);
        o3d2teaser(PCDs[i],teaser_PCDs[i]);
    }

    for (int i = 1; i < v_data_dirs.size(); ++i)
    {
        std::shared_ptr<open3d::geometry::PointCloud>& src_cloud_o3d = PCDs[i-1];
        std::shared_ptr<open3d::geometry::PointCloud>& tgt_cloud_o3d = PCDs[i];

        teaser::PointCloud& src_cloud = teaser_PCDs[i-1];
        teaser::PointCloud& tgt_cloud = teaser_PCDs[i];
        std::vector<std::pair<int, int>> correspondences = ComputeCorrespondencesBetweenPointClouds(pc_feature[i-1], pc_feature[i]);
        // Run TEASER++ registration
        // Prepare solver parameters
        teaser::RobustRegistrationSolver::Params params;
        params.cbar2 = 0.5;
        params.estimate_scaling = false;
        params.rotation_max_iterations = 1000;
        params.rotation_gnc_factor = 1.4;
        params.rotation_estimation_algorithm =
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
        params.rotation_cost_threshold = 0.005;

        // Solve with TEASER++
        teaser::RobustRegistrationSolver solver(params);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        solver.solve(src_cloud, tgt_cloud, correspondences);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        auto solution = solver.getSolution();

        // Compare results
        std::cout << "=====================================" << std::endl;
        std::cout << "          TEASER++ Results           " << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Estimated rotation: " << std::endl;
        std::cout << solution.rotation << std::endl;
        std::cout << std::endl;
        std::cout << "Estimated translation: " << std::endl;
        std::cout << solution.translation << std::endl;
        std::cout << std::endl;
        std::cout << "Time taken (s): "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0
                  << std::endl;
        Eigen::Matrix4f Trans;
        Trans.setIdentity();
        Trans.block(0, 0, 3, 3) = solution.rotation.cast<float>();
        Trans.block(0, 3, 3, 1) = solution.translation.cast<float>();
        std::cout<<"Trans: \n"<< Trans <<std::endl;
        std::shared_ptr<open3d::geometry::PointCloud> pc_show(new open3d::geometry::PointCloud(*src_cloud_o3d));
        pc_show->Transform(Trans.cast<double>());
        o3d_vis.addPointCloudShow(tgt_cloud_o3d, 10, true);
        pc_show->PaintUniformColor(Eigen::Vector3d(1, 0, 0.0));
        o3d_vis.addPointCloudShow(pc_show, 100, false);

        Trans.block(0,3,3,1) *= 1000.0;
        ofstream pose_file(v_data_dirs[i-1] + "/coarse_trans2next.txt");
        pose_file << Trans << endl;
        pose_file.close();
    }
    return 0;
}

int
main(int argc, char *argv[]){
    string input_config_file = argv[1];
    readParameters(input_config_file);

    for (int data_id = 0; data_id < CfgParam.test_data_list.size(); ++data_id)
    {
        string test_data_root_path = CfgParam.data_root_path + "/" + CfgParam.test_data_list[data_id];
        apply_test(test_data_root_path);
    }
    cout << "process finished!" << endl;
    return 0;
}



