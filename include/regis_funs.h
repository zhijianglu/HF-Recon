//
// Created by zjl1 on 2023/10/26.
//

#ifndef MULTIPATHREGIS_REGIS_FUNS_H
#define MULTIPATHREGIS_REGIS_FUNS_H

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

void
cut_coview_o3d(std::shared_ptr<open3d::geometry::PointCloud> source_in,
               std::shared_ptr<open3d::geometry::PointCloud> target_in,
               std::shared_ptr<open3d::geometry::PointCloud>& source_out,
               std::shared_ptr<open3d::geometry::PointCloud>& target_out,
               double coview_distance)
{
    open3d::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*target_in);
    open3d::geometry::KDTreeSearchParamHybrid feature_search_param = open3d::geometry::KDTreeSearchParamHybrid(coview_distance, 10);
//    std::shared_ptr<open3d::geometry::PointCloud> source_coview(new open3d::geometry::PointCloud);
    vector<size_t> coview_idx;
    std::set<size_t> coview_target_idx;

    for (int i = 0; i < source_in->points_.size(); ++i)
    {
// 指定要查询的点
        Eigen::Vector3d query_point = source_in->points_[i];

// 执行最近点搜索，返回查询点最近的点的索引和距离
        std::vector<int> indices;
        std::vector<double> distances;
        kdtree.Search(query_point, feature_search_param, indices, distances);
        if (indices.size() > 5)
        {
            coview_idx.push_back(i);
            coview_target_idx.insert(indices.begin(), indices.end());
        }
    }

    source_out = source_in->SelectByIndex(coview_idx);
    std::vector<size_t> coview_target_idx_vec(coview_target_idx.begin(), coview_target_idx.end());
    target_out = target_in->SelectByIndex(coview_target_idx_vec);
}

double
evaluate_coview_o3d(open3d::geometry::PointCloud &source,
                    open3d::geometry::PointCloud &target,
                    Eigen::Matrix4d T_st,
                    double coview_distance)
{
    open3d::geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    open3d::geometry::KDTreeSearchParamHybrid feature_search_param = open3d::geometry::KDTreeSearchParamHybrid(coview_distance, 50);
//    std::shared_ptr<open3d::geometry::PointCloud> source_coview(new open3d::geometry::PointCloud);
    int cnt_src = 0;
    for (int i = 0; i < source.points_.size(); ++i)
    {
// 指定要查询的点
        Eigen::Vector3d query_point = T_st.block(0, 0, 3, 3) * source.points_[i] + T_st.block(0, 3, 3, 1);

// 执行最近点搜索，返回查询点最近的点的索引和距离
        std::vector<int> indices;
        std::vector<double> distances;
        kdtree.Search(query_point, feature_search_param, indices, distances);
        if (indices.size() > 5)
        {
            if ((source.colors_[i] - target.colors_[indices[0]]).norm() < CfgParam.global_coview_color_diff)
                cnt_src++;
        }
    }

    double coview_rate_src = double(cnt_src) / double(source.points_.size());
    double coview_rate_tag = double(cnt_src) / double(target.points_.size());
//    source = source->SelectByIndex(coview_idx);
    return min(coview_rate_src, coview_rate_tag);
}

void
reg_icp_point2point(std::shared_ptr<open3d::geometry::PointCloud> src_PC_raw,
                    std::shared_ptr<open3d::geometry::PointCloud> tag_PC_raw,
                    Eigen::Matrix4d &T_st,
                    double coview = 0,
                    bool use_init = false)
{
    std::shared_ptr<open3d::geometry::PointCloud> src_PC;
    std::shared_ptr<open3d::geometry::PointCloud> tag_PC;
    if (coview != 0)
    {
        cut_coview_o3d(src_PC_raw, tag_PC_raw, src_PC, tag_PC, coview);
    }
    else
    {
        src_PC = src_PC_raw;
        tag_PC = tag_PC_raw;
    }

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;

    Vertices vertices_target;
    vertices_target.resize(3, tag_PC->points_.size());
    for (int pt_id = 0; pt_id < tag_PC->points_.size(); ++pt_id)
    {
        vertices_target(0, pt_id) = tag_PC->points_[pt_id].x();
        vertices_target(1, pt_id) = tag_PC->points_[pt_id].y();
        vertices_target(2, pt_id) = tag_PC->points_[pt_id].z();
    }

    //--- Model that will be rigidly transformed
    Vertices vertices_source;
    vertices_source.resize(3, src_PC->points_.size());
    for (int pt_id = 0; pt_id < src_PC->points_.size(); ++pt_id)
    {
        vertices_source(0, pt_id) = src_PC->points_[pt_id].x();
        vertices_source(1, pt_id) = src_PC->points_[pt_id].y();
        vertices_source(2, pt_id) = src_PC->points_[pt_id].z();
    }

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
//    std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;


    ///--- Execute registration
    ICP::Parameters pars;
    pars.nu_end_k = 0.1;
    pars.f = ICP::WELSCH;
    pars.use_AA = true;
    pars.use_init = use_init;
    pars.init_trans = T_st;
    PairWiseAlign::Align::MYICP fricp;
    fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
    cout << "point_to_point finished!" << endl;

    MatrixXX res_trans;
    res_trans = pars.res_trans;
    vertices_source = scale * vertices_source;
    res_trans.block(0, 3, 3, 1) *= scale;

    T_st.setIdentity();
    T_st.block(0, 0, 3, 3) = res_trans.block(0, 0, 3, 3);
    T_st.block(0, 3, 3, 1) = res_trans.block(0, 3, 3, 1);
}

void
reg_aaicp_point2point(std::shared_ptr<open3d::geometry::PointCloud> src_PC_raw,
                    std::shared_ptr<open3d::geometry::PointCloud> tag_PC_raw,
                    Eigen::Matrix4d &T_st,
                    double coview = 0,
                    bool use_init = false)
{
    std::shared_ptr<open3d::geometry::PointCloud> src_PC;
    std::shared_ptr<open3d::geometry::PointCloud> tag_PC;
    if (coview != 0)
    {
        cut_coview_o3d(src_PC_raw, tag_PC_raw, src_PC, tag_PC, coview);
    }
    else
    {
        src_PC = src_PC_raw;
        tag_PC = tag_PC_raw;
    }

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;

    Vertices vertices_target;
    vertices_target.resize(3, tag_PC->points_.size());
    for (int pt_id = 0; pt_id < tag_PC->points_.size(); ++pt_id)
    {
        vertices_target(0, pt_id) = tag_PC->points_[pt_id].x();
        vertices_target(1, pt_id) = tag_PC->points_[pt_id].y();
        vertices_target(2, pt_id) = tag_PC->points_[pt_id].z();
    }

    //--- Model that will be rigidly transformed
    Vertices vertices_source;
    vertices_source.resize(3, src_PC->points_.size());
    for (int pt_id = 0; pt_id < src_PC->points_.size(); ++pt_id)
    {
        vertices_source(0, pt_id) = src_PC->points_[pt_id].x();
        vertices_source(1, pt_id) = src_PC->points_[pt_id].y();
        vertices_source(2, pt_id) = src_PC->points_[pt_id].z();
    }

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
//    std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;


    ///--- Execute registration
    ICP::Parameters pars;
//    pars.nu_end_k = 0.1;
//    pars.f = ICP::WELSCH;
//    pars.use_AA = true;
//    pars.use_init = use_init;
//    pars.init_trans = T_st;
//    PairWiseAlign::Align::MYICP fricp;
//    fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
    PairWiseAlign::AAICP::point_to_point_aaicp(vertices_source, vertices_target, source_mean, target_mean, pars);
    cout << "point_to_point finished!" << endl;

    MatrixXX res_trans;
    res_trans = pars.res_trans;
    vertices_source = scale * vertices_source;
    res_trans.block(0, 3, 3, 1) *= scale;

    T_st.setIdentity();
    T_st.block(0, 0, 3, 3) = res_trans.block(0, 0, 3, 3);
    T_st.block(0, 3, 3, 1) = res_trans.block(0, 3, 3, 1);
}

void
reg_sicp_point2point(std::shared_ptr<open3d::geometry::PointCloud> src_PC_raw,
                      std::shared_ptr<open3d::geometry::PointCloud> tag_PC_raw,
                      Eigen::Matrix4d &T_st,
                      double coview = 0,
                      bool use_init = false)
{
    std::shared_ptr<open3d::geometry::PointCloud> src_PC;
    std::shared_ptr<open3d::geometry::PointCloud> tag_PC;
    if (coview != 0)
    {
        cut_coview_o3d(src_PC_raw, tag_PC_raw, src_PC, tag_PC, coview);
    }
    else
    {
        src_PC = src_PC_raw;
        tag_PC = tag_PC_raw;
    }

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;

    Vertices vertices_target;
    vertices_target.resize(3, tag_PC->points_.size());
    for (int pt_id = 0; pt_id < tag_PC->points_.size(); ++pt_id)
    {
        vertices_target(0, pt_id) = tag_PC->points_[pt_id].x();
        vertices_target(1, pt_id) = tag_PC->points_[pt_id].y();
        vertices_target(2, pt_id) = tag_PC->points_[pt_id].z();
    }

    //--- Model that will be rigidly transformed
    Vertices vertices_source;
    vertices_source.resize(3, src_PC->points_.size());
    for (int pt_id = 0; pt_id < src_PC->points_.size(); ++pt_id)
    {
        vertices_source(0, pt_id) = src_PC->points_[pt_id].x();
        vertices_source(1, pt_id) = src_PC->points_[pt_id].y();
        vertices_source(2, pt_id) = src_PC->points_[pt_id].z();
    }

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
//    std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;

    // set Sparse-ICP detectorParameters
    PairWiseAlign::SICP::Parameters spars;
    spars.p = 0.4;
    spars.print_icpn  = false;
    PairWiseAlign::SICP::point_to_point(vertices_source, vertices_target, source_mean, target_mean, spars);
    cout << "point_to_point finished!" << endl;

    MatrixXX res_trans;
    res_trans = spars.res_trans;
    vertices_source = scale * vertices_source;
    res_trans.block(0, 3, 3, 1) *= scale;

    T_st.setIdentity();
    T_st.block(0, 0, 3, 3) = res_trans.block(0, 0, 3, 3);
    T_st.block(0, 3, 3, 1) = res_trans.block(0, 3, 3, 1);
}

void
reg_sicp_point2plane(std::shared_ptr<open3d::geometry::PointCloud> src_PC_raw,
                     std::shared_ptr<open3d::geometry::PointCloud> tag_PC_raw,
                     Eigen::Matrix4d &T_st,
                     double coview = 0,
                     bool use_init = false)
{
    std::shared_ptr<open3d::geometry::PointCloud> src_PC;
    std::shared_ptr<open3d::geometry::PointCloud> tag_PC;
    if (coview != 0)
    {
        cut_coview_o3d(src_PC_raw, tag_PC_raw, src_PC, tag_PC, coview);
    }
    else
    {
        src_PC = src_PC_raw;
        tag_PC = tag_PC_raw;
    }

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;

    Vertices vertices_target, normal_target;
    vertices_target.resize(3, tag_PC->points_.size());
    normal_target.resize(3, tag_PC->points_.size());
    cout<<"tag_PC->normals_.size() = "<<tag_PC->normals_.size()<<endl;
    for (int pt_id = 0; pt_id < tag_PC->points_.size(); ++pt_id)
    {
        vertices_target(0, pt_id) = tag_PC->points_[pt_id].x();
        vertices_target(1, pt_id) = tag_PC->points_[pt_id].y();
        vertices_target(2, pt_id) = tag_PC->points_[pt_id].z();
        normal_target(0, pt_id) = tag_PC->normals_[pt_id].x();
        normal_target(1, pt_id) = tag_PC->normals_[pt_id].y();
        normal_target(2, pt_id) = tag_PC->normals_[pt_id].z();
    }

    //--- Model that will be rigidly transformed
    Vertices vertices_source;
    vertices_source.resize(3, src_PC->points_.size());
    for (int pt_id = 0; pt_id < src_PC->points_.size(); ++pt_id)
    {
        vertices_source(0, pt_id) = src_PC->points_[pt_id].x();
        vertices_source(1, pt_id) = src_PC->points_[pt_id].y();
        vertices_source(2, pt_id) = src_PC->points_[pt_id].z();
    }

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
//    std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;

    // set Sparse-ICP detectorParameters
    PairWiseAlign::SICP::Parameters spars;
    spars.p = 0.4;
    spars.print_icpn = false;
    PairWiseAlign::SICP::point_to_plane(vertices_source, vertices_target, normal_target, source_mean, target_mean, spars);
    cout << "point_to_point finished!" << endl;

    MatrixXX res_trans;
    res_trans = spars.res_trans;
    vertices_source = scale * vertices_source;
    res_trans.block(0, 3, 3, 1) *= scale;

    T_st.setIdentity();
    T_st.block(0, 0, 3, 3) = res_trans.block(0, 0, 3, 3);
    T_st.block(0, 3, 3, 1) = res_trans.block(0, 3, 3, 1);
}


void
reg_icp_point2plane(std::shared_ptr<open3d::geometry::PointCloud> src_PC_raw,
                    std::shared_ptr<open3d::geometry::PointCloud> tag_PC_raw,
                    Eigen::Matrix4d &T_st,
                    double coview = 0,
                    bool use_init = false)
{
    std::shared_ptr<open3d::geometry::PointCloud> src_PC;
    std::shared_ptr<open3d::geometry::PointCloud> tag_PC;
    if (coview != 0)
    {
        cut_coview_o3d(src_PC_raw, tag_PC_raw, src_PC, tag_PC, coview);
    }
    else
    {
        src_PC = src_PC_raw;
        tag_PC = tag_PC_raw;
    }

    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;

    Vertices vertices_target, normal_target;
    vertices_target.resize(3, tag_PC->points_.size());
    normal_target.resize(3, tag_PC->points_.size());
    cout<<"tag_PC->normals_.size() = "<<tag_PC->normals_.size()<<endl;
    for (int pt_id = 0; pt_id < tag_PC->points_.size(); ++pt_id)
    {
        vertices_target(0, pt_id) = tag_PC->points_[pt_id].x();
        vertices_target(1, pt_id) = tag_PC->points_[pt_id].y();
        vertices_target(2, pt_id) = tag_PC->points_[pt_id].z();
        normal_target(0, pt_id) = tag_PC->normals_[pt_id].x();
        normal_target(1, pt_id) = tag_PC->normals_[pt_id].y();
        normal_target(2, pt_id) = tag_PC->normals_[pt_id].z();
    }

    //--- Model that will be rigidly transformed
    Vertices vertices_source, normal_source;
    vertices_source.resize(3, src_PC->points_.size());
    normal_source.resize(3, src_PC->points_.size());
    for (int pt_id = 0; pt_id < src_PC->points_.size(); ++pt_id)
    {
        vertices_source(0, pt_id) = src_PC->points_[pt_id].x();
        vertices_source(1, pt_id) = src_PC->points_[pt_id].y();
        vertices_source(2, pt_id) = src_PC->points_[pt_id].z();
        normal_source(0, pt_id) = src_PC->normals_[pt_id].x();
        normal_source(1, pt_id) = src_PC->normals_[pt_id].y();
        normal_source(2, pt_id) = src_PC->normals_[pt_id].z();
    }

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
//    std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;


    ///--- Execute registration
    ICP::Parameters pars;
    pars.nu_end_k = 0.1;
    pars.f = ICP::WELSCH;
    pars.use_AA = true;
    pars.use_init = use_init;
    pars.init_trans = T_st;
    PairWiseAlign::Align::MYICP fricp;
    fricp.point_to_plane_GN(vertices_source, vertices_target, normal_source, normal_target, source_mean, target_mean, pars);
    cout << "point_to_plane_GN finished!" << endl;

    MatrixXX res_trans;
    res_trans = pars.res_trans;
    vertices_source = scale * vertices_source;
    res_trans.block(0, 3, 3, 1) *= scale;

    T_st.setIdentity();
    T_st.block(0, 0, 3, 3) = res_trans.block(0, 0, 3, 3);
    T_st.block(0, 3, 3, 1) = res_trans.block(0, 3, 3, 1);
}

void
pairwise_registration_global(open3d::geometry::PointCloud &source,
                             open3d::geometry::PointCloud &target,
                             Eigen::Matrix4d init_transformation,
                             const double max_correspondence_distance_fine,
                             Eigen::Matrix4d &transformationICP,
                                   O3dVis *o3d_vis=nullptr)
{
    //----- Coarse registration -----//
    std::shared_ptr<open3d::pipelines::registration::RobustKernel> kernel_coarse(new open3d::pipelines::registration::L2Loss());


    std::shared_ptr<open3d::geometry::PointCloud> transed_src_coview(new open3d::geometry::PointCloud());
    std::shared_ptr<open3d::geometry::PointCloud> tag_coview(new open3d::geometry::PointCloud());

    open3d::geometry::PointCloud source_tmp = source;
    source_tmp.Transform(init_transformation);
    cut_coview_o3d(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
                   std::make_shared<open3d::geometry::PointCloud>(target),
                   transed_src_coview,
                   tag_coview,
                   max_correspondence_distance_fine);

    if (o3d_vis != nullptr)
    {
        std::shared_ptr<open3d::geometry::PointCloud> transed_src(new open3d::geometry::PointCloud());
        std::shared_ptr<open3d::geometry::PointCloud> transed_tag(new open3d::geometry::PointCloud());
        *transed_src = *transed_src_coview;
        *transed_tag = *tag_coview;

        transed_src->PaintUniformColor(Eigen::Vector3d(1, 0.5, 1));
        transed_tag->PaintUniformColor(Eigen::Vector3d(1, 1, 0.5));

        o3d_vis->addPointCloudShow(transed_src, 1, true);
        o3d_vis->addPointCloudShow(transed_tag, 0, false);
    }

    bool use_fricp = true;
    Eigen::Matrix4d fine_transformation = Eigen::Matrix4d::Identity();

    if (use_fricp)
    {
        cout << "Apply Point-to-plane ICP" << endl;
        reg_icp_point2plane(transed_src_coview, tag_coview, fine_transformation, 0);
    }
    else
    {
        //method 1
        auto icp_fine = open3d::pipelines::registration::RegistrationICP(  *transed_src_coview,
                                                                           *tag_coview,
                                                                           max_correspondence_distance_fine,
                                                                           Eigen::Matrix4d::Identity(),
                                                                           open3d::pipelines::registration::TransformationEstimationPointToPlane(kernel_coarse));
        fine_transformation = icp_fine.transformation_;

        //method 2
//    auto icp_coarse = open3d::pipelines::registration::RegistrationColoredICP(source,
//                                                                              target,
//                                                                              max_correspondence_distance_fine,
//                                                                              init_transformation,
//                                                                              open3d::pipelines::registration::TransformationEstimationForColoredICP());
//

        //method 3
//        auto icp_coarse = open3d::pipelines::registration::RegistrationGeneralizedICP(source,
//                                                                                      target,
//                                                                                      max_correspondence_distance_fine,
//                                                                                      Eigen::MatrixXd::Identity(4, 4),
//                                                                                      open3d::pipelines::registration::TransformationEstimationForGeneralizedICP());

//        coarse_transformation = icp_coarse.transformation_;
    }

//    double coview_distance = max_correspondence_distance_fine;
//    double align_score = evaluate_coview_o3d(source, target, coarse_transformation, coview_distance);
//    if(align_score < 0.5)
//    {
//        return align_score;
//    }


//    auto icp_fine =open3d::pipelines::registration::RegistrationColoredICP(source,
//                                                                                target,
//                                                                               max_correspondence_distance_fine,
//                                                                               icp_coarse.transformation_,
//                                                                                open3d::pipelines::registration::TransformationEstimationForColoredICP());

    transformationICP = fine_transformation * init_transformation;

    //信息矩阵是source转换到target下，能够匹配到的target中的点构成集合，将这些点协方差矩阵相加得到的,见论文公式(7);注意，这里是target中的点的协方差矩阵
//    transed_src_coview->Transform(init_transformation.inverse());
}


auto
pairwise_registration(open3d::geometry::PointCloud &source,
                      open3d::geometry::PointCloud &target,
                      Eigen::Matrix4d init_transformation,
                      const double max_correspondence_distance_coarse,
                      const double max_correspondence_distance_fine)
{
    //----- Coarse registration -----//
    std::shared_ptr<open3d::pipelines::registration::RobustKernel> kernel_coarse(new open3d::pipelines::registration::L2Loss());

    Eigen::Matrix4d pairwise_transformation =  Eigen::Matrix4d::Identity();
    switch (CfgParam.pairRegMethod) {
        case PairRegMethod::FRICP:
            std::cout << "Executing Fast Robust ICP..." << std::endl;
            {
                //        auto skew = get_skew();
//        init_transformation = skew * init_transformation;
//        pairwise_transformation = init_transformation;
                cout << "Apply Point-to-plane ICP" << endl;
                //将source变成shared
                open3d::geometry::PointCloud source_tmp = source;
                source_tmp.Transform(init_transformation);
                Eigen::Matrix4d fine_transformation = Eigen::Matrix4d::Identity();
                reg_icp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
                                    std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
//        reg_icp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
//                            std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
                pairwise_transformation = fine_transformation * init_transformation;
            }
            break;
        case PairRegMethod::AAICP:
            std::cout << "Executing Advanced Automated ICP..." << std::endl;
            {
                //将source变成shared
                open3d::geometry::PointCloud source_tmp = source;
                source_tmp.Transform(init_transformation);
                Eigen::Matrix4d fine_transformation = Eigen::Matrix4d::Identity();
                reg_aaicp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
                                      std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
//        reg_icp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
//                            std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
                pairwise_transformation = fine_transformation * init_transformation;
            }
            break;
        case PairRegMethod::SPARSEICP_PT:
            std::cout << "Executing SPARSEICP_PT..." << std::endl;
            {
                //将source变成shared
                open3d::geometry::PointCloud source_tmp = source;
                source_tmp.Transform(init_transformation);
                Eigen::Matrix4d fine_transformation = Eigen::Matrix4d::Identity();
                reg_sicp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
                                     std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
//        reg_icp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
//                            std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
                pairwise_transformation = fine_transformation * init_transformation;
            }
            break;
        case PairRegMethod::SPARSEICP_PL:
            std::cout << "Executing SPARSEICP_PL..." << std::endl;
            {
                //将source变成shared
                open3d::geometry::PointCloud source_tmp = source;
                source_tmp.Transform(init_transformation);
                Eigen::Matrix4d fine_transformation = Eigen::Matrix4d::Identity();
                reg_sicp_point2plane(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
                                     std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
//        reg_icp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
//                            std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
                pairwise_transformation = fine_transformation * init_transformation;
            }
            break;
        case PairRegMethod::COLORICP:
            std::cout << "Executing Color ICP..." << std::endl;
            {
//                auto icp_coarse = open3d::pipelines::registration::RegistrationICP(source,
//                                                                                   target,
//                                                                                   max_correspondence_distance_fine,
//                                                                                   init_transformation,
//                                                                                   open3d::pipelines::registration::TransformationEstimationPointToPlane(
//                                                                                       kernel_coarse));
                auto icp_coarse = open3d::pipelines::registration::RegistrationColoredICP(source,
                                                                                          target,
                                                                                          max_correspondence_distance_coarse,
                                                                                          init_transformation,
                                                                                          open3d::pipelines::registration::TransformationEstimationForColoredICP());
                pairwise_transformation = icp_coarse.transformation_;
            }
            break;
        case PairRegMethod::OPEN3DICP:
            std::cout << "Executing Open3D ICP..." << std::endl;
            {
                auto icp_coarse = open3d::pipelines::registration::RegistrationICP(source,
                                                                                   target,
                                                                                   max_correspondence_distance_coarse,
                                                                                   init_transformation,
                                                                                   open3d::pipelines::registration::TransformationEstimationPointToPlane(
                                                                                       kernel_coarse));
                pairwise_transformation = icp_coarse.transformation_;
            }
            break;
        case PairRegMethod::NONE:
        default:
            std::cout << "NON REGIS" << std::endl;
            pairwise_transformation = Eigen::Matrix4d::Identity();
            break;
    }

//    if (use_fricp)
//    {
//        auto skew = get_skew();
//        init_transformation = skew * init_transformation;
//        pairwise_transformation = init_transformation;
//        cout << "Apply Point-to-plane ICP" << endl;
//        //将source变成shared
//        open3d::geometry::PointCloud source_tmp = source;
//        source_tmp.Transform(init_transformation);
//        Eigen::Matrix4d fine_transformation = Eigen::Matrix4d::Identity();
//        reg_icp_point2plane(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
//                            std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
//        reg_icp_point2point(std::make_shared<open3d::geometry::PointCloud>(source_tmp),
//                            std::make_shared<open3d::geometry::PointCloud>(target), fine_transformation, max_correspondence_distance_coarse, 0);
//        pairwise_transformation = fine_transformation * init_transformation;
//    }
//    else
//    {
//        //method 1
//        auto icp_coarse = open3d::pipelines::registration::RegistrationICP(source,
//                                                                           target,
//                                                                           max_correspondence_distance_coarse,
//                                                                           init_transformation,
//                                                                           open3d::pipelines::registration::TransformationEstimationPointToPlane(
//                                                                               kernel_coarse));
//
        //method 2
//    auto icp_coarse = open3d::pipelines::registration::RegistrationColoredICP(source,
//                                                                              target,
//                                                                              max_correspondence_distance_coarse,
//                                                                              init_transformation,
//                                                                              open3d::pipelines::registration::TransformationEstimationForColoredICP());
//

        //method 3
//        auto icp_coarse = open3d::pipelines::registration::RegistrationGeneralizedICP(source,
//                                                                                      target,
//                                                                                      max_correspondence_distance_coarse,
//                                                                                      Eigen::MatrixXd::Identity(4, 4),
//                                                                                      open3d::pipelines::registration::TransformationEstimationForGeneralizedICP());
//        pairwise_transformation = icp_coarse.transformation_;
//    }

    double coview_distance = max_correspondence_distance_fine;

    double coview_rate = evaluate_coview_o3d(source, target, pairwise_transformation, coview_distance);
    if (coview_rate < 0.2)
    {
        cout << RED << "coview score is too low: " << coview_rate << " , considering use other method to registration" << BOLDWHITE << endl;
        pairwise_transformation = Eigen::Matrix4d::Identity();

        for (int i = 0; i < 100; ++i)
        {
            open3d::pipelines::registration::ICPConvergenceCriteria criteria(1, 1e-6, 1);
            auto icp_coarse = open3d::pipelines::registration::RegistrationICP(source,
                                                                               target,
                                                                               50,
                                                                               pairwise_transformation,
                                                                               open3d::pipelines::registration::TransformationEstimationPointToPoint(),
                                                                               criteria);
            pairwise_transformation = icp_coarse.transformation_;
        }
        coview_rate = evaluate_coview_o3d(source, target, pairwise_transformation, coview_distance);
        cout << RED << "re-estimated coview score: " << coview_rate << " , considering use other method to registration" << BOLDWHITE << endl;
    }
    else
    {
        cout << GREEN << "coview score is ok: " << coview_rate << BOLDWHITE << endl;
    }
    // CHECK if the transformation is valid

//    std::shared_ptr<open3d::pipelines::registration::RobustKernel> kernel_fine(new open3d::pipelines::registration::TukeyLoss(0.1));
//    auto icp_fine = open3d::pipelines::registration::RegistrationICP(source,
//                                                                     target,
//                                                                     max_correspondence_distance_fine,
//                                                                     pairwise_transformation,
//                                                                     open3d::pipelines::registration::TransformationEstimationPointToPlane(kernel_fine));

//    auto icp_fine =open3d::pipelines::registration::RegistrationColoredICP(source,
//                                                                                target,
//                                                                               max_correspondence_distance_fine,
//                                                                               icp_coarse.transformation_,
//                                                                                open3d::pipelines::registration::TransformationEstimationForColoredICP());
//    return icp_fine.transformation_;
    return pairwise_transformation;
}

void
display_align_results(O3dVis *o3d_vis,
                      std::vector<std::shared_ptr<open3d::geometry::PointCloud>> &PCDs,
                      int sourceID,
                      int targetID,
                      Eigen::Matrix4d transformationICP,
                      Eigen::Vector3d color_src,
                      Eigen::Vector3d color_tag
)
{
    if (o3d_vis != nullptr)
    {
        std::shared_ptr<open3d::geometry::PointCloud> transed_src(new open3d::geometry::PointCloud());
        std::shared_ptr<open3d::geometry::PointCloud> transed_tag(new open3d::geometry::PointCloud());
        *transed_src = *PCDs[sourceID];
        *transed_tag = *PCDs[targetID];

        transed_src->PaintUniformColor(color_src);
        transed_tag->PaintUniformColor(color_tag);

        transed_src->Transform(transformationICP);

        o3d_vis->addPointCloudShow(transed_src, 1, true);
        o3d_vis->addPointCloudShow(transed_tag, 0, false);
    }
}

Eigen::Matrix4d
get_abs_pose(vector<Eigen::Matrix4d> &prior_inter_poses, int src_id, int tag_id)
{
    Eigen::Matrix4d abs_pose = Eigen::Matrix4d::Identity();
    if (src_id == tag_id)
        return abs_pose;
    if (src_id < tag_id)
    {
        for (int i = src_id; i < tag_id; i++)
        {
            abs_pose = prior_inter_poses[i] * abs_pose;
        }
    }
    else
    {
        for (int i = tag_id; i < src_id; i++)
        {
            abs_pose = prior_inter_poses[i] * abs_pose;
        }
        abs_pose = abs_pose.inverse();
    }
    return abs_pose;
}

struct PairRegisInfo
{
    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> *PCDs;
    Eigen::Matrix6d_u informationICP;
    Eigen::Matrix4d transformationICP;
    int source_id;
    int target_id;
    bool edge_valid = true;
    bool odometry_node = true;
    PairRegisInfo(std::vector<std::shared_ptr<open3d::geometry::PointCloud>> *PCDs_,
                  Eigen::Matrix4d transformationICP,
                  int source_id_,
                  int target_id_,
                  bool odometry_node_) :
        PCDs(PCDs_),
        transformationICP(transformationICP),
        source_id(source_id_),
        target_id(target_id_),
        odometry_node(odometry_node_){}
        void update_informationmatrix(double max_correspondence_distance_fine){
            //信息矩阵是source转换到target下，能够匹配到的target中的点构成集合，将这些点协方差矩阵相加得到的,见论文公式(7);注意，这里是target中的点的协方差矩阵
            std::vector<std::shared_ptr<open3d::geometry::PointCloud>> &ref_PCDs = *PCDs;
            informationICP = open3d::pipelines::registration::GetInformationMatrixFromPointClouds(*ref_PCDs[source_id],
                                                                                                  *ref_PCDs[target_id],
                                                                                                  max_correspondence_distance_fine,
                                                                                                  transformationICP);
    }
    PairRegisInfo(){}
};

void apply_global_optimization(PoseGraphOpt::PoseGraph& pose_graph){
    auto option = PoseGraphOpt::OptOption(
        CfgParam.global_max_correspondence_distance,  //double max_correspondence_distance =
        CfgParam.edge_prune_threshold, //double edge_prune_threshold
        CfgParam.preference_loop_closure,  //double preference_loop_closure
        0 //int reference_node
    );
    PoseGraphOpt::GlobalLMOptimization opt;
    PoseGraphOpt::OptConvergenceCriteria Criteria(
        CfgParam.global_max_iteration,     //int max_iteration
        1e-10,    //double min_relative_increment =
        1e-10,    //double min_relative_residual_increment =
        1e-10,    //double min_right_term =
        1e-10,    //double min_residual =
        40,      //int max_iteration_lm =
        2. / 3., //double upper_scale_factor =
        1. / 3.  // double lower_scale_factor =
    );
    PoseGraphOpt::GlobalOptimizationIRLS(pose_graph, opt,
                                         Criteria, option);
//    open3d::pipelines::registration::GlobalOptimization(pose_graph, open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
//                                                        Criteria, option);
}

bool compareEdge(const PoseGraphOpt::PoseGraphEdge& a, const PoseGraphOpt::PoseGraphEdge& b) {
    return a.information_(5, 5) > b.information_(5, 5); // 按 value decrease 序排序
}

void
loop_registration_global_parallel(std::vector<std::shared_ptr<open3d::geometry::PointCloud>> &PCDs,
                         O3dVis *o3d_vis,
                         vector<Eigen::Matrix4d> prior_inter_poses,
                         vector<Eigen::Matrix4d>& v_pairwise_trans2base,
                         vector<Eigen::Matrix4d>& v_globalopt_trans2base,
                         vector<vector<Eigen::Matrix4d>> &v_iter_poses)
{
    cout << "start local to global registration "<< endl;
    double max_correspondence_distance_coarse = CfgParam.scale_max_correspondence_distance_coarse *CfgParam.voxel_size;
    double max_correspondence_distance_fine = CfgParam.scale_max_correspondence_distance_fine * CfgParam.voxel_size;

    int n_frame = PCDs.size();
    vector<PairRegisInfo> localPairRegisInfo(n_frame-1);
//并行
#pragma omp parallel for if(o3d_vis == nullptr)
    for (int sourceID = 0; sourceID < n_frame - 1; sourceID++)
    {
        cout << YELLOW << "evaluating local edge: [" << sourceID << "<==>" << sourceID + 1 << "]" << BOLDWHITE << endl;
        Eigen::Matrix4d transformationICP;
        transformationICP = Eigen::MatrixXd::Identity(4, 4);
        transformationICP = get_abs_pose(prior_inter_poses, sourceID, sourceID + 1);
        if (o3d_vis != nullptr)
            display_align_results(o3d_vis,
                                  PCDs,
                                  sourceID,
                                  sourceID + 1,
                                  transformationICP,
                                  Eigen::Vector3d(0, 0, 1) * 0.8,
                                  Eigen::Vector3d(0, 1, 0) * 0.8);

        transformationICP = pairwise_registration(*PCDs[sourceID],*PCDs[sourceID + 1],
                                                                            transformationICP,
                                                                            max_correspondence_distance_coarse,
                                                                            max_correspondence_distance_fine);
        if (o3d_vis != nullptr)
            display_align_results(o3d_vis, PCDs, sourceID, sourceID + 1, transformationICP, Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 1, 0));
        localPairRegisInfo[sourceID] = PairRegisInfo(&PCDs, transformationICP, sourceID, sourceID + 1, true);
    }

    //记录下initial位姿，用于后面优化后的位姿的对比
    vector<Eigen::Matrix4d> v_initial_trans2base(n_frame);
    v_initial_trans2base[0] = Eigen::MatrixXd::Identity(4, 4);
    for (int pointID = 1; pointID < n_frame; ++pointID)
    {
        v_initial_trans2base[pointID] = v_initial_trans2base[pointID - 1] * prior_inter_poses[pointID - 1].inverse();
    }
    v_iter_poses.push_back(v_initial_trans2base);


    //记录下优化之前的pairwise registration的位姿，用于后面优化后的位姿的对比
    Eigen::Matrix4d odom_base2ithframe = Eigen::MatrixXd::Identity(4, 4);
    v_pairwise_trans2base[0] = Eigen::MatrixXd::Identity(4, 4);
    for (int pointID = 0; pointID < n_frame-1; ++pointID)
    {
        odom_base2ithframe = localPairRegisInfo[pointID].transformationICP * odom_base2ithframe;
        Eigen::Matrix4d trans_ithframe2base = odom_base2ithframe.inverse();
        v_pairwise_trans2base[pointID+1] = trans_ithframe2base;
    }
    v_iter_poses.push_back(v_pairwise_trans2base);
    //并行
    std::vector<PoseGraphOpt::PoseGraphEdge> global_edges_;
#pragma omp parallel for if(o3d_vis == nullptr)
    for (int sourceID = 2; sourceID < n_frame - 1; sourceID++)
    {
        int sub_targetID = sourceID;
        for (int sub_sourceID = 0; sub_sourceID < sub_targetID - 1; sub_sourceID++)
        {
            cout << BLUE << "evaluating global edge: [" << sub_sourceID << "<==>" << sub_targetID << "]" << BOLDWHITE << endl;
            Eigen::Matrix4d init_transICP_s2t;
            init_transICP_s2t = Eigen::MatrixXd::Identity(4, 4);
            init_transICP_s2t = v_pairwise_trans2base[sub_targetID].inverse() * v_pairwise_trans2base[sub_sourceID];
//            init_transICP_s2t = get_abs_pose(prior_inter_poses, sub_sourceID, sub_targetID);
            double coview_rate = evaluate_coview_o3d(*PCDs[sub_sourceID], *PCDs[sub_targetID], init_transICP_s2t, CfgParam.global_coview_distance);
            if (coview_rate < CfgParam.global_edge_coview_rate)
            {
                cout << YELLOW << "coview score is too low: " << coview_rate << " , skip this edge" << BOLDWHITE << endl;
                continue;
            }
            if (o3d_vis != nullptr)
                display_align_results(o3d_vis,
                                      PCDs,
                                      sub_sourceID,
                                      sub_targetID,
                                      init_transICP_s2t,
                                      Eigen::Vector3d(1, 0, 1),
                                      Eigen::Vector3d(1, 1, 0));

            Eigen::Matrix4d transformationICP;
            pairwise_registration_global(*PCDs[sub_sourceID], *PCDs[sub_targetID], init_transICP_s2t, max_correspondence_distance_fine, transformationICP, o3d_vis);
            Eigen::Matrix6d informationICP = PoseGraphOpt::GetInformationMatrixFromPointCloudsMy(*PCDs[sub_sourceID],
                                                                                                 *PCDs[sub_targetID],
                                                                                                 CfgParam.global_max_correspondence_distance,
                                                                                                 transformationICP,
                                                                                                 CfgParam.color_weight_mu);
            if (o3d_vis != nullptr)
                display_align_results(o3d_vis,
                                      PCDs,
                                      sub_sourceID,
                                      sub_targetID,
                                      transformationICP,
                                      Eigen::Vector3d(1, 0, 1),
                                      Eigen::Vector3d(1, 1, 0));

#pragma omp critical
            {
                global_edges_.push_back(PoseGraphOpt::PoseGraphEdge(sub_sourceID, sub_targetID, transformationICP, informationICP, true));
            }
            cout << GREEN << "adding global edge: [" << sub_sourceID << "<==>" << sub_targetID << "], information matrix is: \n" << informationICP
                 << "\n transformation: \n" << transformationICP << BOLDWHITE << endl;
        }
    }

    std::sort(global_edges_.begin(), global_edges_.end(), compareEdge);
    for (int edgeID = 0; edgeID < global_edges_.size(); ++edgeID)
    {
        cout << "\n edgeID: "<<edgeID<<" information matrix: \n"<<global_edges_[edgeID].information_ << endl;
    }

    //-------------------------------------------------------------------------------------
    //构建pose graph local 部分
    v_globalopt_trans2base = v_pairwise_trans2base;

    for (int it = 0; it < CfgParam.global_opt_itertime; ++it)
    {
        PoseGraphOpt::PoseGraph pose_graph = PoseGraphOpt::PoseGraph();
        pose_graph.nodes_.push_back(PoseGraphOpt::PoseGraphNode(v_globalopt_trans2base[0])); //第0个节点,不要忘记！！！其排序就是id=sourceID，所以注意顺序
        for (int sourceID = 0; sourceID < n_frame - 1; sourceID++)
        {
            //这里的odomentry是第一个节点到第n个节点的变换,来自于pairwise registration
            Eigen::Matrix4d trans_ithframe2base = v_globalopt_trans2base[sourceID + 1];
            pose_graph.nodes_.push_back(PoseGraphOpt::PoseGraphNode(trans_ithframe2base));  //对于待优化的节点node, 这里要的是 从第n个节点到第一个节点的变换，其排序就是id=sourceID，所以注意顺序
        }

        std::vector<PoseGraphOpt::PoseGraphEdge> local_edges_;
        local_edges_.resize(localPairRegisInfo.size());
#pragma omp parallel for
        for (int edgeId = 0; edgeId < localPairRegisInfo.size(); edgeId++)
        {
            //这里pairwise 是由于n作为source， n+1作为target，所以这里的edge是n+1到n的变换
            localPairRegisInfo[edgeId].update_informationmatrix(max_correspondence_distance_fine);
            local_edges_[edgeId] = (PoseGraphOpt::PoseGraphEdge(localPairRegisInfo[edgeId].source_id, localPairRegisInfo[edgeId].target_id,
                                                                localPairRegisInfo[edgeId].transformationICP,
                                                                localPairRegisInfo[edgeId].informationICP,
                                                                !localPairRegisInfo[edgeId].odometry_node));
            cout << "adding pairwise edge: [" << localPairRegisInfo[edgeId].source_id << "<==>" << localPairRegisInfo[edgeId].target_id<< "]"<< endl;
        }

        std::cout << '\n' << "[Optimizing PoseGraph ...]" << std::endl;
        TicToc timer;

        pose_graph.edges_.resize(local_edges_.size());

        for (int edgeID = 0; edgeID < local_edges_.size(); edgeID++)
            pose_graph.edges_[edgeID] = local_edges_[edgeID];

        for (int edgeID = 0; edgeID < global_edges_.size() / (it+1); edgeID++)
        {
            pose_graph.edges_.push_back(global_edges_[edgeID]);
        }

        apply_global_optimization(pose_graph);

        cout << it << " time global optimization cost:" << timer.toc() << endl;

        for (int pointID = 0; pointID < n_frame; ++pointID)
        {
            v_globalopt_trans2base[pointID] = pose_graph.nodes_[pointID].pose_.cast<double>();
        }
        v_iter_poses.push_back(v_globalopt_trans2base);
        for (int edgeID = 0; edgeID < localPairRegisInfo.size(); ++edgeID)
        {
            int src = localPairRegisInfo[edgeID].source_id;
            int tgt = localPairRegisInfo[edgeID].target_id;
            localPairRegisInfo[edgeID].transformationICP = v_globalopt_trans2base[tgt].inverse() * v_globalopt_trans2base[src];
        }
    }
}




#endif //MULTIPATHREGIS_REGIS_FUNS_H
