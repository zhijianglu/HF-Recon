//
// Created by zjl1 on 2023/8/31.
//

#ifndef OPEN3D_DEMOS_TOOLS_H
#define OPEN3D_DEMOS_TOOLS_H

#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


#include <vector>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;
using namespace Eigen;

#include <iostream>

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */



static
double
RCalcRotationError(Eigen::Matrix3f Rg, Eigen::Matrix3f Re)
{
    // 计算Rg * Re的转置的迹
    float trace = (Rg * Re.transpose()).trace();

    float cos_angle = (trace - 1.0f) / 2.0f;
//    cos_angle = std::max(-1.0, std::min(cos_angle, 1.0)); // 限制在 [-1, 1] 范围内

    if(cos_angle > 1.0)
        cos_angle = 2.0-cos_angle;
    else if(cos_angle < -1.0)
        cos_angle = 2.0+cos_angle;

    // 计算角度
    double error_angle = std::acos(cos_angle);

//    std::cout << "角度（弧度）: " << error_angle << std::endl;
//    std::cout << "角度（度）: " << (error_angle * 180 / M_PI) << std::endl;

    return (error_angle * 180.0f / 3.14159265358979323846f);
}

Eigen::Vector2f
evoPose(Eigen::Matrix4d &T_est, Eigen::Matrix4d &T_gt)
{
    Eigen::Vector2f extrinsic_error;
    extrinsic_error.x() = RCalcRotationError(T_gt.block(0, 0, 3, 3).cast<float>(), T_est.block(0, 0, 3, 3).cast<float>() );

    Eigen::Vector3d error_t = T_est.block(0, 3, 3, 1) - T_gt.block(0, 3, 3, 1);
    extrinsic_error.y() = error_t.norm();
//    cout << "extrinsic_error:"<<extrinsic_error.transpose() << endl;
//    open3d::geometry::PointCloud pc_src_gt;
//    pc_src_gt = pc_src;
//    pc_src.Transform(T_est);
//    pc_src_gt.Transform(T_gt);
//
//    //计算两个点云的距离
//    double sum;
//    for (int i = 0; i < pc_src.points_.size(); ++i)
//    {
//        sum += (pc_src.points_[i] - pc_src_gt.points_[i]).norm();
//    }
//    extrinsic_error.y() = sum / double(pc_src.points_.size());


    return extrinsic_error;
}

#endif //OPEN3D_DEMOS_TOOLS_H
