//
// Created by lab on 2022/8/25.
//

#ifndef FRICP_VISUALIZATION_H
#define FRICP_VISUALIZATION_H
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h> //PCL is migrating to PointCloud2

#include <pcl/common/common_headers.h>

//will use filter objects "passthrough" and "voxel_grid" in this example
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h> //PCL的PCD格式文件的输入输出头文件
#include <pcl/point_types.h> //PCL对各种格式的点的支持头文件
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
//#include <pcl/filters/statistical_outlier_oval.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace pcl;

//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_VIS_H
#define CHECKERBOARD_VIS_H
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <mutex>
#include <thread>
#include <binders.h>
#include <boost/thread.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <opencv2/imgproc/types_c.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h> //PCL is migrating to PointCloud2

#include <pcl/common/common_headers.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h> //PCL的PCD格式文件的输入输出头文件
#include <pcl/point_types.h> //PCL对各种格式的点的支持头文件
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
//#include <pcl/filters/statistical_outlier_oval.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

typedef pcl::PointXYZ PoinT;
typedef pcl::PointXYZRGB DisplayType;

typedef pcl::PointCloud<PoinT> PC;
typedef pcl::PointCloud<PoinT>::Ptr PCPtr;

typedef pcl::PointCloud<PoinT> CPC;
typedef pcl::PointCloud<PoinT>::Ptr CPCPtr;

typedef pcl::PointXYZRGB ColorP;
typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPC;
typedef ColorPC::Ptr ColorPCPtr;



using namespace std;
using namespace cv;
using namespace pcl;


//void add_pointClouds_show(visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<PoinT>::Ptr cloud2show, bool remove_before= true, string cloud_id= "cloud", int show_size = 1, int spintime = 0){
//    if (remove_before)
//        viewer->removeAllPointClouds();
//
//    viewer->addPointCloud<PoinT>(cloud2show, cloud_id);
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
//                                             show_size,
//                                             cloud_id);
//    if(spintime==0)
//        viewer->spin();
//    else
//        viewer->spinOnce(spintime);
//}



void
add_pointClouds_show(visualization::PCLVisualizer::Ptr viewer,
                     pcl::PointCloud<PoinT>::Ptr cloud2show,
                     bool remove_before = true,
                     string cloud_id = "cloud",
                     int show_size = 1,
                     int spintime = 0,
                     Eigen::Vector3i color = Eigen::Vector3i(255,255,255))
{

    if (remove_before)
        viewer->removeAllPointClouds();

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb(cloud2show, color.x(), color.y(), color.z());

    viewer->addPointCloud<PoinT>(cloud2show, rgb, cloud_id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             show_size,
                                             cloud_id);


    if (spintime == 0)
        viewer->spin();
    else
        viewer->spinOnce(spintime);
}



void add_pointClouds_show(visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<DisplayType>::Ptr cloud2show, bool remove_before= true, string cloud_id= "cloud", int show_size = 1, int spintime = 0){
    if (remove_before)
        viewer->removeAllPointClouds();

    viewer->addPointCloud<DisplayType>(cloud2show, cloud_id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             show_size,
                                             cloud_id);
    if(spintime==0)
        viewer->spin();
    else
        viewer->spinOnce(spintime);
}

//void eular_cluster()

#endif //CHECKERBOARD_VIS_H




#endif //FRICP_VISUALIZATION_H
