//
// Created by zjl1 on 2023/8/31.
//

#ifndef OPEN3D_DEMOS_GLOBAL_H
#define OPEN3D_DEMOS_GLOBAL_H


#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <glog/logging.h>
#include <boost/thread.hpp>
#include "getfile.h"
#include "PinholeCam.h"

using namespace std;
enum class PairRegMethod {
    FRICP,
    AAICP,
    COLORICP,
    SPARSEICP_PT,
    SPARSEICP_PL,
    OPEN3DICP,
    NONE // 用于未知或不匹配的输入
};

struct param
{
//  data info=================================================
    string data_root_path;

    string data_label;
    vector<string> all_methods;
    int selected_method_id;
    PairRegMethod pairRegMethod;

    int start_idx;
    double voxel_size;
    double teaser_align_voxel;
    double scale_max_correspondence_distance_coarse;
    double scale_max_correspondence_distance_fine;
    double global_max_correspondence_distance;
    double edge_prune_threshold;
    double preference_loop_closure;
    double global_edge_coview_rate;
    double global_coview_distance;
    double global_coview_color_diff;
    double scale_normal_radius;
    double max_n_normal_pt;
    double color_weight_mu;
    int global_opt_itertime;

    int trig_mesh_generate;
    int poission_recon_depth;
    double mesh_cut_scale;
    double mesh_voxel;

    int is_add_noise;
    vector<double> noise_sigma_range;
    vector<double> noise_distance_range;

    int global_max_iteration;
    int final_align_spin;
    int show_pairalign;
    int apply_mesh_recon;

    vector<string> excluded_dir;
    vector<string> test_data_list;

    string pairwise_reg_method;

};

extern param CfgParam;

//extern Ptr<aruco::Dictionary> dictionary;
//extern Ptr<aruco::DetectorParameters> detectorParameters;
void
readParameters(std::string config_file);

PairRegMethod determinePairRegMethod(const std::string& method);

#endif //OPEN3D_DEMOS_GLOBAL_H
