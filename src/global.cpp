//
// Created by zjl1 on 2023/8/31.
//

#include "global.h"


#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "opencv2/opencv.hpp"

param CfgParam;

PairRegMethod determinePairRegMethod(const std::string& method) {
    if (method == "fricp") {
        return PairRegMethod::FRICP;
    } else if (method == "aaicp") {
        return PairRegMethod::AAICP;
    } else if (method == "coloricp") {
        return PairRegMethod::COLORICP; //
    } else if (method == "sparseicp-pt") {
        return PairRegMethod::SPARSEICP_PT; //sparseicp
    } else if (method == "sparseicp-pl") {
        return PairRegMethod::SPARSEICP_PL; //sparseicp
    } else if (method == "open3dicp") {
        return PairRegMethod::OPEN3DICP;
    } else {
        return PairRegMethod::NONE; // 如果输入不匹配任何已知方法
    }
}


void
readParameters(std::string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    cout << "start loading parameters from:" << config_file <<endl;

//    detectorParameters->cornerRefinementMethod = aruco::CORNER_REFINE_APRILTAG;
//    detectorParameters->cornerRefinementWinSize = 20;


    fsSettings["selected_method_id"] >> CfgParam.selected_method_id;
    fsSettings["pairwise_reg_method"] >> CfgParam.all_methods;
    if(CfgParam.selected_method_id>CfgParam.all_methods.size() || CfgParam.selected_method_id<0){
        CfgParam.pairRegMethod = PairRegMethod::NONE;
    }else{
        CfgParam.pairwise_reg_method = CfgParam.all_methods[CfgParam.selected_method_id];
        CfgParam.pairRegMethod = determinePairRegMethod(CfgParam.pairwise_reg_method);
    }


    fsSettings["data_root_path"] >> CfgParam.data_root_path;
    fsSettings["test_data_list"] >> CfgParam.test_data_list;

    fsSettings["scale_max_correspondence_distance_coarse"] >> CfgParam.scale_max_correspondence_distance_coarse;
    fsSettings["scale_max_correspondence_distance_fine"] >> CfgParam.scale_max_correspondence_distance_fine;
    fsSettings["start_idx"] >> CfgParam.start_idx;

    fsSettings["is_add_noise"] >> CfgParam.is_add_noise;
    fsSettings["noise_sigma_range"] >> CfgParam.noise_sigma_range;
    fsSettings["noise_distance_range"] >> CfgParam.noise_distance_range;
    fsSettings["data_label"] >> CfgParam.data_label;
    fsSettings["teaser_align_voxel"] >> CfgParam.teaser_align_voxel;

    fsSettings["pairwise_align"]["voxel_size"] >> CfgParam.voxel_size;
    fsSettings["pairwise_align"]["scale_normal_radius"] >> CfgParam.scale_normal_radius;
    fsSettings["pairwise_align"]["max_n_normal_pt"] >> CfgParam.max_n_normal_pt;

    fsSettings["global_optimize"]["global_max_correspondence_distance"] >> CfgParam.global_max_correspondence_distance;
    fsSettings["global_optimize"]["edge_prune_threshold"] >> CfgParam.edge_prune_threshold;
    fsSettings["global_optimize"]["preference_loop_closure"] >> CfgParam.preference_loop_closure;
    fsSettings["global_optimize"]["global_max_iteration"] >> CfgParam.global_max_iteration;
    fsSettings["global_optimize"]["global_edge_coview_rate"] >> CfgParam.global_edge_coview_rate;
    fsSettings["global_optimize"]["global_coview_distance"] >> CfgParam.global_coview_distance;
    fsSettings["global_optimize"]["global_coview_color_diff"] >> CfgParam.global_coview_color_diff;
    fsSettings["global_optimize"]["color_weight_mu"] >> CfgParam.color_weight_mu;
    fsSettings["global_optimize"]["global_opt_itertime"] >> CfgParam.global_opt_itertime;

    fsSettings["visualize"]["show_pairalign"] >> CfgParam.show_pairalign;
    fsSettings["visualize"]["final_align_spin"] >> CfgParam.final_align_spin;

    fsSettings["mesh_recon"]["apply_mesh_recon"] >> CfgParam.apply_mesh_recon;
    fsSettings["mesh_recon"]["mesh_cut_scale"] >> CfgParam.mesh_cut_scale;
    fsSettings["mesh_recon"]["mesh_voxel"] >> CfgParam.mesh_voxel;
    fsSettings["mesh_recon"]["trig_mesh_generate"] >> CfgParam.trig_mesh_generate;
    fsSettings["mesh_recon"]["poission_recon_depth"] >> CfgParam.poission_recon_depth;

    fsSettings["others"]["excluded_dir"] >> CfgParam.excluded_dir;

}
