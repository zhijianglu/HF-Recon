//
// Created by zjl1 on 2023/10/26.
//

#ifndef MULTIPATHREGIS_O3DVIS_H
#define MULTIPATHREGIS_O3DVIS_H

#include <open3d/geometry/KDTreeSearchParam.h>
#include <open3d/visualization/rendering/Open3DScene.h>
#include <open3d/visualization/visualizer/Visualizer.h>
#include <open3d/visualization/visualizer/VisualizerWithKeyCallback.h>
#include <open3d/visualization/visualizer/GuiVisualizer.h>
#include "open3d/Open3D.h"

class O3dVis
{
public:
    O3dVis();


    ~O3dVis();

    void
    run_viz();

    void
    addPointCloudShow(const std::shared_ptr<open3d::geometry::PointCloud> &pc_show, double wait = 0, bool clean = true);

    open3d::visualization::VisualizerWithKeyCallback *vis;
    bool update_viz = false;
    bool init_zoomed = false;
    bool shouldQuit = false;
    std::mutex mtx;
    std::thread viz_thread;
};



#endif //MULTIPATHREGIS_O3DVIS_H
