//
// Created by zjl1 on 2023/10/26.
//

#include <armadillo>
#include "O3dVis.h"

O3dVis::O3dVis()
{
    vis = new open3d::visualization::VisualizerWithKeyCallback();
//        if(!vis)
    //----open3d
    // 创建一个可视化窗口
    open3d::visualization::SetGlobalColorMap(open3d::visualization::ColorMap::ColorMapOption::Gray);
    vis->RegisterKeyCallback(GLFW_KEY_E,
                             [&](open3d::visualization::Visualizer *visu)
                             {
                                 mtx.lock();
                                 update_viz = !update_viz;
                                 mtx.unlock();
                                 return true;
                             });

    vis->CreateVisualizerWindow("Open3D scene", 1920, 540);
//        viz_thread = std::thread(&O3dVis::run_viz, this);
}

O3dVis::~O3dVis()
{
    vis->DestroyVisualizerWindow();
    delete vis;
}

void
O3dVis::run_viz()
{
    vis->Run();
}

void
O3dVis::addPointCloudShow(const std::shared_ptr<open3d::geometry::PointCloud> &pc_show, double wait, bool clean)
{
    if (clean)
        vis->ClearGeometries();

    // 可视化点云
    vis->AddGeometry(pc_show, false);
    vis->UpdateGeometry();
    vis->UpdateRender();

    if (!init_zoomed)
    {
        init_zoomed = true;
        vis->ResetViewPoint(true);
    }

    update_viz = false;

    if (wait > 0)
    {
        int cnter = 0;
        while (!update_viz)
        {
            vis->PollEvents();
#ifdef _WIN32
            Sleep(1);
#else
            usleep(1000);
#endif
            cnter++;
            if(cnter > wait)
                break;
        }
        update_viz = true;
    }
    else
    {
        while (!update_viz)
        {
            vis->PollEvents();
        }
    }
}

