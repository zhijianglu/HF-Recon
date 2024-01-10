# Multi-View Point Cloud Registration for High-Fidelity 3D Reconstruction

The purpose of this repository is to establish a high-fidelity 3D reconstruction system based on multi-frame point cloud registration. The point cloud data is derived from various viewpoints around the target object, acquired through scanning. The data sources can be depth cameras or LiDAR (Light Detection and Ranging). 

## Paper

The article is currently undergoing peer review:

<div align=center><img src="resources/paper.png" style="zoom:20%;" />

## Reconstruction results

- For simulated data (rendering by Open3D):

  | deagon                                                       | deer                                                         | vase                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="resources/dragon.png" style="max-width: 40%; height: auto; margin: 0 5px;" /> | <img src="resources/deer.png" style="max-width: 40%; height: auto; margin: 0 5px;" /> | <img src="resources/vase.png" style="max-width: 40%; height: auto; margin: 0 5px;" /> |

  

- For real-world data:

| deagon                                                       | deer                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="resources/status.png" style="max-width: 40%; height: auto; margin: 0 5px;" /> | <img src="resources/sofa.png" style="max-width: 40%; height: auto; margin: 0 5px;" /> |

## Requirements

- PCL (>1.10)
- Eigen3(3.3.4)
- OpenCV (>4.0)
- Open3D
- [Teaser-pp][https://github.com/MIT-SPARK/TEASER-plusplus]

## Preparation 

1.  Install a feature point-based registration algorithm according to [Teaser-pp][https://github.com/MIT-SPARK/TEASER-plusplus]'s guidance, to serve as the initial registration pose for our algorithm.

2. Download data from [GoogleDrive](https://drive.google.com/drive/folders/1zLcwRlwguh5txwxgK075HkXIG-hlaE5V?usp=sharing) and place it in the `data` folder.

3. build project:

   ` mkdir build && cd build `

   `cmake .. && make` 

## Usage

1. Firstly, use [Teaser-pp][https://github.com/MIT-SPARK/TEASER-plusplus] to generate the initial pose, which is based on a feature point matching method, allowing for a rough alignment of the point cloud sequence.

   `./teaser_coarse_align ../cfg/simrecon_params.yaml`

2. Start pairwise and global registration.

   `./multi_way_align_sim ../cfg/simrecon_params.yaml `

## Simulation and Real-world Datasets

All dataset can be downloded at [GoogleDrive](https://drive.google.com/drive/folders/1zLcwRlwguh5txwxgK075HkXIG-hlaE5V?usp=sharing)  

The real-world data is automatically acquired through the omnidirectional collection platform we designed:



<div align=center><img src="resources/platform.png" style="zoom:30%;" />



Simulation data is collected using a Kinect camera in the Gazebo platform:


<div align=center><img src="resources/gazebo.png" style="zoom:50%;" />



## Note.

### The code for the Gazebo simulation data platform, along with more details of our method, will be published subsequently.
