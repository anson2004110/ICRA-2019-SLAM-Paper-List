ICRA 2019 SLAM相关Paper List

1. Deep Learning Session

1.1 E2E-VO/ SLAM

GEN-SLAM: Generative Modeling for Monocular Simultaneous Localization and Mapping（深度学习位姿和深度图）

Keywords: SLAM, Localization, Visual-Based Navigation

Beyond Photometric Loss for Self-Supervised Ego-Motion Estimation（深度学习，自监督的深度和里程计，参考了GeoNet和SfmLearner）

Keywords: SLAM, Visual Learning, Localization

代码：
https://github.com/hlzz/DeepMatchVO

Learning Monocular Visual Odometry through Geometry-Aware Curriculum Learning（深度学习的VO）

Keywords: Localization, Visual Learning, Deep Learning in Robotics and Automation

GANVO: Unsupervised Deep Monocular Visual Odometry and Depth Estimation with Generative Adversarial Networks（深度学习 基于GAN的无监督深度和VO方法）

Keywords: Deep Learning in Robotics and Automation, Localization, Visual Tracking
 
Unsupervised Learning of Monocular Depth and Ego-Motion Using Multiple Masks（无监督深度学习的深度图和位姿网络）

Keywords: Deep Learning in Robotics and Automation, SLAM
 
1.2 E2E Navigation

（AWARD）Variational End-To-End Navigation and Localization（端到端定位导航）

Keywords: Deep Learning in Robotics and Automation, Computer Vision for Transportation, Autonomous Vehicle Navigation
 
Deep Reinforcement Learning of Navigation in a Complex and Crowded Environment with a Limited Field of View（强化学习机器人视觉导航）

Keywords: Deep Learning in Robotics and Automation, Collision Avoidance, Service Robots
 
Generalization through Simulation: Integrating Simulated and Real Data into Deep Reinforcement Learning for Vision-Based Autonomous Flight（强化学习的无人机自主导航）

Keywords: Deep Learning in Robotics and Automation

代码
https://github.com/gkahn13/GtS

1.3 Feature & VPR

（AWARD）Learning Scene Geometry for Visual Localization in Challenging Conditions （RGB和Depth中找出场景的结构化描述特征用于VPR）

Keywords: Localization, RGB-D Perception, Computer Vision for Other Robotic Applications
 
Localizing Discriminative Visual Landmarks for Place Recognition（VPR路标的显著性检测）

Keywords: Localization, Visual-Based Navigation, Computer Vision for Automation
 
Improving Keypoint Matching Using a Landmark-Based Image Representation（深度学习地标区域描述符和特征点匹配）

Keywords: SLAM, Localization
 
A Comparison of CNN-Based and Hand-Crafted Keypoint Descriptors（传统和深度学习特征描述子的光照和角度变化下的性能分析）

Keywords: SLAM, Visual-Based Navigation, Deep Learning in Robotics and Automation
 
A Multi-Domain Feature Learning Method for Visual Place Recognition（迁移学习的特征学习用于场景识别）

Keywords: Localization, SLAM, Performance Evaluation and Benchmarking
 
Night-To-Day Image Translation for Retrieval-Based Localization（图像迁移方法的的位置定位）

Keywords: Localization, Visual Learning, Autonomous Vehicle Navigation

2D3D-MatchNet: Learning to Match Keypoints across 2D Image and 3D Point Cloud（深度学习，2D3D数据下的匹配特征点提取网络）
Feng, Mengdan	National University of Singapore

Keywords: Deep Learning in Robotics and Automation, Visual Learning, Localization

Look No Deeper: Recognizing Places from Opposing Viewpoints under Varying Scene Appearance Using Single-View Depth Estimation（用深度学习的深度预测来完成反向视角下的VPR）

Keywords: Localization, Deep Learning in Robotics and Automation
 
Multi-Process Fusion: Visual Place Recognition Using Multiple Image Processing Methods——IRAL（图像上多信息融合做VPR）

Keywords: Localization, Visual-Based Navigation
 
1.4 Depth & Disparity

（AWARD）Geo-Supervised Visual Depth Prediction（深度图网络）

Keywords: Visual Learning, Sensor Fusion

代码
https://github.com/feixh/GeoSup

FastDepth: Fast Monocular Depth Estimation on Embedded Systems（178fps TX2上的224x224深度图计算方法）

Keywords: Deep Learning in Robotics and Automation, Range Sensing, Computer Vision for Other Robotic Applications

代码
http://fastdepth.mit.edu
https://github.com/dwofk/fast-depth

SuperDepth: Self-Supervised, Super-Resolved Monocular Depth Estimation

Keywords: Deep Learning in Robotics and Automation, Visual Learning, Mapping

Depth Completion with Deep Geometry and Context Guidance（稀疏深度图补齐网络）

Keywords: RGB-D Perception, Computer Vision for Other Robotic Applications
 
Self-Supervised Sparse-To-Dense: Self-Supervised Depth Completion from LiDAR and Monocular Camera（自监督学习的Lidar深度数据补齐）

Keywords: Visual Learning, RGB-D Perception, Sensor Fusion

代码
https://github.com/fangchangma/self-supervised-depth-completion

Self-Supervised Learning for Single View Depth and Surface Normal Estimation（自监督的深度和法向图估计）

Keywords: Deep Learning in Robotics and Automation, Visual Learning, Mapping
 
Plug-And-Play: Improve Depth Prediction Via Sparse Data Propagation（循环优化深度图）

Keywords: Deep Learning in Robotics and Automation, RGB-D Perception, Computer Vision for Automation

Depth Generation Network: Estimating Real World Depth from Stereo and Depth Images（左右图生成深度图网络）

Keywords: AI-Based Methods, RGB-D Perception, Range Sensing
 
Anytime Stereo Image Depth Estimation on Mobile Devices（双目深度图匹配计算方法，快速）

Keywords: Deep Learning in Robotics and Automation, Computer Vision for Automation, Computer Vision for Other Robotic Applications

代码
https://github.com/mileyan/AnyNet

UWStereoNet: Unsupervised Learning for Depth Estimation and Color Correction of Underwater Stereo Imagery（深度学习的双目立体匹配）

Keywords: Marine Robotics, Deep Learning in Robotics and Automation, Computer Vision for Other Robotic Applications
 
Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations（深度学习语义和深度的分割网络）

Keywords: Visual Learning, Semantic Scene Understanding, SLAM

代码
https://github.com/DrSleep/multi-task-refinenet

DSNet: Joint Learning for Scene Segmentation and Disparity Estimation（深度学习左右图估计语义分割和深度图）

Keywords: Semantic Scene Understanding, Deep Learning in Robotics and Automation, Object Detection, Segmentation and Categorization
 
SweepNet: Wide-Baseline Omnidirectional Depth Estimation（宽基线，多摄像头的深度估计方法）

Keywords: Omnidirectional Vision, Computer Vision for Automation, Deep Learning in Robotics and Automation
 
A Supervised Approach to Predicting Noise in Depth Images（预测深度图的噪声区域）

Keywords: RGB-D Perception, Perception for Grasping and Manipulation, Deep Learning in Robotics and Automation

1.5 Point Cloud Segmentation

SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud （深度学习，雷达数据分割路面上的物体）

Keywords: Object Detection, Segmentation and Categorization, Semantic Scene Understanding, AI-Based Methods

https://github.com/BichenWuUCB/SqueezeSeg
https://github.com/xuanyuzhou98/SqueezeSegV2

Hierarchical Depthwise Graph Convolutional Neural Network for 3D Semantic Segmentation of Point Clouds（点云语义分割方法）

Keywords: Semantic Scene Understanding, AI-Based Methods, RGB-D Perception

1.6 Autonomous Vehicle

Learning to Drive from Simulation without Real World Labels（自动驾驶中的学习方法）

Keywords: Deep Learning in Robotics and Automation, Visual Learning, Learning from Demonstration

Learning to Drive in a Day（强化学习的自动驾驶）

Keywords: AI-Based Methods, Deep Learning in Robotics and Automation, Computer Vision for Transportation
 
Building a Winning Self-Driving Car in Six Months（与Uber 合作的自动驾驶平台）

Keywords: Autonomous Vehicle Navigation, Intelligent Transportation Systems, Computer Vision for Transportation

Multimodal Spatio-Temporal Information in End-To-End Networks for Automotive Steering Prediction （BMW合作的自动驾驶）

Keywords: Autonomous Vehicle Navigation, Deep Learning in Robotics and Automation, Visual Learning
 
Monocular Semantic Occupancy Grid Mapping with Convolutional Variational Encoder-Decoder Networks——IRAL（单目图生成避障图）

Keywords: Semantic Scene Understanding, Object Detection, Segmentation and Categorization, Computer Vision for Transportation
 
2.Deep Learning + Traditional SLAM Session

2.1 SLAM

CNN-SVO: Improving the Mapping in Semi-Direct Visual Odometry Using Single-Image Depth Prediction（深度学习，CNN参与深度估计，并且用于SVO）

Keywords: SLAM, Localization, Visual Learning

https://github.com/yan99033/CNN-SVO

Real-Time Monocular Object-Model Aware Sparse SLAM（深度学习，语义物体SLAM）

Keywords: SLAM
 
Semantic Mapping for View-Invariant Relocalization（物体关联的SLAM，可以对视角变化的重定位鲁棒）

Keywords: Semantic Scene Understanding, Visual-Based Navigation, SLAM
 
A Unified Framework for Mutual Improvement of SLAM and Semantic Segmentation（语义和SLAM相互促进的方法）

Keywords: SLAM, Object Detection, Segmentation and Categorization, RGB-D Perception

Multimodal Semantic SLAM with Probabilistic Data Association（图优化，地图数据关联）

Keywords: SLAM, Visual-Based Navigation, Localization
 
Efficient Constellation-Based Map-Merging for Semantic SLAM（数据关联，地图的点的融合和语义slam）

Keywords: SLAM, Localization, Autonomous Vehicle Navigation
 
Enhancing V-SLAM Keyframe Selection with an Efficient ConvNet for Semantic Analysis（深度学习，根据图像质量和语义信息选择关键帧）

Keywords: Computer Vision for Other Robotic Applications, Semantic Scene Understanding, Deep Learning in Robotics and Automation

https://github.com/Shathe/MiniNet

Pose Graph Optimization for Unsupervised Monocular Visual Odometry（深度学习VO和传统图优化方法结合）

Keywords: Deep Learning in Robotics and Automation, SLAM, Localization
 
Learning Wheel Odometry and IMU Errors for Localization（里程计和IMU融合定位）

Keywords: Localization, Deep Learning in Robotics and Automation, Autonomous Vehicle Navigation

https://github.com/CAOR-MINES-ParisTech/lwoi

Global Localization with Object-Level Semantics and Topology（3D语义地图的重定位，利用物体的拓扑关系完成数据关联）

Keywords: Localization, Semantic Scene Understanding, Computer Vision for Other Robotic Applications

Robust Object-Based SLAM for High-Speed Autonomous Navigation（基于平面上物体的SLAM）

Keywords: Aerial Systems: Perception and Autonomy, Autonomous Vehicle Navigation, Mapping
 
2.2 Mapping & 3D Reconstruction

MID-Fusion: Octree-Based Object-Level Multi-Instance Dynamic SLAM (深度学习，语义的三维重建，去除动态物体，跟踪相机和物体的位姿)

Keywords: SLAM, Mapping, RGB-D Perception
 
Probabilistic Projective Association and Semantic Guided Relocalization for Dense Reconstruction（语义信息促进SLAM建图的工作，SLAM中的跟踪和回环用到了语义的分割结果）

Keywords: SLAM, RGB-D Perception, Object Detection, Segmentation and Categorization

Dense 3D Visual Mapping Via Semantic Simplification（3D重建中的物体分类，用于判断哪些点需要细节，哪些点就只需要简化）

Keywords: Mapping, Semantic Scene Understanding

DeepFusion: Real-Time Dense 3D Reconstruction for Monocular SLAM Using Single-View Depth and Gradient Predictions（深度学习，用CNN预计的depth来完成关键帧深度的估计，从而结合slam完成密集三维重建）

Keywords: SLAM, Deep Learning in Robotics and Automation, Mapping
 
Sparse2Dense: From Direct Sparse Odometry to Dense 3D Reconstruction—IRAL （深度学习，稀疏到密集的三维重建，CNN估计深度，法向图用于密集重建）

Keywords: SLAM, Mapping, Visual Learning

 
3. Traditional SLAM Session

3.1 SLAM——Direct, 2D/3D feature, Lidar SLAM

FMD Stereo SLAM: Fusing MVG and Direct Formulation towards Accurate and Fast Stereo SLAM（中科院，特征点法和直接法结合）

Keywords: SLAM, Localization, Mapping
 
RESLAM: A Real-Time Robust Edge-Based SLAM System （边缘SLAM）

Keywords: SLAM, Visual-Based Navigation, RGB-D Perception

代码：
https://github.com/fabianschenk/RESLAM
https://github.com/fabianschenk/REVO

Leveraging Structural Regularity of Atlanta World for Monocular SLAM（Atlanta世界坐标系下的边缘线约束SLAM）

Keywords: SLAM, Localization, Mapping
 
Illumination Robust Monocular Direct Visual Odometry for Outdoor Environment Mapping（抗光照变化的直接法视觉里程计）
 
Loosely-Coupled Semi-Direct Monocular SLAM——IRAL（直接法跟踪，特征点法做地图优化和回环）

Keywords: SLAM, Localization, Mapping

代码
https://github.com/sunghoon031/LCSD_SLAM

3D Keypoint Repeatability for Heterogeneous Multi-Robot SLAM（多机器人系统的不同传感器的特征点匹配，3D关键点KPQ-SI和NARF两个特征点比较适合用于Loopclosure和多机器人重定位中）

Keywords: SLAM, Performance Evaluation and Benchmarking, Mapping
 
Local Descriptor for Robust Place Recognition Using LiDAR Intensity——IRAL （ISHOT点云描述子用于定位）

Keywords: Localization, Field Robots, Autonomous Vehicle Navigation
 
1-Day Learning, 1-Year Localization: Long-Term LiDAR Localization Using Scan Context Image——IRAL（激光雷达的长期定位方法）

Keywords: Localization, Range Sensing, SLAM
 
3.2 SLAM——Pose Optimization

On-Line 3D Active Pose-Graph SLAM Based on Key Poses Using Graph Topology and Sub-Maps（位姿优化，子地图）

Keywords: SLAM, Motion and Path Planning
 
MH-iSAM2: Multi-Hypothesis iSAM Using Bayes Tree and Hypo-Tree（非线性增量优化，解决SLAM歧义）

Keywords: SLAM, Localization, Mapping
 
Visual SLAM: Why Bundle Adjust?（BA的替代优化方法，解决纯旋转和弱平移下的位姿估计）

Keywords: SLAM
 
Modeling Perceptual Aliasing in SLAM Via Discrete-Continuous Graphical Models——IRAL （离散连续图模型的优化方法）

Keywords: SLAM, Sensor Fusion, Optimization and Optimal Control
 
POSEAMM: A Unified Framework for Solving Pose Problems Using an Alternating Minimization Method（使用交替最小化方法解决姿势优化问题的统一框架）

Keywords: Computer Vision for Automation, Omnidirectional Vision, Localization
 
Visual-Odometric Localization and Mapping for Ground Vehicles Using SE(2)-XYZ Constraints（平面移动机器人的位姿估计约束模型）

Keywords: Localization, SLAM, Sensor Fusion

https://github.com/izhengfan/se2lam

Direct Relative Edge Optimization, a Robust Alternative for Pose Graph Optimization（边缘约束的图优化）

Keywords: SLAM, Mapping, Multi-Robot Systems
 
A White-Noise-On-Jerk Motion Prior for Continuous-Time Trajectory Estimation on SE(3) （位姿估计方法）

Keywords: SLAM

Low-Latency Visual SLAM with Appearance-Enhanced Local Map Building（一种快速局部地图的策略）

Keywords: SLAM
 
3.3 SLAM——VIO/ VISLAM

Fast and Robust Initialization for Visual-Inertial SLAM（VINS初始化）

Keywords: SLAM, Mapping, Localization
 
Visual-Inertial Navigation: A Concise Review

Keywords: Autonomous Vehicle Navigation, Localization, Sensor Fusion

https://github.com/rpng

Tightly-Coupled Aided Inertial Navigation with Point and Plane Features（点面特征的VINS系统）

Keywords: Range Sensing, Sensor Fusion, SLAM
 
Tightly-Coupled Visual-Inertial Localization and 3D Rigid-Body Target Tracking——IRAL（VINS和跟踪物体紧融合）

Keywords: Localization, Visual Tracking, SLAM
 
Aided Inertial Navigation: Unified Feature Representations and Observability Analysis（点，线，面多特征融合的VINS系统）

Keywords: Localization, SLAM, Visual-Based Navigation
 
A Linear-Complexity EKF for Visual-Inertial Navigation with Loop Closures（一种MSCKF的VINS方法，带回环）

Keywords: Localization, SLAM, Mapping
 
Multi-Camera Visual-Inertial Navigation with Online Intrinsic and Extrinsic Calibration（多相机VINS系统的在线标定相机，IMU内外参数方法）

Keywords: Visual-Based Navigation, Sensor Fusion, Localization
 
Sensor-Failure-Resilient Multi-IMU Visual-Inertial Navigation（一种多IMU的VINS系统）

Keywords: Localization, SLAM, Failure Detection and Recovery
 
Efficient 2D-3D Matching for Multi-Camera Visual Localization（多camera imu的重定位）

Keywords: Localization, Computer Vision for Transportation, Omnidirectional Vision
 
Keyframe-Based Direct Thermal–Inertial Odometry（低质量图像下的VIO方法，基于关键帧的直接法，可以借鉴他借鉴低照度下的vo问题）

Keywords: Localization, Sensor Fusion, Field Robots
 
Improving the Robustness of Visual-Inertial Extended Kalman Filtering（VINS 系统姿态估计提升方案）

Keywords: Visual-Based Navigation, Aerial Systems: Perception and Autonomy, Robust/Adaptive Control of Robotic Systems
 
Towards Fully Dense Direct Filter-Based Monocular Visual-Inertial Odometry（密集直接法VINS系统）

Keywords: Sensor Fusion, Visual-Based Navigation, Localization
 
Experimental Comparison of Visual-Aided Odometry Methods for Rail Vehicles—IRAL  （在火车的数据集上实验比对VO、VIO方法）

Keywords: Computer Vision for Transportation, Intelligent Transportation Systems, SLAM
 
RaD-VIO: Rangefinder-Aided Downward Visual-Inertial Odometry（测距融合VIO）

Keywords: Aerial Systems: Perception and Autonomy, Localization, Performance Evaluation and Benchmarking
 
3.4 SLAM——Multi-sensor Fusion

Accurate Direct Visual-Laser Odometry with Explicit Occlusion Handling and Plane Detection（激光雷达融合视觉定位，区分平面和非平面的特征点）

Keywords: SLAM, Localization
 
Robust Pose-Graph SLAM Using Absolute Orientation Sensing（激光雷达+天花板摄像头SLAM）

Keywords: SLAM, Industrial Robots
 
Tightly Coupled 3D Lidar Inertial Odometry and Mapping（雷达和IMU融合）

Keywords: Computer Vision for Automation, Sensor Fusion, SLAM
 
IN2LAMA: INertial Lidar Localisation and Mapping（IMU和Lidar融合的SLAM）

Keywords: Mapping, SLAM, Sensor Fusion
 
ROVO: Robust Omnidirectional Visual Odometry for Wide-Baseline Wide-FOV Camera Systems（多鱼眼SLAM）

Keywords: SLAM, Omnidirectional Vision, Autonomous Vehicle Navigation
 
3.5 Depth & Mapping & 3D Reconstruction

ScalableFusion: High-Resolution Mesh-Based Real-Time 3D Reconstruction（三维重建）

Keywords: SLAM, Mapping, RGB-D Perception
 
Surfel-Based Dense RGB-D Reconstruction with Global and Local Consistency（用SFM计算全局的关键帧位姿，同时用slam方法计算局部相邻帧的位姿，然后用FGO factor graph optimization方法将全局和局部信息融合计算出密集三维重建）

Keywords: SLAM, Localization, Mapping

Real-Time Scalable Dense Surfel Mapping

Keywords: Mapping, Sensor Fusion, Aerial Systems: Perception and Autonomy

代码
https://github.com/HKUST-Aerial-Robotics/DenseSurfelMapping

Real-Time Dense Mapping for Self-Driving Vehicles Using Fisheye Cameras（鱼眼相机的密集三维重建）

Keywords: Mapping, Computer Vision for Transportation, Omnidirectional Vision

https://zhpcui.github.io/projects/arxiv18_densemapping/

Real Time Dense Depth Estimation by Fusing Stereo with Sparse Depth Measurements（用TOF辅助双目密集匹配算法）

Keywords: Range Sensing, Aerial Systems: Perception and Autonomy

Dense Surface Reconstruction from Monocular Vision and LiDAR（雷达和视觉融合三维重建）

Keywords: Mapping, SLAM, Range Sensing

Incremental Visual-Inertial 3D Mesh Generation with Structural Regularities（VIO输出的稀疏点做三维重建的三角面片）

Keywords: SLAM, Visual-Based Navigation, Sensor Fusion
 
OVPC Mesh: 3D Free-Space Representation for Local Ground Vehicle Navigation（3D Mesh表示方法，用于无人车避障）

Keywords: Autonomous Vehicle Navigation, Field Robots, Mapping

KO-Fusion: Dense Visual SLAM with Tightly-Coupled Kinematic and Odometric Tracking（机器人运动学与里程计结合的密集三维重建）

Keywords: SLAM, Sensor Fusion, Kinematics
 
3D Surface Reconstruction Using a Two-Step Stereo Matching Method Assisted with Five Projected Patterns（三维双目结构光重建设备）

Keywords: Computer Vision for Automation, Range Sensing, Computer Vision for Other Robotic Applications

3.6 Localization——Lidar / Vision

Beyond Point Clouds: Fisher Information Field for Active Visual Localization（3D landmark来做视觉定位）

Keywords: Visual-Based Navigation, Localization, Motion and Path Planning
 
Effective Visual Place Recognition Using Multi-Sequence Maps—IRAL（场景识别定位）

Keywords: Localization
 
MRS-VPR: A Multi-Resolution Sampling Based Visual Place Recognition Method（场景识别和回环检测，高效、多尺度、粗到细的长期序列VPR）

Keywords: SLAM, Deep Learning in Robotics and Automation, Visual Learning
 
Probabilistic Appearance-Based Place Recognition through Bag of Tracked Words——IRAL （BTW场景定位）

Keywords: SLAM, Visual-Based Navigation, Recognition

Geometric Relation Distribution for Place Recognition——IRAL（激光雷达的重定位和回环）

Keywords: Mapping, Localization, Range Sensing

代码
https://github.com/dlr1516/grd

3.7 Others

A-SLAM: Human-In-The-Loop Augmented SLAM（交互式SLAM地图和位姿修正方法）

Keywords: SLAM, Virtual Reality and Interfaces, Wheeled Robots

Iteratively Reweighted Midpoint Method for Fast Multiple View Triangulation——IRAL （三角化误差消除方法）

Keywords: SLAM, Mapping
 
CELLO-3D: Estimating the Covariance of ICP in the Real World（点云ICP）

Keywords: SLAM, Range Sensing, Learning and Adaptive Systems
 
4. SLAM Evaluation & Datasets

The Open Vision Computer: An Integrated Sensing and Compute System for Mobile Robots（宾夕法尼亚大学无人机 集成化方案）

Keywords: Aerial Systems: Perception and Autonomy
   
SLAMBench 3.0: Systematic Automated Reproducible Evaluation of SLAM Systems for Robot Vision Challenges and Scene Understanding（SLAM方法和数据集）

Keywords: SLAM, Performance Evaluation and Benchmarking, Semantic Scene Understanding

Project AutoVision: Localization and 3D Scene Perception for an Autonomous Vehicle with a Multi-Camera System（自动驾驶系统，数据，GNSS+IMU+Camera的稀疏建图和定位）
 
BLVD: Building a Large-Scale 5D Semantics Benchmark for Autonomous Driving

Keywords: Performance Evaluation and Benchmarking, Intelligent Transportation Systems

https://github.com/VCCIV/BLVD/

Characterizing Visual Localization and Mapping Datasets（RGBD数据集）

Keywords: Performance Evaluation and Benchmarking, SLAM
 
Are We Ready for Autonomous Drone Racing? the UZH-FPV Drone Racing Dataset（stereo camera，event-camera数据集）

Keywords: Performance Evaluation and Benchmarking, Localization, Aerial Systems: Perception and Autonomy
 
An Empirical Evaluation of Ten Depth Cameras for Indoor Environments——IRAM IEEE Robotics & Automation Magazine  （深度传感器的评测）

Keywords: Performance Evaluation and Benchmarking, Range Sensing, RGB-D Perception

5. ICRA AWARD list

Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks（触觉，视觉反馈的机器人装配）

Keywords: Deep Learning in Robotics and Automation, Perception for Grasping and Manipulation, Sensor-based Control

Closing the Sim-To-Real Loop: Adapting Simulation Randomization with Real World Experience（虚拟数据到真实数据的迁移）

Keywords: Learning and Adaptive Systems, Model Learning for Control, Deep Learning in Robotics and Automation
