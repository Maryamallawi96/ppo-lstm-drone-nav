# 🛸 PPO-LSTM Drone Navigation
# Memory-Augmented Reinforcement Learning for UAV Navigation using PPO-LSTM

🛸 🛸 This project implements autonomous drone navigation in a 3D indoor simulated environment using Deep Reinforcement Learning (PPO + LSTM). The drone learns to navigate through obstacles toward a target using only LiDAR input, leveraging memory (LSTM) for better decision-making in partially observable environments.
## 🧠 Features

- ✅ PPO + LSTM agent using Stable-Baselines3
- ✅ Navigation using only 2D LiDAR input
- ✅ Memory-augmented for partially observable environments
- ✅ Fully integrated with Gazebo, PX4 SITL, and MAVROS
- ✅ Collision reset, OFFBOARD mode, and reward shaping


## 📁 Package Name
drone_ppolstm_nav

## 🧭 Simulation Overview

| Component         | Description                        |
|------------------|------------------------------------|
| Simulator         | Gazebo 11                          |
| Flight Controller | PX4 Autopilot (SITL)               |
| Middleware        | MAVROS (ROS Noetic)                |
| DRL Algorithm     | PPO with LSTM (Stable-Baselines3)  |
| Sensor            | 720° 2D LiDAR (/scan topic)      |
| Goal              | (33, -2, 0)                         |


## 🧠 Folder Structure

drone_ppolstm_nav/
├── launch/                       # Simulation launch files
├── models/                       # Drone model with LiDAR
├── scripts/                      # Training and environment code
├── worlds/                       # Gazebo world with obstacles
├── CMakeLists.txt
└── package.xml

⚙️ Dependencies
Make sure the following are installed:
PX4-Autopilot
ROS Noetic
Gazebo 11
mavros, mavros_extras
stable-baselines3, gymnasium, rospy, numpy, sensor_msgs, geometry_msgs
Install Python packages:
pip install stable-baselines3[extra] torch tensorboard

🚀 Launching the Simulation
1. Launch PX4 and Gazebo:
roslaunch drone_ppolstm_nav simulation.launch
2. Start MAVROS:
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
🧠 Training the PPO-LSTM Agent
rosrun drone_ppolstm_nav train_lstm_recurrentppo.py
The drone will arm, take off, switch to OFFBOARD, and navigate using LSTM.
Learning is driven by a reward function encouraging goal-reaching and avoiding obstacles.

📊 Training Results
🟣 Performance Comparison
🟠 Generalization to Unseen Environments
🟢 Evaluation in Dynamic Obstacle Scenario



## 🎥 Demo Videos

▶️  
▶️ [Demo: Generalization in Unseen Environment](media/Testing.MP4)
 https://github.com/Maryamallawi96/ppo-lstm-drone-nav/blob/main/media/Testing%20wih%20path.MOV
👩‍💻 Author
Maryam Allawi
📬 pgs.maryam.allawi@uobasrah.edu.iq
🌐 GitHub: Maryamallawi96
