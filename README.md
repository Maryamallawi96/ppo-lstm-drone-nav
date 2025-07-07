
PPO-GRU + ΔLiDAR UAV Navigation 🚁
Lightweight Temporal-Aware Reinforcement Learning for UAVs in Dynamic, Partially Observable Environments

🌟 Overview
This repository presents an enhanced UAV navigation system powered by Proximal Policy Optimization (PPO) combined with a Gated Recurrent Unit (GRU) and a novel temporal input called Delta LiDAR (ΔLiDAR). The model is designed to handle complex, partially observable 3D environments using lightweight memory-aware policies and raw 2D LiDAR input.

✅ Lower collisions – ✅ Faster convergence – ✅ Smooth trajectories – ✅ Better generalization

📦 Package Name
ppogru_deltalider

🧠 Key Features
✅ DRL Agent: PPO + GRU with ΔLiDAR temporal encoding

✅ Input: 360° 2D LiDAR + velocity + attitude vectors

✅ Efficient policy for memory-limited drones

✅ Compatible with ROS Noetic, PX4 SITL, Gazebo 11

✅ Collision resets, offboard control, reward shaping

✅ Generalization tested in foggy and cluttered tunnels

🧭 Simulation Setup
Component	Description
Simulator	Gazebo 11
DRL Algorithm	PPO + GRU (with ΔLiDAR)
Flight Controller	PX4 SITL
Middleware	MAVROS (ROS Noetic)
Sensor Input	360-beam LiDAR + velocity + attitude
Target	(25, -2, 0)

🧪 Delta-LiDAR Explained
ΔLiDAR computes the temporal difference between two LiDAR frames, providing the GRU with motion cues such as:

Obstacle velocity

Heading direction

Environmental transitions

This temporal differencing eliminates the need for LSTM-style memory while maintaining robust performance.

🗂️ Project Structure
bash
Copy
Edit
ppogru_deltalider/
├── launch/               # Launch files (PX4 + Gazebo)
├── scripts/              # PPO-GRU training & environment
├── models/               # Drone with LiDAR
├── worlds/               # Gazebo 3D environments
├── Media/                # Demo videos & screenshots
├── CMakeLists.txt
└── package.xml
⚙️ Dependencies
Ensure the following are installed:

PX4-Autopilot

ROS Noetic

Gazebo 11

MAVROS, MAVROS Extras

stable-baselines3, rospy, gymnasium, torch, sensor_msgs, geometry_msgs

Install Python dependencies:


pip install stable-baselines3[extra] torch tensorboard
🚀 Running the Simulation
Launch Simulator

roslaunch ppogru_deltalider simulation.launch
Start MAVROS


roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
Train the Agent

rosrun ppogru_deltalider train_ppogru_deltalidar.py
📊 Performance Summary
Metric	PPO-GRU	PPO-GRU + ΔLiDAR
Success Rate (%)	82.5	91.0
Avg Collisions/Episode	35.02	20.07
Avg Trajectory Length	17.24 m	12.97 m
Smoothness Score	15.78	8.58
Training Time	1h 40m	1h 45m
Model Parameters	589k	580k

🎥 Media & Demos
📸 Screenshot – Generalization in Unseen Foggy Tunnel
![Unseen Env](Media/Unseen\ env.jpg)

🎬 Demo: PPO-GRU + ΔLiDAR in Action
Media/Generazeion env.MOV

👩‍💻 Author
Maryam Allawi
📧 pgs.maryam.allawi@uobasrah.edu.iq
🌐 GitHub: Maryamallawi96

