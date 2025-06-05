# PPO-LSTM Drone Navigation

This project implements memory-augmented reinforcement learning for UAV navigation using PPO + LSTM in ROS/Gazebo.

# 🛸 PPO-LSTM Drone Navigation

This project implements **autonomous drone navigation** in a 3D indoor environment using **Deep Reinforcement Learning (PPO + LSTM)** in ROS/Gazebo. The drone navigates to a target using only LiDAR data and memory (LSTM) for decision-making in partially observable environments.

---

## 🧠 Features

- ✅ PPO + LSTM agent using Stable-Baselines3
- ✅ Navigation using only 2D LiDAR input
- ✅ Memory-augmented for partially observable environments
- ✅ Fully integrated with Gazebo, PX4 SITL, and MAVROS
- ✅ Collision reset, OFFBOARD mode, and reward shaping

---

## 🛠️ Setup

- ROS Noetic
- Gazebo 11
- PX4 Autopilot (SITL)
- `mavros`, `mavros_extras`
- Python packages:  
  ```bash
  pip install stable-baselines3[extra] torch gymnasium

🚀 Run & Train
Launch simulator:

roslaunch drone_ppolstm_nav simulation.launch
Start MAVROS:

roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
Train PPO-LSTM agent:

rosrun drone_ppolstm_nav train_lstm_recurrentppo.py
🎯 Navigation Goal
Start point: (0, -2, 0)

Target goal: (33, -2, 0)

📊 Media & Results
PPO vs PPO + LSTM

Dynamic Obstacle Evaluation

Testing Videos
▶️ Testing Demo
▶️ Unseen Environment

👩‍💻 Author
Maryam Allawi
📬 pgs.maryam.allawi@uobasrah.edu.iq
🌐 GitHub: Maryamallawi96
