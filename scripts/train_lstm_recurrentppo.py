
from datetime import datetime
import os
import shutil
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from drone_lstm import DroneEnv

# === File Paths ===
now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
policy_path = "recurrent_ppo_drone_model.zip"
backup_path = f"backup_model_{now_str}.zip"
tensorboard_log_dir = "recurrent_ppo_tensorboard"
log_metrics_path = "recurrent_ppo_training_metrics.txt"
path_log_path = os.path.join(tensorboard_log_dir, "path_log.csv")
summary_path = f"recurrent_ppo_summary_{now_str}.txt"

# === Prepare Logs ===
os.makedirs(tensorboard_log_dir, exist_ok=True)
with open(path_log_path, "w") as f:
    f.write("step,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,yaw\n")
with open(log_metrics_path, "w") as f:
    f.write("step,total_reward,distance,collisions,success,episodes\n")

# === Environment ===
raw_env = DroneEnv()
vec_env = make_vec_env(lambda: raw_env, n_envs=1)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

# === Backup Existing Model ===
if os.path.exists(policy_path):
    shutil.copy(policy_path, backup_path)
    print(f"🯭 Backup model saved to '{backup_path}'")

# === PPO Policy Settings ===
policy_kwargs = dict(
    lstm_hidden_size=128,
    shared_lstm=True,
    enable_critic_lstm=False,
    net_arch=dict(pi=[64], vf=[64]),
    activation_fn=torch.nn.Tanh
)

# === Load or Create PPO Model ===
if os.path.exists(policy_path):
    print(f"📦 Found existing model: '{policy_path}' → Loading...")
    model = RecurrentPPO.load(policy_path, env=vec_env, device="cuda")
    print("✅ Loaded model. Continuing training.")
else:
    print("🚀 Starting new training.")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        device="cuda",
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        ent_coef=0.05,
        clip_range=0.2,
        vf_coef=0.1,
        max_grad_norm=0.3,
        n_steps=512,
        normalize_advantage=True,
        use_sde=False,
        policy_kwargs=policy_kwargs,
     
    )

class MetricsLoggingCallback(BaseCallback):
    def __init__(self, log_file_path, verbose=0):
        super().__init__(verbose)
        self.log_file_path = log_file_path
        self.episode_rewards = []
        self.episode_distances = []
        self.episodes = 0
        self.successes = 0
        self.clean_successes = 0
        self.total_collisions = 0
        self.auc = 0

    def is_mostly_decreasing(self, distances):
        count = sum(1 for i in range(1, len(distances)) if distances[i] < distances[i-1])
        return count / len(distances) > 0.7

    def _on_step(self) -> bool:
        try:
            rewards = self.locals.get("rewards", None)
            dones = self.locals.get("dones", None)
            infos = self.locals.get("infos", [{}])
            info = infos[0]

            if rewards:
                self.episode_rewards.append(rewards[0])
                self.episode_distances.append(info.get("distance", 0))
                self.auc += rewards[0]

            if dones and dones[0]:
                env_unwrapped = self.training_env.envs[0].unwrapped
                distance_to_goal = np.linalg.norm(env_unwrapped.goal_position - env_unwrapped.position)
                success = int(info.get("is_success", False))
                collisions = int(info.get("collisions", 0))
                clean_success = int(success and collisions == 0)
                total_reward = sum(self.episode_rewards)

                self.episodes += 1
                self.successes += success
                self.clean_successes += clean_success
                self.total_collisions += collisions

                with open(self.log_file_path, "a") as f:
                    f.write(f"{self.num_timesteps},{total_reward},{distance_to_goal},{collisions},{success},{self.episodes}\n")

                if success and len(self.episode_distances) > 5 and self.episode_distances[0] > 20:
                    if self.is_mostly_decreasing(self.episode_distances):
                        for d in self.episode_distances:
                            self.logger.record("Distance_to_goal vs Episode", d)

                    px, py, pz = env_unwrapped.position
                    vx, vy, vz = env_unwrapped.velocity
                    yaw = env_unwrapped.yaw
                    with open(path_log_path, "a") as f:
                        f.write(f"{self.num_timesteps},{px},{py},{pz},{vx},{vy},{vz},{yaw}\n")

                self.logger.record("Total Reward vs Episode", total_reward)
                self.logger.record("train/collisions", collisions)
                self.logger.record("success", success)
                self.logger.record("clean_success", clean_success)
                self.logger.record("train/episode", self.episodes)
                self.logger.record("Success Rate vs Episode", self.successes / self.episodes if self.episodes else 0)
                self.logger.record("train/clean_success_rate", self.clean_successes / self.episodes if self.episodes else 0)
                self.logger.dump(self.num_timesteps)

                self.episode_rewards = []
                self.episode_distances = []

        except Exception as e:
            print(f"⚠ Logging error: {e}")
        return True

    def _on_rollout_end(self):
        try:
            if hasattr(self.model, "logger"):
                if hasattr(self.model, "value_loss"):
                    self.logger.record("train/value_loss", self.model.value_loss)
        except Exception as e:
            print(f"⚠ Value loss logging error: {e}")

    def write_summary(self):
        try:
            with open(summary_path, "w") as f:
                f.write("| Metric              | Value            | Description                         |\n")
                f.write("|---------------------|------------------|-------------------------------------|\n")
                f.write(f"| AUC                 | {int(self.auc)}        | Total cumulative reward            |\n")
                f.write(f"| Success Rate        | {self.successes}/{self.episodes}     | Success episodes                  |\n")
                f.write(f"| Clean Successes     | {self.clean_successes}/{self.episodes}  | Without collisions                |\n")
                f.write(f"| Avg Reward/Episode  | {self.auc / self.episodes:.1f}         | Mean reward per episode           |\n")
                f.write(f"| Avg Collisions      | {self.total_collisions / self.episodes:.1f}    | Mean collisions                   |\n")
            print(f"📄 Summary saved to {summary_path}")
        except Exception as e:
            print(f"⚠ Summary write failed: {e}")

# === Train ===
metrics_callback = MetricsLoggingCallback(log_metrics_path)
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./ppo_checkpoints",
    name_prefix="recurrent_ppo_checkpoint"
)

model.learn(
    total_timesteps=50000,
    callback=[metrics_callback, checkpoint_callback],
    tb_log_name="FinalRun",
    reset_num_timesteps=True,
    progress_bar=True
)

model.save(policy_path)
vec_env.save("vecnorm_lstm.pkl")
metrics_callback.write_summary()
