
import time
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import LaserScan, Imu
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from gymnasium import Env, spaces
from tf.transformations import euler_from_quaternion

class DroneEnv(Env):
    def __init__(self):
        super().__init__()
        rospy.init_node('drone_env_node', anonymous=True)
        rospy.set_param('/use_sim_time', True)

        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.cmd_callback)
        rospy.Subscriber('/mavros/imu/data', Imu, self.imu_callback)

        self.cmd_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.pos_raw_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)

        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        self.arming_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        self.max_steps = 10000
        self.goal_position = np.array([30.0, 2.0, 7.0])
        self.boundary_limits = np.array([[-4.0, 40.5], [-10.0, 10.0], [0.1, 10.0]])
        self.reset_vars()

        self.action_space = spaces.Box(low=np.array([-0.5, -0.5, -0.5, -1.0]),
                                       high=np.array([1.0, 0.5, 0.5, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(190,), dtype=np.float32)

    def reset_vars(self):
        self.steps = 0
        self.position = np.zeros(3)
        self.prev_position = np.zeros(3)
        self.prev_distance_to_goal = np.linalg.norm(self.goal_position - self.position)
        self.velocity = np.zeros(3)
        self.lidar_data = np.ones(180) * 6.0
        self.linear_acceleration = np.zeros(3)
        self.yaw = 0.0
        self.last_action = np.zeros(4)
        self.collision_counter = 0

    def lidar_callback(self, msg):
        data = np.nan_to_num(np.array(msg.ranges), nan=6.0, posinf=6.0, neginf=0.1)
        self.lidar_data = np.clip(data[::len(data)//180][:180], 0.1, 6.0)

    def position_callback(self, msg):
        self.position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def cmd_callback(self, msg):
        self.velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def imu_callback(self, msg):
        self.linear_acceleration = np.array([
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        q = msg.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def _get_obs(self):
        return np.concatenate([self.lidar_data, self.position, self.velocity, self.linear_acceleration, [self.yaw]])

    def _within_boundaries(self):
        x, y, z = self.position
        return (self.boundary_limits[0][0] <= x <= self.boundary_limits[0][1] and
                self.boundary_limits[1][0] <= y <= self.boundary_limits[1][1] and
                self.boundary_limits[2][0] <= z <= self.boundary_limits[2][1])

    def wait_for_stable_z(self, target_z=7.0, threshold=0.1, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            if abs(self.position[2] - target_z) < threshold and abs(self.velocity[2]) < 0.2:
                return True
            rospy.sleep(0.1)
        return False

    def wait_for_heartbeat(self, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            if np.linalg.norm(self.velocity) < 100 and not np.isnan(self.position[0]):
                return True
            time.sleep(0.2)
        return False

    def reset(self, *, seed=None, options=None):
        while True:
            self.reset_vars()
            try:
                dummy = TwistStamped()
                for _ in range(50):
                    self.cmd_pub.publish(dummy)
                    rospy.sleep(0.05)

                if not self.wait_for_heartbeat():
                    raise Exception("No heartbeat")

                self.set_mode_srv(custom_mode='OFFBOARD')
                self.arming_srv(True)
                rospy.sleep(0.5)

                x = np.random.uniform(3.0, 5.0)
                y = np.random.uniform(-2.0, 2.0)

                target = PositionTarget()
                target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                target.type_mask = 4088
                target.position.x = x
                target.position.y = y
                target.position.z = 7.0

                for _ in range(100):
                    self.pos_raw_pub.publish(target)
                    rospy.sleep(0.05)

                if self.wait_for_stable_z(7.0) and np.min(self.lidar_data) > 1.5:
                    break
                else:
                    print("[RESET] Unsafe position detected – retrying...")

            except Exception as e:
                print("Reset Exception:", e)
                rospy.sleep(1.0)

        for _ in range(30):
            self.cmd_pub.publish(TwistStamped())
            rospy.sleep(0.1)

        self.prev_position = self.position.copy()
        self.prev_distance_to_goal = np.linalg.norm(self.goal_position - self.position)
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        smoothed = 0.8 * self.last_action + 0.2 * action
        self.last_action = smoothed

        cmd = TwistStamped()
        if self.steps < 5:
            cmd.twist.linear.z = 0.1
        elif self.steps < 10:
            cmd.twist.linear.x = smoothed[0]
            cmd.twist.linear.y = smoothed[1]
            cmd.twist.linear.z = min(0.2, smoothed[2])
            cmd.twist.angular.z = smoothed[3]
        else:
            cmd.twist.linear.x = smoothed[0]
            cmd.twist.linear.y = smoothed[1]
            cmd.twist.linear.z = smoothed[2]
            cmd.twist.angular.z = smoothed[3]

        self.cmd_pub.publish(cmd)
        rospy.sleep(0.1)

        obs = self._get_obs()
        distance = np.linalg.norm(self.goal_position - self.position)
        progress = self.prev_distance_to_goal - distance
        min_lidar = np.min(self.lidar_data)

        reward = progress * 500
        reward += max(0, (25 - distance)) * 20
        reward -= np.linalg.norm(self.linear_acceleration) * 1.5

        terminated = False
        truncated = False
        success = False

        if min_lidar < 1.0:
            reward -= 100
            self.collision_counter += 1
        elif min_lidar < 1.5:
            reward -= 150

        if abs(self.position[0] - self.goal_position[0]) < 0.5:

                reward += 1000
                if self.collision_counter == 0:
                    reward += 1500
                success = True
                terminated = True
                for _ in range(40):
                    self.cmd_pub.publish(TwistStamped())
                    rospy.sleep(0.5)

        if not self._within_boundaries():
            reward -= 200
            terminated = True
            for _ in range(40):
                self.cmd_pub.publish(TwistStamped())
                rospy.sleep(0.5)

        if self.steps >= self.max_steps:
            truncated = True

        self.prev_position = self.position.copy()
        self.prev_distance_to_goal = distance

        return obs, reward, terminated, truncated, self._info(success, distance)

    def _info(self, success, dist):
        return {
            "is_success": success,
            "collisions": self.collision_counter,
            "distance": dist
        }
