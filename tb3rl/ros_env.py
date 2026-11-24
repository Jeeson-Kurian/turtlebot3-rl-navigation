"""ROS Noetic + Gazebo wrapper with the same interface as sim.NavEnv.

Requires a ROS Noetic workspace with turtlebot3_gazebo, e.g.
    export TURTLEBOT3_MODEL=burger
    roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch
NOTE: this wrapper has not been run in this repository's CI; the 2D simulator results in
results/ come from sim.py. Treat Gazebo numbers as a separate experiment.
"""
import numpy as np

from .sim import COLLISION_DIST, DT, GOAL_DIST, MAX_ANG, MAX_LIN, MAX_RANGE, N_BEAMS

try:
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Empty
except ImportError:  # allows importing the module without ROS installed
    rospy = None


class GazeboNavEnv:
    obs_dim = N_BEAMS + 2

    def __init__(self, goals, max_steps=300):
        if rospy is None:
            raise RuntimeError("ROS Noetic (rospy) is required for GazeboNavEnv")
        rospy.init_node("tb3_rl_agent", anonymous=True)
        self.cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        rospy.Subscriber("/odom", Odometry, self._odom)
        self.goals, self.max_steps, self.k = list(goals), max_steps, 0
        self.pos, self.yaw = np.zeros(2), 0.0

    def _odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pos = np.array([p.x, p.y])
        self.yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))

    def _scan(self):
        msg = rospy.wait_for_message("/scan", LaserScan, timeout=5)
        r = np.array(msg.ranges, dtype=float)
        r[~np.isfinite(r) | (r == 0)] = MAX_RANGE
        idx = (np.linspace(-np.pi, np.pi, N_BEAMS, endpoint=False) % (2 * np.pi)) / (2 * np.pi) * len(r)
        return np.clip(r[idx.astype(int) % len(r)], 0, MAX_RANGE)

    def _obs(self):
        self.scan = self._scan()
        rel = self.goal - self.pos
        dist = np.hypot(*rel)
        ang = (np.arctan2(rel[1], rel[0]) - self.yaw + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([self.scan / MAX_RANGE, [min(dist / 4.0, 1.0), ang / np.pi]])

    def reset(self):
        self.cmd.publish(Twist())
        self.reset_world()
        self.goal = np.array(self.goals[self.k % len(self.goals)]); self.k += 1
        self.steps = 0
        rospy.sleep(0.5)
        self.prev_dist = np.hypot(*(self.goal - self.pos))
        return self._obs()

    def step(self, ang_vel, lin_vel=0.15):
        t = Twist()
        t.linear.x = float(np.clip(lin_vel, 0, MAX_LIN))
        t.angular.z = float(np.clip(ang_vel, -MAX_ANG, MAX_ANG))
        self.cmd.publish(t)
        rospy.sleep(DT)
        self.steps += 1
        obs = self._obs()
        dist = np.hypot(*(self.goal - self.pos))
        reward = 10.0 * (self.prev_dist - dist) - 0.01
        self.prev_dist = dist
        info = {"collision": False, "goal": False}
        done = False
        if self.scan.min() < COLLISION_DIST:
            reward, done, info["collision"] = -100.0, True, True
        elif dist < GOAL_DIST:
            reward, done, info["goal"] = 100.0, True, True
        elif self.steps >= self.max_steps:
            done = True
        if done:
            self.cmd.publish(Twist())
        return obs, reward, done, info
