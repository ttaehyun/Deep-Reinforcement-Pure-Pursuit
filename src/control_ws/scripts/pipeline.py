#!/usr/bin/env python

import rospy
import math
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from fs_msgs.msg import ControlCommand
from fs_msgs.srv import Reset
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Imu
import tf  
from tf.transformations import euler_from_quaternion

import wandb
# pure pursuit controller
class PurePursuit:
    def __init__(self):

        self.path_data = Path()

        self.wheel_base = 1.55 # meters

        self.lookahead_distance = 5.0 # meters
        self.max_steering_angle = 1 # radians
        self.target_v_ms = 10 / 3.6 # m/s

        self.current_v_ms = 0.0

        # PID 계산을 위한 상태 변수 초기화
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = time.time()
    def publish_marker(self,publisher, frame_id, point, ns, color):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point
        marker.pose.orientation.w = 1.0

        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1

        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]

        publisher.publish(marker)

    def set_lookahead_distance(self, lfd):
        self.lookahead_distance = lfd

    def set_target_speed(self, speed):
        self.target_v_ms = speed # m/s
    
    def set_path_data(self, path):
        self.path_data = path

    def pidVelocity(self, current_v_ms):
        # PID 제어기 파라미터 (튜닝 필요)
        # accel
        # Kp	0.25087
        # Ki	0.0034507
        # Kd	3.5599
        # 1. 시간 변화량 (dt) 계산
        current_time = time.time()
        dt = current_time - self.last_time
        # dt가 0이거나 너무 작으면 계산 오류를 방지하기 위해 이전 값을 반환
        if dt <= 0:
            return 0.0, 0.0

        # 2. 오차(Error) 계산
        error = self.target_v_ms - current_v_ms
        if (error * self.previous_error) < 0:
            self.integral = 0.0
        # 3. P, I, D 각 항 계산
        # 비례(Proportional) 항
        p_term = error

        # 적분(Integral) 항
        self.integral += error * dt
        # --- 적분 와인드업 방지 (Anti-windup) ---
        # 제어 출력이 최대/최소에 도달했을 때 적분값이 무한정 커지는 것을 방지
        # 예를 들어, 스로틀/브레이크가 이미 1.0일 때 적분값이 계속 커지는 것을 막음
        # 아래 clamp 로직과 함께 적분값을 특정 범위로 제한할 수도 있습니다.
        # self.integral = max(-1.0, min(1.0, self.integral)) 
        i_term = self.integral

        # 미분(Derivative) 항
        derivative = (error - self.previous_error) / dt
        d_term = derivative

        # 4. 상황에 맞는 PID 게인 선택 (가속 vs 감속)
        # 오차가 양수(+)이면 목표 속도보다 느리므로 가속 필요
        if error > 0:
            # 목표 속도까지 도달하는데 0.4초 달성
            kp = 0.05
            ki = 0.02
            kd = 0.0
        # 오차가 음수(-)이면 목표 속도보다 빠르므로 감속 필요
        else:
            # 목표 속도까지 도달하는데 0.5초 달성
            kp = 0.1
            ki = 0.02
            kd = 0.0
        # print("error: ", error)
        # 5. 최종 제어 출력 계산
        # P, I, D 항에 각각의 게인(Kp, Ki, Kd)을 곱하여 더함
        pid_output = (kp * p_term) + (ki * i_term) + (kd * d_term)

        # 6. 다음 계산을 위해 현재 상태 저장
        self.previous_error = error
        self.last_time = current_time
        
        # 7. 제어 출력을 스로틀과 브레이크 값으로 변환
        throttle = 0.0
        brake = 0.0
        
        if pid_output > 0: # 출력이 양수이면 스로틀 조작
            throttle = pid_output
        else: # 출력이 음수이면 브레이크 조작
            brake = -pid_output # 브레이크 값은 양수여야 하므로 부호 변경

        # 스로틀과 브레이크 값을 0과 1 사이로 제한 (Clamping)
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        # print("current_v_ms: %.4f, target_v_ms: %.4f" % (current_v_ms, self.target_v_ms))
        # print("PID Output: %.4f, Throttle: %.4f, Brake: %.4f" % (pid_output, throttle, brake))
        return throttle, brake
    def search_direction(self, nearest_idx, yaw):
        vehicle_heading_vector = (math.cos(yaw), math.sin(yaw))

        # 3. 가장 가까운 점 주변의 경로 방향 벡터 계산 및 내적(dot product)을 통한 방향 결정
        #    경로의 끝에 가까울 경우를 대비하여 인덱스 확인
        if nearest_idx + 1 < len(self.path_data.poses):
            p1 = self.path_data.poses[nearest_idx].pose.position
            p2 = self.path_data.poses[nearest_idx + 1].pose.position
            
        else: # 경로의 마지막 점일 경우, 이전 점을 사용하여 방향 계산
            p1 = self.path_data.poses[nearest_idx - 1].pose.position
            p2 = self.path_data.poses[nearest_idx].pose.position

        path_vector = (p2.x - p1.x, p2.y - p1.y)
        # 내적 계산
        dot_product = vehicle_heading_vector[0] * path_vector[0] + vehicle_heading_vector[1] * path_vector[1]

        # 4. 결정된 방향으로 lookahead point 탐색
        search_direction = 1 if dot_product >= 0 else -1
        
        return search_direction
    def find_lookahead_point(self, nearest_idx, search_direction, yaw):
        target_point = Point()

        accumulated_distance = 0.0
        current_idx = nearest_idx
        
        while True:
            next_idx = current_idx + search_direction
            
            # 경로의 시작 또는 끝에 도달하면 탐색 중지
            if next_idx < 0 or next_idx >= len(self.path_data.poses):
                break

            p1 = self.path_data.poses[current_idx].pose.position
            p2 = self.path_data.poses[next_idx].pose.position
            
            segment_len = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            accumulated_distance += segment_len

            if accumulated_distance >= self.lookahead_distance:
                over = accumulated_distance - self.lookahead_distance
                ratio = (segment_len - over) / segment_len
                
                target_point.x = p1.x + ratio * (p2.x - p1.x)
                target_point.y = p1.y + ratio * (p2.y - p1.y)
                return target_point
            current_idx = next_idx
            
        # 경로 끝까지 탐색했지만 LFD보다 짧은 경우, 경로의 마지막 점을 목표로 설정
        last_idx = -1 if search_direction == 1 else 0
        target_point.x = self.path_data.poses[last_idx].pose.position.x
        target_point.y = self.path_data.poses[last_idx].pose.position.y
        return target_point
    
    def computeSteeringAngle(self, target_point, cx, cy, yaw):

        dx = target_point.x - cx
        dy = target_point.y - cy

        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        if (local_x == 0.0):
            return 0.0
        
        curvature = (2.0 * local_y) / (self.lookahead_distance ** 2)
        # rospy.loginfo("Curvature: %.4f Yaw: %.2f", curvature, yaw)

        steering_angle = math.atan(curvature * self.wheel_base)
        steering_angle = max(min(self.max_steering_angle, steering_angle), -self.max_steering_angle)
        return steering_angle

    def calculateTotalPathLength(self):
        length = 0.0

        for i in range(1, len(self.path_data.poses)):
            p1 = self.path_data.poses[i-1].pose.position
            p2 = self.path_data.poses[i].pose.position
            length += math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        
        return length

    # 경로 완주 길이
    def calculateCurrentPathLength(self, nearest_idx, search_direction, cx, cy):
        completed_length = 0.0
        current_idx = nearest_idx
        while True:
            next_idx = current_idx - search_direction
            
            # 경로의 시작 또는 끝에 도달하면 탐색 중지
            if next_idx < 0 or next_idx >= len(self.path_data.poses):
                break

            p1 = self.path_data.poses[current_idx].pose.position
            p2 = self.path_data.poses[next_idx].pose.position
            completed_length += math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            current_idx = next_idx

        if search_direction > 0:
            if nearest_idx + 1 < len(self.path_data.poses):
                p_start = self.path_data.poses[nearest_idx].pose.position
                p_end = self.path_data.poses[nearest_idx + 1].pose.position
            else: # 경로의 마지막 점일 경우, 이전 점을 사용하여 방향 계산
                p_start = self.path_data.poses[nearest_idx].pose.position
                p_end = self.path_data.poses[0].pose.position
        else:
            if nearest_idx - 1  >= 0:
                p_start = self.path_data.poses[nearest_idx].pose.position
                p_end = self.path_data.poses[nearest_idx - 1].pose.position
            else:
                p_start = self.path_data.poses[nearest_idx].pose.position
                p_end = self.path_data.poses[-1].pose.position
            
    
        path_vector = np.array([p_end.x - p_start.x, p_end.y - p_start.y])
        robot_vector = np.array([cx - p_start.x, cy - p_start.y])

        path_norm = np.linalg.norm(path_vector)
        if path_norm < 1e-6:
            length = 0.0
        else:
            length = np.dot(robot_vector, path_vector) / path_norm

        length = np.clip(length, 0, path_norm)
        total = completed_length + length

        return total

class DrappEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action and observation space

        # ---------(observation space)-------------
        # heading error compared to path
        # cross track error
        # Future waypoints (x, y) - 5 points
        # current speed
        # IMU data (angular velocity z)
        # previous action (LFD, speed)
        # ----------------------------------------
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        # ----------(action space)-------------
        # LFD (lookahead distance) -> action[0]
        # target speed             -> action[1]
        # -------------------------------------

        LFD_MIN = 1.0; LFD_MAX = 5.0 # meters
        SPEED_MIN = 1.38; SPEED_MAX = 10.0 # m/s
        self.action_space = spaces.Box(low=np.array([LFD_MIN, SPEED_MIN]).astype(np.float32),
                                       high=np.array([LFD_MAX, SPEED_MAX]).astype(np.float32))

        # --- ROS Node Initialization ---
        rospy.init_node('drapp_env_node')
        # Publisher
        self.cmd_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        self.odom_marker_pub = rospy.Publisher('/fsds/odom_marker', Marker, queue_size=1)
        self.lfd_marker_pub = rospy.Publisher('/fsds/lfd_marker', Marker, queue_size=1)
        # Subscriber
        rospy.Subscriber('/fsds/imu', Imu, self.imu_cb)
        rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, self.gss_cb)
        rospy.Subscriber('/trajectory', Path, self.path_cb)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_cb)
        # Service Client
        rospy.wait_for_service('/fsds/reset')
        self.reset_service_client = rospy.ServiceProxy('/fsds/reset', Reset)
        # --------------------------------

        # Pure Pursuit Controller
        self.ppControl = PurePursuit()
        
        self.curriculum_stage = 3

        self.is_imu_received = False
        self.is_gss_received = False
        self.is_path_received = False

        self.path = Path()
        self.odom = Odometry()
        
        self.curvature_proxy = 0.0

        self.prev_action = np.array([0.0, 0.0])
        self.current_vel_ms = 0.0
        self.angular_velocity_z = 0.0
        self.prev_progress = 0.0

        self.search_direction = 1 # 1: forward, -1: backward
        # 4. 보상 항목별 가중치 설정 (튜닝 필요!)
        self.w_cte = -1.0
        self.w_heading = -1.0
        self.w_vel_error = -0.5
        self.w_vel = 0.5
        self.w_steering = -0.5
        self.w_progress = 18  # 진행 보상에 대한 가중치

        self.w_target = 10.0 # 코너 타겟팅 가중치

        self.lfd_prev = 3.0
        self.v_prev   = 7.0
        self.beta = 0.7   # LFD 스무딩
        self.gamma = 0.4  # 속도 스무딩
        self.lfd_min, self.lfd_max = 1.0, 5.0
        self.v_min,   self.v_max   = 4.0, 10.0

        self.set_reward_weights()
    def imu_cb(self, msg):

        self.angular_velocity_z = msg.angular_velocity.z

    def gss_cb(self, msg):
        self.current_vel_ms = msg.twist.twist.linear.x

    def path_cb(self, msg):
        if not self.is_path_received:
            self.is_path_received = True
            self.path = msg
            self.ppControl.set_path_data(self.path)
    
    def odom_cb(self, msg):
        if rospy.is_shutdown():
            return
        self.odom = msg

        self.ppControl.publish_marker(self.odom_marker_pub, "map", msg.pose.pose.position, "odom", (1, 0, 0))

    def set_reward_weights(self):
        if self.curriculum_stage == 1:
            rospy.loginfo("Curriculum Stage 1: Learning to stay on track")
            self.w_cte = -8.0
            self.w_heading = -0.5
            self.w_progress = 10.0
            self.w_vel_error = 0.0
            self.w_vel = 0.0
            self.w_steering = -4.0
            self.w_target = 0.0
        elif self.curriculum_stage == 2:
            rospy.loginfo("Curriculum Stage 2: Learning speed control.")
            # --- 보상 항목 ---
            self.w_vel = 0.3      
            self.w_progress = 10.0    
            self.w_target = 10.0

            # --- 페널티 항목 ---
            self.w_cte = -1.0         
            self.w_steering = -0.5   
            self.w_heading = -1.0
            
            # --- 비활성화 항목 ---
            self.w_vel_error = -0.0 # 급한 가속 감속 페널티

        else: # Stage 3 or default
            rospy.loginfo("Curriculum Stage 3: Advanced cornering.")
            self.w_cte = -1.0
            self.w_heading = -1.0
            self.w_vel_error = -0.1
            self.w_vel = 0.3
            self.w_steering = -0.5
            self.w_progress = 10.0
            self.w_target = 10.0
    def reset_parameters(self):
        self.prev_action = np.array([0.0, 0.0])
        self.current_vel_ms = 0.0
        self.angular_velocity_z = 0.0
        self.prev_progress = 0.0
        self.is_imu_received = False

    def reset(self, seed=None, options = None):
        try:
            self.reset_service_client(True)

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
        
        # reset parameters
        self.reset_parameters()
        rospy.sleep(1.0) # wait for a moment to get messages

        # return initial observation
        observation = self.get_observation()
        info = {}
        return observation, info

    def get_observation(self):
        # ---------(observation space)-------------
        # heading error compared to path
        # cross track error
        # Future waypoints (x, y) - 10 points
        # current speed
        # IMU data (angular velocity z)
        # previous action (LFD, speed)
        # ----------------------------------------

        cte, heading_error = self.calculate_cte_and_heading_error()
        future_waypoints = self.load_future_waypoints(num_waypoints=10)
        
        # IMU data (angular velocity z)
        angular_vel_z = self.angular_velocity_z

        obs = np.array([
            cte, heading_error,
            self.current_vel_ms, angular_vel_z,
            future_waypoints[0][0], future_waypoints[0][1],
            future_waypoints[1][0], future_waypoints[1][1],
            future_waypoints[2][0], future_waypoints[2][1],
            future_waypoints[3][0], future_waypoints[3][1],
            future_waypoints[4][0], future_waypoints[4][1],
            future_waypoints[5][0], future_waypoints[5][1],
            future_waypoints[6][0], future_waypoints[6][1],
            future_waypoints[7][0], future_waypoints[7][1],
            future_waypoints[8][0], future_waypoints[8][1],
            future_waypoints[9][0], future_waypoints[9][1],
            self.prev_action[0], self.prev_action[1]
        ]).astype(np.float32)

        return obs
    
    def get_euler(self):
        orientation_q = self.odom.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        return roll, pitch, yaw
    
    def calculate_cte_and_heading_error(self):
        """ Cross Track Error와 Heading Error를 계산합니다. """
        nearest_idx, cx, cy = self.find_nearest_segment()
        # print(nearest_idx, cx, cy, len(self.path.poses))
        _, _, cyaw = self.get_euler()

        # 1. 가장 가까운 경로 선분 찾기
        # p1 = self.path.poses[nearest_idx].pose.position
        # if nearest_idx == len(self.path.poses) - 1:
        #     p2 = self.path.poses[0].pose.position
        # else:
        #     p2 = self.path.poses[nearest_idx + 1].pose.position
        if self.search_direction > 0:
            if nearest_idx + 1 < len(self.path.poses):
                p_start = self.path.poses[nearest_idx].pose.position
                p_end = self.path.poses[nearest_idx + 1].pose.position
            else: # 경로의 마지막 점일 경우, 이전 점을 사용하여 방향 계산
                p_start = self.path.poses[nearest_idx].pose.position
                p_end = self.path.poses[0].pose.position
        else:
            if nearest_idx - 1  >= 0:
                p_start = self.path.poses[nearest_idx].pose.position
                p_end = self.path.poses[nearest_idx - 1].pose.position
            else:
                p_start = self.path.poses[nearest_idx].pose.position
                p_end = self.path.poses[-1].pose.position
        path_yaw = 0.0
    
        # 2. 경로 선분의 각도(yaw) 계산
        path_yaw = math.atan2(p_end.y - p_start.y, p_end.x - p_start.x)

        # 3. Cross Track Error (CTE) 계산
        # 경로 선분과 로봇 위치 사이의 벡터를 이용
        path_dx = p_end.x - p_start.x
        path_dy = p_end.y - p_start.y
        robot_dx = cx - p_start.x
        robot_dy = cy - p_start.y
        # print(path_dx, path_dy, robot_dx, robot_dy)
        # 외적(Cross Product)을 이용하여 경로의 좌/우측 어디에 있는지 부호를 결정
        cross_product = path_dx * robot_dy - path_dy * robot_dx
        # print(cross_product)
        cte = math.copysign(np.linalg.norm(np.cross([path_dx, path_dy, 0], [robot_dx, robot_dy, 0])) / np.linalg.norm([path_dx, path_dy, 0]), cross_product)

        # 4. Heading Error 계산
        heading_error = path_yaw - cyaw
        # print("Heading Error (raw): %.2f" % heading_error)
        # -pi ~ +pi 범위로 정규화
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
            
        return cte, heading_error

    def find_nearest_segment(self):
        # Current position
        cx = self.odom.pose.pose.position.x
        cy = self.odom.pose.pose.position.y

        # Find the closest point on the path
        nearest_idx = 0
        min_dist = float('inf')

        for i, pose in enumerate(self.path.poses):
            px = pose.pose.position.x
            py = pose.pose.position.y
            dist = math.sqrt((px - cx)**2 + (py - cy)**2)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx, cx, cy
        
    def load_future_waypoints(self, num_waypoints=5):

        nearest_idx, cx, cy = self.find_nearest_segment()
        global_waypoints = []
        for i in range(nearest_idx, nearest_idx + num_waypoints):
            if i < len(self.path.poses):
                global_waypoints.append(self.path.poses[i].pose.position)
            else:
                # 경로가 부족할 경우 마지막 점을 반복해서 사용
                global_waypoints.append(self.path.poses[-1].pose.position)

        _, _, cyaw = self.get_euler()

        relative_waypoints = []
        for point in global_waypoints:
            dx = point.x - cx
            dy = point.y - cy

            # 회전 변환
            rotated_x = dx * math.cos(-cyaw) - dy * math.sin(-cyaw)
            rotated_y = dx * math.sin(-cyaw) + dy * math.cos(-cyaw)
            relative_waypoints.append([rotated_x, rotated_y])

        return np.array(relative_waypoints)
         
    def step(self, action):
        # action[0] : LFD
        # action[1] : target speed
        if self.curriculum_stage == 1:
            lfd = action[0]
            target_speed = 6.0 # 1단계에서는 속도 고정
        elif self.curriculum_stage == 2:
            # 곡률이 작아질수록 LFD 키우기 (hysteresis 느낌)
            base = max(0.4*self.current_vel_ms - 1.0, 1.0)
            gain = 0.5/(0.1 + self.curvature_proxy)       # 곡률↓ → gain↑
            lfd = np.clip(base + gain, 1.0, 3.0)     # 상한 3.0으로 확대
            target_speed = action[1]
        else:
            lfd_raw = action[0]
            v_raw = action[1] # 2, 3단계에서는 에이전트가 결정
            # 안전 레이어(클램프 + 스무딩)
            lfd = np.clip(self.beta  * self.lfd_prev + (1-self.beta)  * lfd_raw, self.lfd_min, self.lfd_max)
            target_speed = np.clip(self.gamma * self.v_prev   + (1-self.gamma) * v_raw,   self.v_min,   self.v_max)

            self.lfd_prev = lfd
            self.v_prev   = target_speed
        rospy.loginfo_throttle(0.05, "LFD: %.2f, Target Speed: %.2f" % (lfd, target_speed))
        self.ppControl.set_lookahead_distance(lfd)
        self.ppControl.set_target_speed(target_speed)
        self.ppControl.set_path_data(self.path)

        nearest_idx, cx, cy = self.find_nearest_segment()
        _, _, yaw = self.get_euler()
        # ppControl의 기능을 활용하여 조향각 계산
        self.search_direction = self.ppControl.search_direction(nearest_idx, yaw)
        target_point = self.ppControl.find_lookahead_point(nearest_idx=nearest_idx, search_direction=self.search_direction, yaw=yaw)
        self.ppControl.publish_marker(self.lfd_marker_pub, "map", target_point, "lfd", (0, 1, 0))

        steering_angle = self.ppControl.computeSteeringAngle(target_point=target_point, cx=cx, cy= cy, yaw= yaw)
        throttle, brake = self.ppControl.pidVelocity(self.current_vel_ms)

        # 제어 명령 발행
        cmd_msg = ControlCommand()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.header.frame_id = "FScar"
        cmd_msg.throttle = throttle
        cmd_msg.brake = brake
        cmd_msg.steering = -steering_angle
        self.cmd_pub.publish(cmd_msg)

        # 다음 상태 관측 및 보상 계산
        rospy.sleep(0.01) # 시뮬레이션이 진행될 시간을 줌
        observation = self.get_observation()

        nearest_idx, cx, cy = self.find_nearest_segment()
        current_progress = self.ppControl.calculateCurrentPathLength(nearest_idx,self.search_direction, cx, cy)

        step_progress = current_progress - self.prev_progress
        step_progress = max(0.0, step_progress) # progress가 감소하는 것을 방지
        self.prev_progress = current_progress

        # rospy.loginfo_throttle(0.1, "Step Progress: %.2f, Current Progress: %.2f" % (step_progress, current_progress))
       
        # 보상 계산
        cte = observation[0]
        heading_error = observation[1]
        linear_vel_ms = observation[2] 
        angular_vel_z = observation[3]

        future_waypoints_y = observation[5:24:2]  # y 좌표들만 추출 (상대 좌표계이므로 y가 측면 편차)

        # 모든 y좌표의 절대값 평균을 내어 곡률 지표로 사용합니다.
        # 경로가 직선에 가까우면 y값들이 0에 가까워지고, 휠수록 y값들이 커집니다.
        self.curvature_proxy = np.mean(np.abs(future_waypoints_y))
        
        rospy.loginfo_throttle(0.05,"Curvature Proxy (New): %.4f" % (self.curvature_proxy))


        # 하이퍼파라미터 설정
        TRANSITION_CURVATURE = 0.4 # ★★★ 새로운 곡률 지표에 맞게 기준점 상향 조정 (튜닝 필요)
        # TRANSITION_CURVATURE = 0.070 # 튜닝 대상
        TARGET_CORNER_SPEED = 7.0  # m/s

        # 1. 코너링 가중치 계산
        cornering_weight = min(1.0, self.curvature_proxy / TRANSITION_CURVATURE)

        # 2. 각 상황에 대한 보상 값 계산
        # 2-1. 직선 보상 (w_vel, 예: 1.0)
        reward_straight = self.w_vel * linear_vel_ms

        # 곡률 기반 목표속도 v*(kappa)
        k0, k1 = 9.5, 4.0          # 직선/코너 기준 속도(튜닝)
        curvature = self.curvature_proxy
        v_target = np.clip(k0 - k1*curvature, 4.0, 10.0)

        # 속도 추종 보상 (정사각 벌점)
        reward_corner_target = -0.2 * (linear_vel_ms - v_target)**2 + 0.0
        # cornering_weight로 블렌딩 유지

        # 3. 가중치를 이용해 두 보상을 부드럽게 결합
        blended_speed_reward = ((1 - cornering_weight) * reward_straight) + (cornering_weight * reward_corner_target)
        rospy.loginfo_throttle(0.05, "Cornering Weight: %.4f, Reward Straight: %.2f, Reward Corner Target: %.2f, Blended: %.2f" % (cornering_weight, reward_straight, reward_corner_target, blended_speed_reward))
        
        # 액션 변화율 벌점(너무 덜컥거리면)
        reward_error = 0.2 * ( (lfd - self.lfd_prev)**2 + (target_speed - self.v_prev)**2 )
        # 코너에서 LFD 과대 억제(언더스티어 방지)
        reward_lfd = 0.1 * (lfd * self.curvature_proxy)**2

        # 보상 함수 설계
        reward_cte = self.w_cte * (cte ** 2)  # CTE에 대한 페널티
        reward_heading = self.w_heading * (heading_error ** 2)  # Heading
        reward_progress = self.w_progress * step_progress  # 진행에 대한 보상

        vel_error = abs(target_speed - linear_vel_ms)
        reward_vel = self.w_vel_error * (vel_error**2)  # 속도 오차에 대한 페널티

        reward_steering = self.w_steering * (angular_vel_z**2) # linear_vel_ms   급격한 조향에 대한 페널티
        

        reward = reward_cte + reward_heading + reward_error + reward_steering + reward_progress + blended_speed_reward + reward_lfd
        wandb.log({
            'custom/reward_cte': reward_cte,
            'custom/reward_heading': reward_heading,
            'custom/reward_error': reward_error,
            'custom/reward_steering': reward_steering,
            'custom/reward_progress': reward_progress,
            'custom/reward_corner_target': reward_corner_target,
            'custom/reward_straight_speed': reward_straight,
            'custom/blended_speed_reward': blended_speed_reward,
            'custom/reward_lfd': reward_lfd,
            'custom/total_reward': reward,
            'raw/linear_velocity': linear_vel_ms,
            'raw/cte': cte,
            'raw/heading_error': heading_error,
            'raw/self.curvature_proxy': self.curvature_proxy,
            'action/lfd': lfd,
            'action/target_speed': target_speed,
        })
        # ★★★★★ 각 보상 항목을 로그로 출력하여 스케일 확인 ★★★★★
        rospy.loginfo_throttle(0.05, 
            f"Rewards -> CTE: {reward_cte:.2f}, Heading: {reward_heading:.2f}, Vel: {reward_vel:.2f}, "
            f"Steering: {reward_steering:.2f}, Progress: {reward_progress:.2f},  blended_speed_reward: {blended_speed_reward:.2f},  Total: {reward:.2f}"
        )
        # 종료 조건: CTE가 너무 크면 종료
        terminated = False
        if abs(cte) > 1.4: # CTE가 1.4m 이상이면 종료 , path 경로를 완전 따라가도록 유도
            terminated = True
            reward = -200.0  # 큰 페널티
            rospy.loginfo("Episode terminated due to large CTE: %.2f", cte)
        self.prev_action = action

        return observation, reward, terminated, False, {}
