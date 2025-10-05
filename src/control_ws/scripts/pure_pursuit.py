#!/usr/bin/env python

import rospy
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from fs_msgs.msg import ControlCommand
from fs_msgs.srv import Reset
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
import tf  
from tf.transformations import euler_from_quaternion

class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit_node')

        # Publisher
        self.cmd_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        self.marker_pub = rospy.Publisher('/lookahead_point_marker', Marker, queue_size=1)
        self.odom_marker_pub = rospy.Publisher('/odom_marker', Marker, queue_size=1)

        # Subscriber
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback )
        rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, self.gss_callback)
        rospy.Subscriber('/trajectory', Path, self.path_callback)
        rospy.Subscriber('/fsds/target_v_ms', Float32, self.set_target_v)
        self.is_odom_received = False
        self.is_path_loaded = False
        self.is_path_update = True
        # self.file_path = '/home/a/KSAE_FSD_2025/src/control_ws/path/CompetitionTestMap2.txt'
        # self.path = self.load_path(self.file_path)
       
        self.odometry = Odometry()
        self.path = Path()

        self.wheel_base = 1.55 # meters

        self.lookahead_distance = 2.0 # meters
        self.max_steering_angle = 1 # radians
        self.target_v_ms = 15 / 3.6 # m/s

        self.current_v_ms = 0.0

        # PID 계산을 위한 상태 변수 초기화
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = time.time()
    
    def set_target_v(self, msg):
        self.target_v_ms = msg.data / 3.6 # km/h to m/s
        rospy.loginfo("Target speed set to: %.2f m/s", self.target_v_ms)
        # print("Target speed set to: %.2f m/s" % self.target_v_ms) --- IGNORE ---
    def path_callback(self, msg):
        if not self.is_path_loaded or self.is_path_update:
            self.path = msg
            self.is_path_loaded = True
            self.is_path_update = False

    def gss_callback(self,msg):
        self.current_v_ms = msg.twist.twist.linear.x

    def odom_callback(self, msg):
        if not self.is_odom_received:
            # rospy.loginfo("첫 번째 Odometry 메시지를 받았습니다.")
            self.is_odom_received = True
        
        self.odometry = msg
        
        point = Point()
        point.x = msg.pose.pose.position.x
        point.y = msg.pose.pose.position.y
        point.z = msg.pose.pose.position.z

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "Odom Point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2
        marker.scale.y = 2
        marker.scale.z = 2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        
        self.odom_marker_pub.publish(marker)

    def load_path(self, path_file):
        path = Path()
        path.header.frame_id = "map"

        with open(path_file, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                # 주석(#)이거나 빈 줄은 건너뛰기
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                
                # 공백을 기준으로 데이터 분리 (x, y, z)
                data = line.split(',')
                
                # PoseStamped 메시지 생성
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "map" # 각 pose의 frame_id도 설정
                
                # 파싱한 데이터로 위치 설정
                pose.pose.position.x = float(data[0])
                pose.pose.position.y = float(data[1])
                # z 좌표가 파일에 있다면 아래 주석 해제
                # pose.pose.position.z = float(data[2]) 
                
                # 방향(orientation)은 기본값(0,0,0,1)으로 설정
                pose.pose.orientation.w = 1.0
                
                # 생성된 PoseStamped 메시지를 Path의 poses 배열에 추가
                path.poses.append(pose)

        self.is_path_loaded = True
        return path

    def publish_marker(self, point, ns):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2
        marker.scale.y = 2
        marker.scale.z = 2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)

    def set_lookahead_distance(self, LFD):
        self.lookahead_distance = LFD

    def set_target_speed(self, speed):
        self.target_v_ms = speed / 3.6 # km/h to m/s
        
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
            kp = 0.1
            ki = 0.08
            kd = 0.01
        # 오차가 음수(-)이면 목표 속도보다 빠르므로 감속 필요
        else:
            # 목표 속도까지 도달하는데 0.5초 달성
            kp = 0.2
            ki = 0.1
            kd = 0.01

        # 5. 최종 제어 출력 계산
        # P, I, D 항에 각각의 게인(Kp, Ki, Kd)을 곱하여 더함
        pid_output = (kp * p_term) + (ki * i_term) + (kd * d_term)
        rospy.loginfo_throttle(0.1,"PID Output: %.4f (P: %.4f, I: %.4f, D: %.4f)" % (pid_output, (kp * p_term), (ki * i_term), (kd * d_term))) # --- IGNORE ---
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
            # throttle = 0.0

        # 스로틀과 브레이크 값을 0과 1 사이로 제한 (Clamping)
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        return throttle, brake

    def find_lookahead_point(self, target_point, LFD):
        if not self.is_odom_received or not self.is_path_loaded or len(self.path.poses) < 2:
            return False
        
        # 현재 차량 위치
        cx = self.odometry.pose.pose.position.x
        cy = self.odometry.pose.pose.position.y

        # 1. 경로 상에서 가장 가까운 점의 인덱스 찾기
        min_dist = float('inf')
        nearest_idx = 0
        for i, pose in enumerate(self.path.poses):
            px = pose.pose.position.x
            py = pose.pose.position.y
            dist = math.sqrt((px - cx)**2 + (py - cy)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # 2. 차량의 현재 진행 방향(yaw)을 기반으로 헤딩 벡터 계산
        orientation_q = self.odometry.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        vehicle_heading_vector = (math.cos(yaw), math.sin(yaw))

        # 3. 가장 가까운 점 주변의 경로 방향 벡터 계산 및 내적(dot product)을 통한 방향 결정
        #    경로의 끝에 가까울 경우를 대비하여 인덱스 확인
        if nearest_idx + 1 < len(self.path.poses):
            p1 = self.path.poses[nearest_idx].pose.position
            p2 = self.path.poses[nearest_idx + 1].pose.position
            path_vector = (p2.x - p1.x, p2.y - p1.y)
        else: # 경로의 마지막 점일 경우, 이전 점을 사용하여 방향 계산
            p1 = self.path.poses[nearest_idx - 1].pose.position
            p2 = self.path.poses[nearest_idx].pose.position
            path_vector = (p2.x - p1.x, p2.y - p1.y)

        # 내적 계산
        dot_product = vehicle_heading_vector[0] * path_vector[0] + vehicle_heading_vector[1] * path_vector[1]

        # 4. 결정된 방향으로 lookahead point 탐색
        search_direction = 1 if dot_product >= 0 else -1
        
        accumulated_distance = 0.0
        current_idx = nearest_idx
        
        while True:
            next_idx = current_idx + search_direction
            
            # 경로의 시작 또는 끝에 도달하면 탐색 중지
            if next_idx < 0 or next_idx >= len(self.path.poses):
                break

            p1 = self.path.poses[current_idx].pose.position
            p2 = self.path.poses[next_idx].pose.position
            
            segment_len = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            accumulated_distance += segment_len

            if accumulated_distance >= LFD:
                over = accumulated_distance - LFD
                ratio = (segment_len - over) / segment_len
                
                target_point.x = p1.x + ratio * (p2.x - p1.x)
                target_point.y = p1.y + ratio * (p2.y - p1.y)
                return True

            current_idx = next_idx
            
        # 경로 끝까지 탐색했지만 LFD보다 짧은 경우, 경로의 마지막 점을 목표로 설정
        last_idx = -1 if search_direction == 1 else 0
        target_point.x = self.path.poses[last_idx].pose.position.x
        target_point.y = self.path.poses[last_idx].pose.position.y
        return True
    
    def computeSteeringAngle(self, target_point, LFD):

        dx = target_point.x - self.odometry.pose.pose.position.x
        dy = target_point.y - self.odometry.pose.pose.position.y
    
        # 쿼터니언 orientation을 리스트/튜플 형태로 변환
        orientation_q = self.odometry.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        # 쿼터니언을 오일러 각으로 변환
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        if (local_x == 0.0):
            return 0.0
        
        curvature = (2.0 * local_y) / (LFD ** 2)
        # rospy.loginfo("Curvature: %.4f Yaw: %.2f", curvature, yaw)

        return math.atan(curvature* self.wheel_base)

    def calculatePathLength(self, path):
        length = 0.0

        for i in range(1, len(path.poses)):
            p1 = path.poses[i-1].pose.position
            p2 = path.poses[i].pose.position
            length += math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        
        return length

    def compute_control_command(self, target_v_ms, LFD):
        if not self.is_odom_received or not self.is_path_loaded:
            # rospy.logwarn("Odometry 또는 Path 정보를 아직 받지 못했습니다.")
            return None
        
        if (len(self.path.poses) < 2):
            # rospy.logwarn("경로에 충분한 포인트가 없습니다.")
            return None
        
        vehicle_target_point = Point()

        if self.find_lookahead_point(vehicle_target_point, LFD):
            self.publish_marker(vehicle_target_point, "lookahead_point")

        # angle to the target point
        steering_angle = self.computeSteeringAngle(vehicle_target_point, LFD)
        steering_angle = max(min(self.max_steering_angle, steering_angle), -self.max_steering_angle)

        cmd_msg = ControlCommand()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.header.frame_id = "FScar"

        cmd_msg.throttle, cmd_msg.brake = self.pidVelocity(self.current_v_ms)
        cmd_msg.steering = -steering_angle

        self.cmd_pub.publish(cmd_msg)
        # rospy.loginfo("throttle: %.2f, brake: %.2f, steering: %.3f", cmd_msg.throttle, cmd_msg.brake, cmd_msg.steering)

        
    def run(self):
        rate = rospy.Rate(50) # 50 Hz

        while not rospy.is_shutdown():
            self.compute_control_command(self.target_v_ms, self.lookahead_distance)
            rate.sleep()

if __name__ == '__main__':
    try:
        pp = PurePursuit()
        pp.run()
    except rospy.ROSInterruptException:
        pass