#!/usr/bin/env python

import rospy
import math
import numpy as np
import time
from fs_msgs.msg import ControlCommand

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
import tf  
from tf.transformations import euler_from_quaternion

class PurePursuit:
    def __init__(self):

        self.is_linear_interpolation = True
        self.is_velocity = True
        self.is_curvature = False

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

        # --- [추가] --- 성능 지표 기록을 위한 변수 초기화
        self.is_4_lap = False
        # 1. 랩타임 관련
        self.lap_start_time = None
        self.lap_count = 0
        self.path_length = 0.0
        self.start_pose = None # 경로의 시작점
        self.lap_completion_threshold = 5.0 # 시작점 근처로 판단할 거리 (m)
        self.min_lap_distance_ratio = 0.8 # 랩으로 인정하기 위한 최소 주행 거리 비율

        # 2. 랩별 데이터 저장을 위한 리스트
        self.lap_speeds = []
        self.lap_ctes = []
        self.lap_steerings = []
        self.lap_computation_times = []
        self.total_distance_traveled = 0.0
        self.last_position = None

        # --- [핵심 기능 추가] --- 속도 프로파일 관련 파라미터 및 변수
        self.velocity_profile = [] # 각 경로 지점별 목표 속도를 저장할 리스트
        self.max_speed_ms = 10 # 최대 주행 속도 (km/h -> m/s)
        self.min_speed_ms = 5 # 최소 주행 속도 (km/h -> m/s)
        self.curvature_tuning_factor = 5.0 # 곡률에 대한 속도 반응 민감도 (클수록 코너에서 더 많이 감속)

        # 3. 실제 주행 경로 기록을 위한 파일 핸들러
       
        try:
            self.path_log_file = open("driven_path.csv", "w")
            self.path_log_file.write("x,y\n") # CSV 헤더
        except IOError:
            rospy.logerr("주행 경로 로그 파일을 열 수 없습니다.")
            self.path_log_file = None

        # 4. 현재 CTE 저장을 위한 변수
        self.current_cte = 0.0
    
    # --- [추가] --- 노드 종료 시 호출될 함수
    def cleanup(self):
        rospy.loginfo("노드를 종료합니다. 로그 파일을 닫습니다.")
        if self.path_log_file:
            self.path_log_file.close()

    def set_target_v(self, msg):
        self.target_v_ms = msg.data / 3.6 # km/h to m/s
        rospy.loginfo("Target speed set to: %.2f m/s", self.target_v_ms)
        # print("Target speed set to: %.2f m/s" % self.target_v_ms) --- IGNORE ---
    def path_callback(self, msg):
        if not self.is_path_loaded or self.is_path_update:
            self.path = msg
            self.is_path_loaded = True
            self.is_path_update = False
            # --- [추가] --- 경로가 로드되면 총 길이와 시작점을 계산
            if len(self.path.poses) > 0:
                self.path_length = self.calculatePathLength(self.path)
                self.velocity_profile = self.generate_velocity_profile(self.path)
                self.start_pose = self.path.poses[0].pose.position
                rospy.loginfo(f"경로가 로드되었습니다. 총 길이: {self.path_length:.2f} m")
    def gss_callback(self,msg):
        self.current_v_ms = msg.twist.twist.linear.x

    def odom_callback(self, msg):
        if not self.is_odom_received:
            # rospy.loginfo("첫 번째 Odometry 메시지를 받았습니다.")
            self.is_odom_received = True
            self.last_position = msg.pose.pose.position # --- [추가] --- 첫 위치 저장
        self.odometry = msg
        
        # --- [추가] --- 주행 거리 누적 및 경로 파일에 기록
        current_pos = msg.pose.pose.position
        if self.last_position:
            dist_increment = math.sqrt((current_pos.x - self.last_position.x)**2 + 
                                     (current_pos.y - self.last_position.y)**2)
            self.total_distance_traveled += dist_increment
        self.last_position = current_pos
        
        if self.lap_count < 1: # 첫 랩이 끝난 후에는 기록 중지
            if self.path_log_file:
                self.path_log_file.write(f"{current_pos.x},{current_pos.y}\n")

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

    def calculate_lookahead_distance(self):
        if self.is_velocity:
            lfd = 0.4 * self.current_v_ms - 1.0
            self.lookahead_distance = max(1.0, min(3.0, lfd))
        else:
            self.lookahead_distance = 2.0
        if self.is_curvature:
            pass # 추후 구현 예정


    def set_target_speed(self, speed):
        self.target_v_ms = speed / 3.6 # km/h to m/s
        
    def pidVelocity(self, target_v_ms):
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
        error = target_v_ms - self.current_v_ms
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
        self.integral = max(-1.0, min(1.0, self.integral)) 
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

        # 5. 최종 제어 출력 계산
        # P, I, D 항에 각각의 게인(Kp, Ki, Kd)을 곱하여 더함
        pid_output = (kp * p_term) + (ki * i_term) + (kd * d_term)
        # rospy.loginfo_throttle(0.1,"PID Output: %.4f (P: %.4f, I: %.4f, D: %.4f)" % (pid_output, (kp * p_term), (ki * i_term), (kd * d_term))) # --- IGNORE ---
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
    def generate_velocity_profile(self, path):
        path_points = path.poses
        if len(path_points) < 3:
            return [self.max_speed_ms] * len(path_points) # 점이 3개 미만이면 최대속도로 설정

        curvatures = []
        # 1. 각 지점의 곡률 계산 (세 점으로 원을 만들어 반지름의 역수를 취함)
        for i in range(len(path_points)):
            if i == 0 or i == len(path_points) - 1:
                curvatures.append(0) # 시작점과 끝점은 곡률 0
                continue
            
            p1 = path_points[i-1].pose.position
            p2 = path_points[i].pose.position
            p3 = path_points[i+1].pose.position

            # 삼각형의 세 변 길이
            a = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            b = math.sqrt((p3.x - p2.x)**2 + (p3.y - p2.y)**2)
            c = math.sqrt((p1.x - p3.x)**2 + (p1.y - p3.y)**2)
            
            # 삼각형의 면적 (헤론의 공식)
            s = (a + b + c) / 2
            area_squared = s * (s - a) * (s - b) * (s - c)
            if area_squared <= 0: # 세 점이 거의 일직선상에 있는 경우
                curvatures.append(0)
                continue
            area = math.sqrt(area_squared)

            # 외접원의 반지름 R = abc / 4*Area, 곡률 K = 1/R
            radius = (a * b * c) / (4 * area)
            curvatures.append(1 / radius)
        
        # 2. 곡률을 목표 속도로 변환
        velocity_profile = []
        for k in curvatures:
            target_v = self.max_speed_ms / (1 + self.curvature_tuning_factor * k)
            # target_v = self.max_speed_ms - (self.max_speed_ms - self.min_speed_ms) * (k/ (max(curvatures)+1e-5) )
            
            # 속도를 min/max 범위 내로 제한
            clamped_v = max(self.min_speed_ms, min(self.max_speed_ms, target_v))
            velocity_profile.append(clamped_v)

        return velocity_profile
    
    def find_lookahead_point(self, LFD):
        
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
        # --- [추가] --- 계산된 최소 거리를 현재 CTE로 저장
        self.current_cte = min_dist
        
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
        target_point = Point()
        
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
                if self.is_linear_interpolation:
                    ratio = (segment_len - over) / segment_len
                    
                    target_point.x = p1.x + ratio * (p2.x - p1.x)
                    target_point.y = p1.y + ratio * (p2.y - p1.y)
                else:
                    target_point.x = p2.x
                    target_point.y = p2.y
                return target_point, next_idx

            current_idx = next_idx
        
        # 경로 끝까지 탐색했지만 LFD보다 짧은 경우, 경로의 마지막 점을 목표로 설정
        last_idx = -1 if search_direction == 1 else 0
        target_point.x = self.path.poses[last_idx].pose.position.x
        target_point.y = self.path.poses[last_idx].pose.position.y
        return target_point, last_idx
    
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
    # --- [추가] --- 랩 종료를 확인하는 함수
    def check_lap_completion(self):
        if not self.start_pose or not self.is_odom_received:
            return False

        # 시작 지점과 현재 차량의 거리 계산
        current_pos = self.odometry.pose.pose.position
        dist_to_start = math.sqrt((current_pos.x - self.start_pose.x)**2 + 
                                  (current_pos.y - self.start_pose.y)**2)
        
        # 랩 시작 타이머가 기록되었고, 시작점 근처에 있으며, 충분한 거리를 주행했다면 랩 완주로 판단
        if self.lap_start_time is not None and \
           dist_to_start < self.lap_completion_threshold and \
           self.total_distance_traveled > self.path_length * self.min_lap_distance_ratio:
            return True
        return False
    
    # --- [추가] --- 랩 결과를 계산하고 출력/저장하는 함수
    def process_lap_results(self):
        self.lap_count += 1
        if self.lap_count == 1:
            self.is_4_lap = True
        lap_time = time.time() - self.lap_start_time

        # 누적된 데이터로 통계 계산
        avg_speed = np.mean(self.lap_speeds) if self.lap_speeds else 0
        avg_cte = np.mean(self.lap_ctes) if self.lap_ctes else 0
        max_cte = np.max(self.lap_ctes) if self.lap_ctes else 0
        steering_std = np.std(self.lap_steerings) if self.lap_steerings else 0
        avg_comp_time = np.mean(self.lap_computation_times) if self.lap_computation_times else 0

        # 결과 출력
        rospy.loginfo("="*40)
        rospy.loginfo(f"LAP {self.lap_count} COMPLETED")
        rospy.loginfo(f"  - Lap Time: {lap_time:.2f} s")
        rospy.loginfo(f"  - Average Speed: {avg_speed:.2f} m/s")
        rospy.loginfo(f"  - Average CTE: {avg_cte:.3f} m")
        rospy.loginfo(f"  - Maximum CTE: {max_cte:.3f} m")
        rospy.loginfo(f"  - Steering Stability (Std Dev): {steering_std:.3f}")
        rospy.loginfo(f"  - Avg Computation Time: {avg_comp_time:.4f} ms")
        rospy.loginfo("="*40)

        # 다음 랩을 위해 리스트와 변수 초기화
        self.lap_speeds.clear()
        self.lap_ctes.clear()
        self.lap_steerings.clear()
        self.lap_computation_times.clear()
        self.total_distance_traveled = 0.0
        self.lap_start_time = time.time() # 다음 랩 시작 시간 기록
    def compute_control_command(self):
        if not self.is_odom_received or not self.is_path_loaded or len(self.path.poses) < 2:
            return None
        # --- [추가] --- 랩 시작 시간 기록 (최초 1회)
        if self.lap_start_time is None and self.is_odom_received:
            self.lap_start_time = time.time()

        
        self.calculate_lookahead_distance()
        vehicle_target_point, target_idx = self.find_lookahead_point(self.lookahead_distance)
        self.publish_marker(vehicle_target_point, "lookahead_point")

        current_target_v = self.target_v_ms
        if self.velocity_profile:
            current_target_v = self.velocity_profile[target_idx]
            # rospy.loginfo("Current Target Speed: %.2f m/s", current_target_v)
        
        # angle to the target point
        steering_angle = self.computeSteeringAngle(vehicle_target_point, self.lookahead_distance)
        steering_angle = max(min(self.max_steering_angle, steering_angle), -self.max_steering_angle)

        cmd_msg = ControlCommand()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.header.frame_id = "FScar"

        if self.is_4_lap:
            cmd_msg.throttle = 0.0
            cmd_msg.brake = 1.0
            cmd_msg.steering = 0.0
            # rospy.loginfo("4랩 완료! 차량 정지.")
        else:
            cmd_msg.throttle, cmd_msg.brake = self.pidVelocity(current_target_v)
            cmd_msg.steering = -steering_angle

        self.cmd_pub.publish(cmd_msg)
        # rospy.loginfo("throttle: %.2f, brake: %.2f, steering: %.3f", cmd_msg.throttle, cmd_msg.brake, cmd_msg.steering)
        # --- [추가] --- 현재 스텝의 데이터를 랩 데이터 리스트에 추가
        if self.lap_start_time is not None:
            self.lap_speeds.append(self.current_v_ms)
            self.lap_ctes.append(self.current_cte)
            self.lap_steerings.append(steering_angle)
        
    def run(self):
        rate = rospy.Rate(50) # 50 Hz

        while not rospy.is_shutdown():
            # --- [추가] --- 연산량 측정을 위한 타이머
            start_comp_time = time.perf_counter()
            self.compute_control_command()

            end_comp_time = time.perf_counter()
            computation_time_ms = (end_comp_time - start_comp_time) * 1000

            if self.lap_start_time is not None:
                self.lap_computation_times.append(computation_time_ms)

            # --- [추가] --- 랩 완주 여부 확인 및 결과 처리
            if not self.is_4_lap:
                if self.check_lap_completion():
                    self.process_lap_results()
            rate.sleep()

if __name__ == '__main__':
    try:
        pp = PurePursuit()
        pp.run()
    except rospy.ROSInterruptException:
        pass