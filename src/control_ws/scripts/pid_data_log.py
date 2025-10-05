#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import csv
import os
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from fs_msgs.msg import ControlCommand
from geometry_msgs.msg import PoseStamped, Point, TwistWithCovarianceStamped

class DataCollector:
    """
    PID 튜닝을 위한 입력(스로틀) 및 출력(속도) 데이터를 수집하는 ROS 노드.
    지정된 시간 동안 스텝 입력을 가하고, 시간, 입력값, 속도를 CSV 파일로 저장합니다.
    """
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('pid_data_collector', anonymous=True)

        # --- 파라미터 설정 (사용자 환경에 맞게 수정) ---
        # 1. 발행(Publish)할 스로틀 토픽 이름
        self.throttle_topic_name = rospy.get_param('~throttle_topic', '/fsds/control_command')


        # 3. 데이터가 저장될 CSV 파일 경로
        default_path = os.path.join(os.path.expanduser('~'), 'pid_data.csv')
        self.output_file_path = "/home/a/KSAE_FSD_2025/src/control_ws/data/data_0.8.csv"

        # 4. 실험 파라미터
        self.brake = 0.5
        self.step_throttle_value = rospy.get_param('~step_value', 0.8) # 스텝 입력으로 가할 스로틀 값 (0~1)
        self.step_time = rospy.get_param('~step_time', 5.0)           # 스텝 입력을 시작할 시간 (초)
        self.duration = rospy.get_param('~duration', 10.0)            # 전체 데이터 수집 시간 (초)
        self.rate = rospy.get_param('~rate', 10)                      # 데이터 수집 주기 (Hz)

        # --- 변수 초기화 ---
        self.current_vel_ms = 0.0
        self.start_time = None
        self.data_log = [] # [시간, 입력 스로틀, 출력 속도] 데이터를 저장할 리스트

        # --- ROS Publisher & Subscriber 설정 ---
        # 스로틀 명령을 보낼 퍼블리셔
        self.throttle_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        
        # 속도 정보를 받을 서브스크라이버
        self.speed_sub = rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, self.gss_cb)

        # 노드 종료 시 CSV 파일 저장 함수 호출 설정
        rospy.on_shutdown(self.save_to_csv)

        rospy.loginfo("Data collector node initialized.")
        rospy.loginfo("Throttle Topic: %s", self.throttle_topic_name)
        rospy.loginfo("Saving data to: %s", self.output_file_path)

    def gss_cb(self, msg):
        self.current_vel_ms = msg.twist.twist.linear.x

    def run(self):
        """
        메인 루프: 스텝 입력을 발행하고 데이터를 주기적으로 기록합니다.
        """
        rate = rospy.Rate(self.rate)
        self.start_time = rospy.get_time()

        rospy.loginfo("Starting data collection for %.1f seconds...", self.duration)

        while not rospy.is_shutdown():
            
            current_time = rospy.get_time() - self.start_time
            
            # 전체 수집 시간이 지나면 노드 종료
            if current_time > self.duration:
                rospy.loginfo("Collection duration reached. Shutting down.")
                break

            # 스텝 입력 로직
            cmd_msg = ControlCommand()
            cmd_msg.header.stamp = rospy.Time.now()
            # cmd_msg.throttle = self.step_throttle_value

            
            if current_time >= self.step_time:
                cmd_msg.throttle = self.step_throttle_value
                cmd_msg.brake = 0.0

                # cmd_msg.throttle = 0.0
                # cmd_msg.brake = self.brake
                
            # 스로틀 명령 발행
            self.throttle_pub.publish(cmd_msg)

            # 데이터 기록: [시간, 현재 적용된 스로틀, 현재 측정된 속도]
            self.data_log.append([current_time, cmd_msg.throttle, self.current_vel_ms])

            # if current_time >= 5.0:
            #     self.data_log.append([current_time, cmd_msg.brake, self.current_vel_ms])
            rate.sleep()

    def save_to_csv(self):
        """
        수집된 데이터를 CSV 파일로 저장합니다.
        """
        rospy.loginfo("Saving collected data to %s", self.output_file_path)
        try:
            with open(self.output_file_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                # CSV 헤더 작성
                writer.writerow(['Time', 'Throttle_Input', 'Speed_Output_m/s'])
                # 데이터 작성
                writer.writerows(self.data_log)
            rospy.loginfo("Data successfully saved.")
        except IOError:
            rospy.logerr("Could not write to file: %s", self.output_file_path)

if __name__ == '__main__':
    try:
        collector = DataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated by user.")