#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
import math

class GpsToTxt:
    def __init__(self):
        rospy.init_node('gps_to_txt_node', anonymous=True)

        # 마지막으로 저장된 위도, 경도. 처음엔 None으로 초기화
        self.last_x = None
        self.last_y = None

        # 점을 찍을 최소 거리 간격 (미터 단위)
        self.distance_threshold = 0.2 

        # 결과를 저장할 파일 경로
        # 사용자의 홈 디렉토리에 gps_coordinates.txt 이름으로 저장됩니다.
        self.file_path = '/home/a/KSAE_FSD_2025/src/control_ws/path/temp.txt' # 'YOUR_USERNAME'을 실제 사용자 이름으로 변경하세요.
        self.file = open(self.file_path, 'w')

        # GPS 데이터를 받는 토픽 이름. '/fix'가 일반적이지만, 다를 경우 수정해야 합니다.
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback)

        rospy.on_shutdown(self.shutdown)
        rospy.spin()

    def odom_callback(self, data):
        current_x = data.pose.pose.position.x
        current_y = data.pose.pose.position.y

        # 첫 번째 데이터를 받은 경우
        if self.last_x is None:
            rospy.loginfo("첫 번째 좌표를 저장합니다: %.6f, %.6f", current_x, current_y)
            self.file.write("%.6f, %.6f\n" % (current_x, current_y))
            self.last_x = current_x
            self.last_y = current_y
            return

        # 이전 좌표와 현재 좌표 사이의 거리를 미터(m) 단위로 계산
        distance = math.sqrt((current_x - self.last_x)**2 + (current_y -self.last_y)**2)
        
        # 계산된 거리가 설정한 임계값(0.5m)보다 크거나 같으면
        if distance >= self.distance_threshold:
            rospy.loginfo("이전 지점에서 %.2f m 이동. 새 좌표 저장: %.6f, %.6f", distance, current_x, current_y)
            self.file.write("%.6f, %.6f\n" % (current_x, current_y))
            
            # 마지막 좌표를 현재 좌표로 업데이트
            self.last_x = current_x
            self.last_y = current_y

    def shutdown(self):
        """
        노드가 종료될 때 호출되는 함수
        """
        self.file.close()
        rospy.loginfo("GPS logger node shutdown and file saved.")

if __name__ == '__main__':
    try:
        GpsToTxt()
    except rospy.ROSInterruptException:
        pass