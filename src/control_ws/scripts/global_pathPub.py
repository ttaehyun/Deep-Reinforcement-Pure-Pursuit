#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def create_path_from_file(file_path):
    """
    텍스트 파일을 읽어 nav_msgs/Path 메시지를 생성합니다.
    """
    # 1. Path 메시지 객체 생성
    path = Path()
    # Path 메시지의 헤더 설정. frame_id는 RViz의 Fixed Frame과 일치해야 합니다.
    path.header.frame_id = "map"
    path.header.stamp = rospy.Time.now()

    # 이전에 추가된 좌표를 저장하기 위한 변수 --- [수정된 부분]
    last_pose = None 

    try:
        # 2. 텍스트 파일 열기
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                # 주석(#)이거나 빈 줄은 무시
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                
                # 3. 각 줄을 파싱하여 x, y, z 좌표 추출
                data = line.split(',')
                if len(data) < 2:  # 최소 x, y 좌표는 있어야 함
                    continue

                # 현재 좌표 추출 --- [수정된 부분]
                current_x = float(data[0])
                current_y = float(data[1])
                # 이전 좌표와 현재 좌표가 중복되는지 확인 --- [수정된 부분]
                if last_pose and \
                   last_pose.pose.position.x == current_x and \
                   last_pose.pose.position.y == current_y:
                    rospy.logdebug("중복 좌표 [%f, %f]를 건너뜁니다.", current_x, current_y)
                    continue # 중복되면 다음 줄로 넘어감
                # 4. PoseStamped 메시지 생성
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "map"
                
                # 위치 정보 설정
                pose.pose.position.x = current_x
                pose.pose.position.y = current_y
                # z 좌표가 파일에 있으면 사용, 없으면 0으로 설정
                if len(data) > 2:
                    pose.pose.position.z = float(data[2])
                else:
                    pose.pose.position.z = 0.0
                
                # 방향(orientation) 정보 설정 (기본값: 정면)
                pose.pose.orientation.w = 1.0
                
                # 5. Path 메시지의 poses 리스트에 추가
                path.poses.append(pose)

                # 마지막으로 추가된 pose 업데이트 --- [수정된 부분]
                last_pose = pose
    except IOError as e:
        rospy.logerr("파일을 읽는 중 오류 발생: %s", e)
        return None
        
    return path

if __name__ == '__main__':
    rospy.init_node('path_publisher_node')

    # '/trajectory' 토픽에 Path 메시지를 발행하는 Publisher 생성
    path_pub = rospy.Publisher('/trajectory', Path, queue_size=10, latch=True)
    
    # 파일 경로를 파라미터 서버에서 가져오거나 직접 지정
    # 'rosparam set path_file_path /경로/to/my_path.txt' 와 같이 설정 가능
    file_path = '/home/avl/Deep-Reinforcement-Pure-Pursuit/src/control_ws/path/competition_map_testday2_centerline.txt'

    rospy.loginfo("경로 파일 '%s'을(를) 읽습니다.", file_path)
    
    # 파일로부터 Path 메시지 생성
    my_path_msg = create_path_from_file(file_path)
    
    if my_path_msg:
        # 생성된 경로를 발행 (latch=True 옵션으로 한 번만 발행해도 유지됨)
        path_pub.publish(my_path_msg)
        rospy.loginfo("Path 메시지를 '/trajectory' 토픽으로 발행했습니다.")
    
    # 노드가 종료되지 않도록 유지
    rospy.spin()