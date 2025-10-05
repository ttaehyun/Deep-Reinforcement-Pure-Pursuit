#!/usr/bin/env python3

import rospy
from pynput import keyboard
from fs_msgs.msg import ControlCommand
from fs_msgs.srv import Reset
class KeyboardControlNode:
    def __init__(self):
        rospy.init_node('keyboard_control_node', anonymous=True)

        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        # 서비스 클라이언트를 초기화 시에 한 번만 생성합니다.
        try:
            rospy.wait_for_service('/fsds/reset', timeout=5.0)
            self.reset_client = rospy.ServiceProxy('/fsds/reset', Reset)
            rospy.loginfo("Reset service client is ready.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service /fsds/reset not available: {e}")
            self.reset_client = None
        except rospy.ROSException:
            rospy.logerr("Timeout waiting for /fsds/reset service.")
            self.reset_client = None

        self.throttle = 0.0
        self.steering = 0.0
        self.brake = 0.0
        self.speed_level = 2.0
        self.max_throttle = 1.0

        self.pressed_keys = set()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        self.rate = rospy.Rate(50)  # 50 Hz

    def on_press(self, key):
        self.pressed_keys.add(key)

        try:
            # 이 블록은 key에 .char 속성이 있을 때만 시도됩니다.
            if key.char.lower() == 'i':
                self.reset()
                print("I key pressed")

        except AttributeError:
            # key에 .char 속성이 없어 오류가 나면 이 부분이 실행됩니다.
            # 특수 키의 경우이므로, 아무것도 안하고 그냥 넘어갑니다.
            pass

    def on_release(self, key):
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
    def run(self):
        while not rospy.is_shutdown():
            self.update_control_values()
            self.publish_control_command()
            self.rate.sleep()

    def update_control_values(self):
        for i in range(1, 9):
            if keyboard.KeyCode.from_char(str(i)) in self.pressed_keys:
                self.speed_level = float(i)
                break
        
        current_throttle = 0.0
        current_steering = 0.0
        current_brake = 0.0
        
        if keyboard.Key.up in self.pressed_keys and keyboard.Key.down in self.pressed_keys:
            current_throttle = 0.0
            current_brake = 1.0
        elif keyboard.Key.up in self.pressed_keys:
            current_throttle = self.max_throttle * (self.speed_level / 8.0)
        elif keyboard.Key.down in self.pressed_keys:
            current_brake = 1.0

        if keyboard.Key.left in self.pressed_keys and keyboard.Key.right in self.pressed_keys:
            current_steering = 0.0
        elif keyboard.Key.left in self.pressed_keys:
            current_steering = -1.0
        elif keyboard.Key.right in self.pressed_keys:
            current_steering = 1.0
        self.throttle = current_throttle
        self.steering = current_steering
        self.brake = current_brake
    
    def publish_control_command(self):
        control_msg = ControlCommand()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.throttle = self.throttle
        control_msg.steering = self.steering
        control_msg.brake = self.brake

        self.control_pub.publish(control_msg)
        
        rospy.loginfo(f"Throttle: {self.throttle:.2f}, Steering: {self.steering:.2f}, Brake: {self.brake:.2f}, Speed Level: {self.speed_level:.1f}")

    def reset(self):
        if self.reset_client is None:
            rospy.logwarn("Reset service is not available.")
            return
        
        try:
            self.reset_client(True)

            rospy.loginfo("Environment reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
           
if __name__ == '__main__':
    try:
        node = KeyboardControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass