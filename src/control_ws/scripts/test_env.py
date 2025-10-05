#!/usr/bin/env python3

import rospy
from pipeline import DrappEnv  # 우리가 만든 환경 클래스를 import 합니다.

def main():
    """
    DrappEnv 환경이 안정적으로 동작하는지 테스트하기 위한 메인 함수.
    랜덤 에이전트를 사용하여 환경과 상호작용합니다.
    """
    try:
        # DrappEnv 클래스의 __init__에서 rospy.init_node()를 호출하므로,
        # 여기서는 별도로 호출할 필요가 없습니다.
        
        # 1. 환경 생성
        env = DrappEnv()
        rospy.loginfo("Environment created. Starting test...")
        
        # 2. 총 3개의 에피소드 동안 테스트를 진행
        num_episodes = 10
        for i in range(num_episodes):
            rospy.loginfo(f"--- Episode {i+1} Starting ---")
            
            # 3. 환경 리셋
            obs, info = env.reset()
            
            # 4. 각 에피소드는 최대 1000 스텝까지 진행
            for step in range(1000):
                # 5. 행동 공간에서 무작위 행동 샘플링
                random_action = env.action_space.sample()
                
                # 6. step 함수 호출하여 환경과 상호작용
                obs, reward, terminated, truncated, info = env.step(random_action)
                
                # 7. 결과 로그 출력 (디버깅에 매우 유용)
                rospy.loginfo_throttle(1.0, # 1초에 한번씩 로그 출력
                    f"Step: {step} | "
                    f"Action(LFD, Speed): [{random_action[0]:.2f}, {random_action[1]:.2f}] | "
                    f"Reward: {reward:.3f} | "
                    f"Terminated: {terminated}"
                )

                # 8. 에피소드가 종료되면 루프 탈출
                if terminated or truncated:
                    rospy.loginfo(f"Episode {i+1} finished after {step+1} steps.")
                    break
                    
    except rospy.ROSInterruptException:
        pass
    finally:
        obs, info = env.reset()
        rospy.loginfo("--- Environment Test Finished ---")

if __name__ == '__main__':
    main()