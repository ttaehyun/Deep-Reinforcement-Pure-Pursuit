#!/usr/bin/env python3
import time # time 모듈을 import 해야 합니다.
import rospy
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from run_pipeline import DrappEnv # 학습할 때 사용했던 환경 클래스를 그대로 import 합니다.

def main():
    """
    미리 학습된 TD3 에이전트를 불러와 자율주행을 수행하는 메인 함수.
    """
    # --- 1. 환경 생성 ---
    # 제어를 수행할 환경을 생성합니다. 학습할 때와 동일한 환경이어야 합니다.
    env = DrappEnv()
    env = DummyVecEnv([lambda: env])

    # --- 2. 정규화 상태 불러오기 (매우 중요!) ---
    # 학습할 때 사용했던 Observation 정규화 상태(평균, 분산 등)를 불러옵니다.
    # 이 파일이 없으면 에이전트가 세상을 완전히 다르게 인식하여 오작동합니다.
    # 저장된 zip 파일과 함께 'vec_normalize.pkl' 파일이 생성되었을 것입니다.
    env = VecNormalize.load("competition_map2_v13.pkl", env)
    
    # VecNormalize를 불러온 후에는 더 이상 학습하거나 통계를 업데이트하지 않도록 설정해야 합니다.
    env.training = False
    env.norm_reward = False
    
    rospy.loginfo("Environment and Normalization stats loaded.")

    # --- 3. 학습된 모델 불러오기 ---
    # 학습을 통해 저장된 모델 파일의 경로를 지정합니다.
    # 예: "td3_drapp_agent_final.zip" 또는 "models/RUN_ID/rl_model_100000_steps.zip"
    model_path = "competition_map2_v13.zip" 
    model = TD3.load(model_path, env=env)
    rospy.loginfo(f"Model loaded from {model_path}")

    # --- 4. 제어 루프 실행 ---
    # 에이전트가 환경과 상호작용하며 주행을 시작합니다.
    obs = env.reset()
    while not rospy.is_shutdown():
        # model.predict() 함수를 사용하여 현재 관측(obs)에 대한 최적의 행동(action)을 결정합니다.
        # deterministic=True는 탐험(Exploration)을 위한 노이즈를 추가하지 않고,
        # 학습된 정책에 따라 가장 확률이 높은 행동만 선택하도록 합니다.
        start_comp_time = time.perf_counter()

        action, _states = model.predict(obs, deterministic=True)
        
        # 결정된 행동을 환경에 전달하고, 다음 상태(obs)와 보상 등을 받습니다.
        # 제어 시에는 보상(reward), 종료 여부(done) 등은 보통 사용하지 않습니다.
        obs, reward, done, info = env.step(action)

        end_comp_time = time.perf_counter()
        computation_time = end_comp_time - start_comp_time
        # env.unwrapped.envs[0]를 통해 원래의 DrappEnv 객체에 접근
        env.unwrapped.envs[0].episode_computation_times.append(computation_time)
        # 에피소드가 종료되면 (예: 경로 이탈) 환경을 리셋합니다.
        if done:
            rospy.logwarn("Episode finished. Resetting environment.")
            obs = env.reset()

if __name__ == '__main__':
    main()