#!/usr/bin/env python3

# ---------------------------------
# 필요한 라이브러리들을 import 합니다.
# ---------------------------------
import rospy
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ★★★ W&B 관련 라이브러리 추가 ★★★
import wandb
from wandb.integration.sb3 import WandbCallback

from pipeline import DrappEnv

# -------------------------------------------------------------------
# 학습 중 Ctrl+C를 눌렀을 때 안전하게 종료하고 모델을 저장하기 위한 콜백 클래스
# -------------------------------------------------------------------
class RosShutdownCallback(BaseCallback):
    """
    매 스텝마다 rospy.is_shutdown()을 확인하여,
    Ctrl+C가 눌렸을 때 학습을 안전하게 중단시키는 콜백.
    """
    def __init__(self, verbose=0):
        super(RosShutdownCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        이 함수는 훈련 중 매 스텝마다 호출됩니다.
        False를 반환하면 훈련이 중단됩니다.
        """
        # rospy.is_shutdown()이 True가 되면 (Ctrl+C를 누르면)
        if rospy.is_shutdown():
            rospy.logwarn("ROS shutdown signal received. Stopping training...")
            return False # 훈련 루프에 "중단하라"는 신호를 보냅니다.
        
        # 그 외의 경우에는 계속 훈련을 진행합니다.
        return True


# ---------------------------------
# 메인 실행 함수
# ---------------------------------
def main():
    """
    Stable Baselines3와 DrappEnv를 사용하여 RL 에이전트를 훈련시키는 메인 함수.
    """
    # ★★★ 1. W&B 실행(run) 초기화 ★★★
    # 이 코드는 학습 시작 전에 W&B 서버에 새로운 실험을 등록합니다.
    run = wandb.init(
        project="KSAE_FSD_2025_TD3",  # W&B 프로젝트 이름 (자유롭게 설정)
        name="td3_with_cornering_reward_v1", # 이번 실험의 이름 (자유롭게 설정)
        sync_tensorboard=True,      # SB3의 TensorBoard 로그를 W&B로 자동 동기화
        monitor_gym=True,           # 환경의 통계(보상 등)를 자동으로 기록
        save_code=True,             # 실행 코드를 W&B에 저장하여 재현성 확보
    )

    try:
        # 1. 강화학습 환경을 생성합니다.
        # DrappEnv의 __init__ 메소드에서 rospy.init_node()가 호출됩니다.
        env = DrappEnv()
        rospy.loginfo("Environment created. Starting training...")

        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs = 10.0)
        rospy.loginfo("Environment wrapped with VecNormalize.")
        # 2. 액션 노이즈(Action Noise)를 설정합니다.
        # 에이전트의 탐험(Exploration)을 돕기 위해 행동에 약간의 무작위성을 부여합니다.
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))
        
        # 3. 안전 종료 콜백 객체를 생성합니다.
        shutdown_callback = RosShutdownCallback()

        # ★★★ 2. W&B 콜백 생성 ★★★
        # SB3가 학습하는 동안 W&B에 로그를 보내도록 설정합니다.
        wandb_callback = WandbCallback(
            model_save_path=f"models/{run.id}", # W&B에 모델을 주기적으로 저장하려면 주석 해제
            verbose=2,
        )
        # ★★★ 3. 여러 콜백을 리스트로 묶기 ★★★
        # 기존의 RosShutdownCallback과 새로운 WandbCallback을 함께 사용합니다.
        callback_list = CallbackList([shutdown_callback, wandb_callback])

        # 4. TD3 에이전트(모델)를 생성하고 하이퍼파라미터를 설정합니다.
        model = TD3(
            policy="MlpPolicy",         # 표준 다층 퍼셉트론 신경망 정책
            env=env,                    # 사용할 환경
            action_noise=action_noise,  # 액션 노이즈
            buffer_size=500000,         # 리플레이 버퍼 크기 (50만)
            learning_rate=1e-4,         # 학습률 (0.0001) - 낮춰서 안정성 확보
            gamma=0.99,                 # 할인 계수
            batch_size=256,              # 배치 사이즈
            verbose=1,                  # 훈련 과정 로그를 터미널에 출력
            tensorboard_log=f"runs/{run.id}"  # TensorBoard 로그 저장 경로
        )

        # 5. 모델 훈련을 시작합니다.
        rospy.loginfo("Starting model training...")
        model.learn(
            total_timesteps=1000000,     # 총 훈련 스텝 수 (100만)
            log_interval=1,            # 로그 출력 간격 (10 에피소드마다)
            callback=callback_list  # 매 스텝마다 안전 종료 콜백을 실행
        )

        # 6. 훈련이 정상적으로 끝나거나 Ctrl+C로 중단되면, 최종 모델을 저장합니다.
        rospy.loginfo("Training finished or stopped. Saving final model...")
        model.save("td3_drapp_agent_final")
        env.save("vec_normalize.pkl")  # VecNormalize 객체도 저장
        
    except rospy.ROSInterruptException:
        # Ctrl+C가 눌리면 rospy.ROSInterruptException이 발생하지만,
        # 콜백이 정상적으로 처리하므로 여기서는 특별한 작업을 할 필요가 없습니다.
        rospy.logwarn("Training process interrupted by user.")
    finally:
        rospy.loginfo("--- Training script finished ---")
        run.finish()


if __name__ == '__main__':
    main()