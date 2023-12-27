import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import time
from OpenAIWrappers import nosso_wrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def cria_env(nome_env):
    def _init():
        env = gym_super_mario_bros.make(nome_env)
        env = JoypadSpace(env, RIGHT_ONLY)
        env = nosso_wrapper(env)
        return env
    return _init

if __name__ == '__main__':
    threads_cpu = 8
    env = SubprocVecEnv([cria_env(nome_env='SuperMarioBros-1-1-v0') for i in range(threads_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=1562, save_path='C:/Users/Computador/Documents/modelos_salvos/', name_prefix='n1024b64l3')

    modelo = PPO(policy='CnnPolicy', env=env, gamma=0.99, n_steps=1024, batch_size=64, learning_rate=0.0003, vf_coef=0.5, ent_coef=0.01, verbose=1)
    modelo.load('C:/Users/Computador/Documents/modelos_salvos/n1024b64l3_318648_steps')

    inicio = time.time()

    modelo.learn(total_timesteps=10000000, log_interval=10, callback=checkpoint_callback)

    fim = time.time()
    print('Tempo total do treinamento em horas:', (fim-inicio)/3600)
