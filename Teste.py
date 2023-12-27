import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import time
from OpenAIWrappers import nosso_wrapper
from stable_baselines3 import PPO

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = nosso_wrapper(env)

modelo = PPO.load('C:/Users/Computador/Documents/modelos_salvos/n1024b64l3_318648_steps')

obs = env.reset()
recompensa_total = 0
terminou = False
while terminou!= True:
    env.render()
    time.sleep(0.02)
    acao, estado = modelo.predict(obs)
    obs, recompensa, terminou, info = env.step(acao.item())
    recompensa_total += recompensa
print('Recompensa total:', recompensa_total)
env.close()
