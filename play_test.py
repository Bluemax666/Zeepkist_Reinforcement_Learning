import time
from new_zeep_env import Env
from sb3_contrib import TQC
time.sleep(6)
env = Env(nb_actions=1)

model = TQC.load("Noel_zeep_map3_1")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
        
    if done: 
        obs = env.reset()
    
    
    
    

