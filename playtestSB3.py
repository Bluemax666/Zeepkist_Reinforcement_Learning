# -*- coding: utf-8 -*-
"""
Created on Sun May  1 23:29:41 2022

@author: maxim
"""

import time
#from mania_env import Env
from zeep_env import Env
from sb3_contrib import TQC
from utils.getKeys import getKeys
time.sleep(1)
#env = Env(3, "model_trackroad.torch", visualize_inputs=True, visualize_frame=False)
env = Env(1, visualize=True)
time.sleep(6)
#TQC_chicane_32
model = TQC.load("runs/Zeep_03_run3")

obs = env.reset()
lt = time.time()
while True:
    action, _states = model.predict(obs, deterministic=True)
    
    keys = getKeys()
    if 'P' in keys:
        obs, reward, done, info = env.step(action)
    else:
        obs, reward, done, info = env.step(action)
        
    if done: 
        # if env.cp_count >=1:
        #     obs = env.reset_at_cp()
        # else:
        obs = env.reset()
    
    
    nt = time.time()
    #print("reward : ",round(reward,2), " fps : ",round(1/(nt-lt),2))
    lt = nt
    
    
    

