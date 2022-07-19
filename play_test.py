# -*- coding: utf-8 -*-
"""
Created on Sun May  1 23:29:41 2022

@author: maxim
"""

import time
from zeep_env import Env
from sb3_contrib import TQC
time.sleep(1)
env = Env(1)

model = TQC.load("Zeep_03_run3")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
        
    if done: 
        obs = env.reset()
    
    
    
    

