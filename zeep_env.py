# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:51:46 2019

@author: maxime
"""

import numpy as np
import cv2
import torch
import time
from gym import spaces
from model_VAE import VAE
from collections import deque
import tesserocr
from PIL import Image
import pygame
import vgamepad as vg
import win32gui, win32ui, win32con, win32api

class Env:
    ENV_NAME = "Zeepkist"

    GAME_WINDOW_COORDS = [0, 30, 1280, 750]
    WI_WIDTH = GAME_WINDOW_COORDS[2] - GAME_WINDOW_COORDS[0]
    WI_HEIGHT = GAME_WINDOW_COORDS[3] - GAME_WINDOW_COORDS[1]

    VAE_dims = 16
    VAE_model_name = "Zeepkist_image_VAE.torch"
    
    EPISODE_DURATION = 100 #1 min 40s
    
    total_cp_number = 6
    checkpoint_reward = 60
    finish_reward = 100
    crash_reward = -25
    
    FPS_limit = 20
    
    episode_duration = EPISODE_DURATION
    FPS_limit = FPS_limit
    
    joystick_deadzone = 0.2
    steering_factor = 1.2
    brake_treshold = 0.5
    
    def __init__(self, nb_actions):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device VAE encode : {}".format(self.device))
        self.vae = self.load_VAE(self.VAE_model_name, self.VAE_dims)
        self.nb_actions = nb_actions
        self.state_length = (self.VAE_dims + self.nb_actions + 1) * 2
            
        self.states_history = deque(maxlen=16)
        self.speed_list = deque(maxlen=6)
           
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_length,), dtype=np.float32)
        print("Observation space:", self.observation_space)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.nb_actions,), dtype=np.float32)
        print("Action space:", self.action_space)
        self.reward_range = 1
        print("reward range:", self.reward_range)
        self.metadata = ""
        
        self.gamepad = vg.VX360Gamepad()
        self.ocr_api = tesserocr.PyTessBaseAPI(path="D:/Tesseract-OCR/tessdata")
            
    
    def reset(self):
        click_x, click_y = self.rel_to_abs_coords([0.063, 0.097,0,0], self.GAME_WINDOW_COORDS)[:2]
        try:
            self.reset_joy()
                
            leftClick(click_x, click_y)
            time.sleep(0.1) 
            self.press_Y(0.5)
            time.sleep(0.5)
            self.press_A(0.5)
            time.sleep(0.5)
            lt = time.time()
            while self.get_speed() < 0:
                if time.time() - lt > 10:
                    leftClick(click_x, click_y)
                    time.sleep(0.1) 
                    self.press_Y(0.5)
                    time.sleep(0.5)
                    self.press_A(0.5)
                    time.sleep(0.5)
                    lt = time.time()
                    
                time.sleep(0.1)
            
        except Exception as e:
            print(e)
        
        self.cp_count = 0
        self.z_frame = self.get_frame()
        self.action = np.zeros((self.nb_actions))
        self.speed = [self.get_speed()]
        
        first_obs = np.concatenate((self.z_frame, self.speed, self.action))
        for i in range(16):
            self.states_history.append(first_obs)
            
        self.state = self.get_state() 
        self.episode_start = time.time()
        self.last_time = time.time()
        
        return self.state
    
    
    def get_speed_reward(self, speed):
        return np.power((speed/150), 1) - 0.35
    
    
    def step(self, action):   
        
        self.steer = action[0]
        if abs(self.steer) < self.joystick_deadzone:
            self.steer = 0
            
        self.joystick_turn(self.steer)
        self.gamepad.update()
        
        if self.nb_actions >= 2:
            if action[1] >= self.brake_treshold:
                self.brake()
            else:
                self.brake_release()
                
        if self.nb_actions >= 3:
            if action[2] >= 0:
                self.stabilize()
            else:
                self.stabilize_release() 
        
        self.limit_FPS()
        
        done = False
        speed = self.get_speed()
        self.speed_list.append(speed)
        new_cp_count = self.get_cp_count()  
        current_time = time.time() - self.episode_start
                
        speed_reward = self.get_speed_reward(speed)
        event_reward = 0
        
        if new_cp_count != self.cp_count:
            if self.get_cp_count() != self.total_cp_number:
                event_reward = self.checkpoint_reward
            else:
                event_reward = self.finish_reward
                done = True
                
            self.cp_count = new_cp_count
                
        if current_time > self.episode_duration:
            done = True
            
        if current_time > 4 and np.average(self.speed_list) < 15:
            done = True
            event_reward = self.crash_reward
        
        reward = speed_reward + event_reward
                
        self.action = action
        
        self.z_frame = self.get_frame() 
        self.speed = [(speed/150)- 1]
        obs = np.concatenate((self.z_frame, self.speed, self.action))
        self.states_history.append(obs)
        self.state = self.get_state()
        info = {}
        
        return self.state, reward, done, info
          
    
    def get_frame(self):

        self.frame = self.record_frame()
        
        self.z_frame = self.encode_frame(self.frame)
        
        return self.z_frame
    
    def record_frame(self):
        RELATIVE_SCREEN_GRAB_COORDS = [0.195, 0.375, 0.804, 0.861]
        self.screen_grab_coords = self.rel_to_abs_coords(RELATIVE_SCREEN_GRAB_COORDS, self.GAME_WINDOW_COORDS)
        self.frame = grabScreen(self.screen_grab_coords)
        self.frame = cv2.resize(self.frame, (64, 64))
        
        return self.frame
    
    def encode_frame(self, frame):
        
        self.im_arr = np.reshape(frame,(1, 3, 64, 64))
        self.im_arr = self.im_arr / 255
        
        self.image_tensor = torch.Tensor(self.im_arr).to(self.device)
        self.z_frame = self.vae.encode(self.image_tensor)[0]
            
        self.z_frame = self.z_frame.cpu().detach().numpy()[0]
        
        return self.z_frame
    
    
    def limit_FPS(self):
        if self.FPS_limit != -1:
            mil_delay = int(((1/self.FPS_limit) - (time.time()-self.last_time))*1000)
            if mil_delay > 0 and mil_delay < 1000/self.FPS_limit:
                pygame.time.delay(mil_delay)
            self.last_time = time.time()
    
    
    def action_space_sample(self):
        return (np.random.rand(self.nb_actions)*2)-1
    
    
    def get_state(self):
        current_state = np.array(self.states_history)[-1:]
        derivative = np.array([self.states_history[-1]]) - np.array([self.states_history[-2]])
        state = np.concatenate((current_state, derivative)).flatten()    
        return state
  
 
    def get_speed(self):
        RELATIVE_SPEED_COORDS =  [0.852, 0.854, 0.971, 0.942]
        self.speed_coords = self.rel_to_abs_coords(RELATIVE_SPEED_COORDS, self.GAME_WINDOW_COORDS)
        speed_img = grabScreen(self.speed_coords)
        mask = cv2.inRange(speed_img, np.array([250,250,250]), np.array([255,255,255]))
        mask = cv2.bitwise_not(mask)
        pil_img = Image.fromarray(mask)
        self.ocr_api.SetImage(pil_img)
        text = self.ocr_api.GetUTF8Text()
        try:
            if len(text) > 0:
                speed = int(text)
                if 0 <= speed <= 999:
                    return speed
            
            return -1
            
        except Exception as e:
            return -1
    
    def get_cp_count(self):
        
        RELATIVE_CP_COORDS = [0.005, 0.865, 0.130, 0.988]
        self.cp_coords = self.rel_to_abs_coords(RELATIVE_CP_COORDS, self.GAME_WINDOW_COORDS)
        cp_img = grabScreen(self.cp_coords)
        mask = cv2.inRange(cp_img, np.array([250,250,250]), np.array([255,255,255]))
        mask = cv2.bitwise_not(mask)
        pil_img = Image.fromarray(mask)
        self.ocr_api.SetImage(pil_img)
        text = self.ocr_api.GetUTF8Text()
        try:
            if len(text) > 0:
                nb_cp = int(text[0])
                if 0 <= nb_cp <= self.total_cp_number and self.cp_count <= nb_cp <= self.cp_count+1:
                    return nb_cp
                
            return self.cp_count
        
        except Exception as e:
            return self.cp_count
        
    
    def joystick_turn(self, val):
        val = np.clip(val*1.2, -1, 1)
        self.gamepad.left_joystick_float(x_value_float=val, y_value_float=0.0)
    
    def press_A(self, duration):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(duration)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        
    def press_Y(self, duration):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        self.gamepad.update()
        time.sleep(duration)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        self.gamepad.update()
        
    
    def stabilize(self):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        
    def stabilize_release(self):
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    
    def brake(self):
         self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        
    def brake_release(self):
         self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)    
        
    def reset_joy(self):
        self.gamepad.reset()
        self.gamepad.update()
    
    def load_VAE(self, model_name, latent_dim):
        vae = VAE(3, latent_dim)
        vae.load_state_dict(torch.load(model_name, map_location='cpu')) 
        vae = vae.to(self.device)
        return vae

    def rel_to_abs_coords(self, rel_coords, win_coords):
        abs_x1 = round(win_coords[0] + rel_coords[0]*self.WI_WIDTH)
        abs_y1 = round(win_coords[1] + rel_coords[1]*self.WI_HEIGHT)
        
        abs_x2 = round(win_coords[0] + rel_coords[2]*self.WI_WIDTH)
        abs_y2 = round(win_coords[1] + rel_coords[3]*self.WI_HEIGHT)
        return [abs_x1, abs_y1, abs_x2, abs_y2]
        
        
    def abs_to_rel_coords(self, abs_coords, win_coords):
        rel_x1 = (abs_coords[0] - win_coords[0]) / self.WI_WIDTH
        rel_y1 = (abs_coords[1] - win_coords[1]) / self.WI_HEIGHT
        
        rel_x2 = (abs_coords[2] - win_coords[0]) / self.WI_WIDTH
        rel_y2 = (abs_coords[3] - win_coords[1]) / self.WI_HEIGHT
        return [rel_x1, rel_y1, rel_x2, rel_y2]
    
    def render(self):
        pass
    

def leftClick(x, y, duration=0.1):
    win32api.SetCursorPos((x,y))
    time.sleep(0.016)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(duration)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def grabScreen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
