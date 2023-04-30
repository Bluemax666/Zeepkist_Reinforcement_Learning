import numpy as np
import win32gui, win32ui, win32con, win32api #windows api to capture screenshots fast
from zeep_VAE_RGB64 import Game_VAE #its the network that is used to encode the image
from datetime import datetime
from collections import deque
from gym import spaces #used by the learning algorithm
import vgamepad as vg #to simulate a virtual gamepad
from PIL import Image #to convert to Image object for tesserocr  
import tesserocr #to read text from images
import pygame #pygame.time.delay(mil_delay) is more precise than time.sleep()
import cv2 #to manipulate images
import time #time.sleep()


ENV_NAME = "Zeepkist"

WI_WIDTH = 1280
WI_HEIGHT = 720

EPISODE_DURATION = 120

checkpoint_reward = 60
finish_reward = 120
crash_reward = -20

FPS_limit = 20

#number of previous frames we give as input to the AI 
agent_history_length = 1

#if we want to give the difference
#between the 2 last observations as input to the AI
give_derivative = True

class Env:
    env_name = ENV_NAME
    wi_width = WI_WIDTH
    wi_height = WI_HEIGHT
    
    episode_duration = EPISODE_DURATION
    FPS_limit = FPS_limit
    
    if give_derivative:
        added_seq_len = 1
    else:
        added_seq_len = 0
    
    joystick_deadzone = 0.2
    steering_factor = 1.2
    brake_treshold = 0.5
    
    global_steps = 0
    finish_times = []
    
    def __init__(self, nb_actions):
        self.vae = Game_VAE()
        self.VAE_dims = self.vae.VAE_dims
        
        self.nb_actions = nb_actions
        self.agent_history_length = agent_history_length
        self.state_length = self.VAE_dims + self.nb_actions + 1
        self.states_length = self.state_length*(self.agent_history_length+self.added_seq_len)
            
        self.states_history = deque(maxlen=16)
        self.speed_list = deque(maxlen=6)
           
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.states_length,), dtype=np.float32)
        print("Observation space:", self.observation_space)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.nb_actions,), dtype=np.float32)
        print("Action space:", self.action_space)
        self.reward_range = 1
        print("reward range:", self.reward_range)
        self.metadata = ""
        
        self.gamepad = vg.VX360Gamepad()
        
        #you might need to change the path
        self.ocr_api = tesserocr.PyTessBaseAPI(path="D:/Tesseract-OCR/tessdata")
        
        self.white_color_lower_bound = np.array([253,253,253])
        self.white_color_upper_bound = np.array([255,255,255])
        self.episode_reward = 0  
        self.cp_count = 0
        self.finished = False
    
    def reset(self):
        """Resets the car at the start of the track
           and returns the first observation"""
        
        try:
            self.reset_joy()
                
            leftClick(80,100)
            time.sleep(0.1) 
            self.press_Y(0.5)
            time.sleep(0.5)
            if self.finished:
                self.press_Dpad_left(0.5)
                time.sleep(0.5)
                
            self.press_A(0.5)
            time.sleep(0.5)
            
            
            lt = time.time()
            while self.get_speed() < 0:
                if time.time() - lt > 10:
                    leftClick(80,100)
                    time.sleep(0.1) 
                    self.press_Y(0.5)
                    time.sleep(0.5)
                    self.press_A(0.5)
                    time.sleep(0.5)
                    lt = time.time()
                    
                time.sleep(0.1)
            
        except Exception as e:
            print(e)
            
        
        self.finished = False    

        self.cp_count = 0
        self.z_frame = self.vae.get_frame()
        self.action = np.zeros((self.nb_actions))
        self.speed = [self.get_speed()]
        self.last_speed = 0
        
        first_obs = np.concatenate((self.z_frame, self.speed, self.action))
        for i in range(16):
            self.states_history.append(first_obs)
            
        self.state = self.get_state() 
        self.episode_start = time.time()
        self.last_time = time.time()
        
        return self.state
    
    
    def get_speed_reward(self, speed):
        return np.power((speed/160), 1) - 0.30
    
    
    def step(self, action, debug=False):
        """
        Applies the action to the game and returns
        the next observation, the reward and the done boolean
                  action :
            
          [0]       [1]         [2]
         steer     brake   stabilisation
        """
        
        #action can have from 1 to 3 elements and is applied to the game
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
            
        self.steer = action[0]
        if abs(self.steer) < self.joystick_deadzone:
            self.steer = 0
        else:
            self.steer = (self.steer - self.joystick_deadzone*np.sign(self.steer))*(1/(1-self.joystick_deadzone))
            
        self.joystick_turn(self.steer)
        
        if not debug:
            self.gamepad.update()
        
        #a delay is added to get a framerate of self.FPS_limit 
        if self.FPS_limit != -1:
            mil_delay = int(((1/self.FPS_limit) - (time.time()-self.last_time))*1000)
            if mil_delay > 0 and mil_delay < 1000/self.FPS_limit:
                pygame.time.delay(mil_delay)
            self.last_time = time.time()
        
        self.global_steps += 1
        
        #we read the speed and the number checkpoints passed,
        #determine if the episode is ended and compute the reward
        done = False
        speed = self.get_speed()
        self.speed_list.append(speed)
        new_cp_count = self.get_cp_count()  
        current_time = time.time() - self.episode_start
                
        speed_reward = self.get_speed_reward(speed)
        event_reward = 0
        
        if new_cp_count != self.cp_count:   #passed a checkpoint
            event_reward = checkpoint_reward
            self.cp_count = new_cp_count
                
        if current_time > self.episode_duration:    #episode maximum time exeeded
            done = True
            
        if current_time > 4 and np.average(self.speed_list) < 15: #too slow or crash/finish
            done = True
            if self.is_finished():
                event_reward = finish_reward
                self.finished = True
                
            else:    
                event_reward = crash_reward
        
        reward = speed_reward + event_reward
        self.episode_reward += reward
                
        self.action = action
        self.last_speed = speed
        
        #then we contruct the next observation
        self.z_frame = self.vae.get_frame() 
        self.speed = [(speed/150)- 1]
        obs = np.concatenate((self.z_frame, self.speed, self.action))
        self.states_history.append(obs)
        self.state = self.get_state()
        info = {}
        
        return self.state, reward, done, info
          
    
    def action_space_sample(self):
        """returns a random action"""
        return (np.random.rand(self.nb_actions)*2)-1
    
    
    def get_state(self):
        """agent_history_length contains the previous states/observations and
        from that it returns the state that will be given to the actor network"""
        
        used_states_history = np.array(self.states_history)[-self.agent_history_length:]
        
        if give_derivative:
            derivative = np.array([self.states_history[-1]]) - np.array([self.states_history[-2]])
            state = np.concatenate((used_states_history, derivative)).flatten()    
        else:
            state = used_states_history.flatten()  
            
        return state
            
    def read_text(self, img):
        """returns the text that is written on an image"""
        
        pil_img = Image.fromarray(img)
        self.ocr_api.SetImage(pil_img)
        
        return  self.ocr_api.GetUTF8Text()
    
        
    def is_finished(self):
        """returns True if the car has finished the map"""
        
        time.sleep(0.5)
        frame = grabScreen([323, 92, 948, 220])
        mask = cv2.inRange(frame, self.white_color_lower_bound, self.white_color_upper_bound)
        mask = cv2.bitwise_not(mask)
        text = self.read_text(mask).strip()
        if text.find("Crash") >= 0:
            return False
        else:
            try:
                t = datetime.strptime(text,'%M:%S.%f')
                finish_time = t.minute*60+t.second+t.microsecond/1_000_000
                self.finish_times.append([self.global_steps, finish_time])
                return True
            except Exception as e:
                pass
            
            return False
  
 
    def get_speed(self):
        """reads the speed from the screen of the game"""
        
        speed_img = grabScreen([1090,645,1243,708])
        mask = cv2.inRange(speed_img, self.white_color_lower_bound, self.white_color_upper_bound)
        mask = cv2.bitwise_not(mask)
        text = self.read_text(mask)
        try:
            if len(text) > 0:
                speed = int(text)
                if 0 <= speed <= 999:
                    return speed
            
            return -1
            
        except Exception as e:
            return -1
    
    def get_cp_count(self):
        """reads the number of checkpoints passed from the screen of the game"""
        
        cp_img = grabScreen([6,653,166,741])
        mask = cv2.inRange(cp_img, self.white_color_lower_bound, self.white_color_upper_bound)
        mask = cv2.bitwise_not(mask)
        text = self.read_text(mask)
        try:
            if len(text) > 0:
                nb_cp = int(text[0])
                if 0 <= nb_cp <= self.total_cp_number and self.cp_count <= nb_cp <= self.cp_count+1:
                    return nb_cp
                
            return self.cp_count
        
        except Exception as e:
            return self.cp_count
    
    def joystick_turn(self, val):
        """Turns the jostick of the vistual controller
        -1 is steer full left and 1 is steer full right"""
        
        val = np.clip(val*1.2, -1, 1)
        self.gamepad.left_joystick_float(x_value_float=val, y_value_float=0.0)
    
    """ Presses the buttons on the virtual Xbox controller """
    
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
        
    def press_Dpad_left(self, duration):
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self.gamepad.update()
        time.sleep(duration)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
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
             
    
    def render(self):
        pass
    

def leftClick(x, y, duration=0.1):
    """left clicks at the location x y on the screeen with the mouse"""
    win32api.SetCursorPos((x,y))
    time.sleep(0.016)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(duration)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def grabScreen(region=None):
    """returns the screen of the game in the region 
    when region=[p1.x, p1.y, p2.x, p2.y] its the box from p1 to p2"""
    
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
         
