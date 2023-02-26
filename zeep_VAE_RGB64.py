import cv2
import torch
import numpy as np
import win32gui, win32ui, win32con, win32api

from model_VAE import VAE


class Game_VAE:
    """This class is used to encode the images of the game"""
    def __init__(self, visualize_frame=False):
        self.SCREEN_GRAB_COORDS = [250,300,1030,650]
        self.VAE_model_name = "Zeepkist_image_VAE.torch"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device VAE encode : {}".format(self.device))
        self.VAE_dims = 16
        self.vae = self.load_VAE(self.VAE_model_name, self.VAE_dims)
        self.visualize_frame = visualize_frame
        
    def load_VAE(self, model_name, latent_dim):
        vae = VAE(3, latent_dim)
        vae.load_state_dict(torch.load(model_name, map_location='cpu')) 
        vae = vae.to(self.device)
        return vae    
    
    def get_frame(self):
        """returns the encoded image"""
        
        self.frame = self.record_frame()
        
        self.z_frame = self.encode_frame(self.frame)
        
        return self.z_frame
    
    def record_frame(self):
        """returns the image of the game"""
        
        self.frame = grabScreen(self.SCREEN_GRAB_COORDS)
        self.frame = cv2.resize(self.frame, (64, 64))
        
        return self.frame
    
    def encode_frame(self, frame):
        
        self.im_arr = np.reshape(frame,(1, 3, 64, 64))
        self.im_arr = self.im_arr / 255
        
        self.image_tensor = torch.Tensor(self.im_arr).to(self.device)
        self.z_frame = self.vae.encode(self.image_tensor)[0]
        
        if self.visualize_frame:
            self.show_reconstruted_image(self.z_frame)
            
        self.z_frame = self.z_frame.cpu().detach().numpy()[0]
        
        return self.z_frame
    
    def show_reconstruted_image(self, z_frame):
        """displays a window with the image reconstructed from an encoded image"""
        
        image_tensor = self.vae.decode(z_frame)
        image = image_tensor.cpu().detach().numpy()[0]
        image = np.reshape(image,(64,64,3))
        image = cv2.resize(image,(400,200),interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imshow("reconstruted_image",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() 
            quit()
    
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
    
