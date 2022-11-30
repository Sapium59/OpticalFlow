import numpy as np
import os
import cv2
from tqdm import tqdm


class VideoSaver:
    def __init__(self, save_path, FPS=30, frameSize=(1920, 1080), isSave=True, isColor=True):
        self.isSave = isSave
        if self.isSave:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(save_path, fourcc, FPS, frameSize, isColor=isColor)
 
    def write(self, frame):
        if self.isSave: 
            self.out.write(frame)
 
    def __del__(self):
        print("killed VideoSaver")
        if self.isSave: 
            self.out.release()
 

def get_optical_flow(shape, center, angle, return_dst=True):
    h, w, c = shape
    x_c, y_c = center
    
    xx, yy = np.meshgrid(np.arange(w) - y_c, np.arange(h) - x_c)
    xx = xx.astype("float32")
    yy = yy.astype("float32")
    grid_src = np.dstack((xx + x_c, yy + y_c))

    rr, tt = cv2.cartToPolar(xx, yy)
    tt += angle
    xx, yy = cv2.polarToCart(rr, tt)
    grid_dst = np.dstack((xx + x_c, yy + y_c))

    if return_dst:
        return grid_dst
    else:
        return grid_dst - grid_src


def main():
    # path
    root_dir = "G:\MyPython\OpticalFlow"
    img_path = os.path.join(root_dir, "mei.png")
    mask_path = os.path.join(root_dir, "mei_mask.png")
    video_save_path = os.path.join(root_dir, "rolling_mei.mp4")

    # load
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.dstack([mask!=0] * 3)

    # log
    # cv2.imwrite(f"G:\MyPython\OpticalFlow\img_mask.png", img*mask)
    m_Saver = VideoSaver(save_path=video_save_path, FPS=5, frameSize=img.shape[:2], isSave=True)

    # setting
    base_ang = np.pi/15
    
    for factor in tqdm(range(20)):
        cur_ang = factor * base_ang
        flow = get_optical_flow(shape=img.shape, center=(95, 100), angle=cur_ang, return_dst=True)
        img_out = cv2.remap(img*mask, flow[..., 0], flow[..., 1], cv2.INTER_LINEAR)
        img_out_path = os.path.join(root_dir, "img_out", f"img_{factor}.png")
        m_Saver.write(img_out)
        cv2.imwrite(img_out_path, img_out)


    
    

if __name__ == "__main__":
    main()
