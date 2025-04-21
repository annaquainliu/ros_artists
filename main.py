'''
    Main.py
    
    Spawns the Processor Thread
    
'''
from Processor import Processor
from Planner import Planner
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sys

def main(args):
    if len(args) < 2:
        print("Planner::Main::Usage::python main.py <image_path_1> <image_path_2> ...")
        sys.exit()
    
    # Load the image (e.g., squirrel.png)
    images = args[1:]
    cv2_images = [cv2.imread(image) for image in images]
    
    P = Processor(cv2_images)
    
    print("Before process image")
    P.process_images()
    print("After process image")
    
    # testingData = []
    # P.process_images(testingData=testingData)
    # for sub_img in testingData:
    #     print(sub_img.shape)
    #     P.show_img(sub_img)
        
if __name__ == '__main__':
    main(sys.argv)