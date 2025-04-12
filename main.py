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
    
    # Load the image (e.g., squirrel.png)
    img = cv2.imread('photos/AngularSquirrel2.png')
    
    P = Processor(img, rows=1, cols=1)
    testingData = []
    P.process_image(testingData=testingData)

    for sub_img in testingData:
        P.show_img(sub_img)
        
if __name__ == '__main__':
    main(sys.argv) 