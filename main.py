'''
    Main.py
    
    Driver file for the program
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
        print("ERROR::main::Command Example::python main.py <image_path_1> <image_path_2> ...")
        sys.exit()
    
    # Load the image (e.g., squirrel.png)
    images = args[1:]
    cv2_images = [cv2.imread(image) for image in images]
    
    # Initialize Processor to perform image processing
    processor = Processor(cv2_images)
    # processor.InitThreadsForProcessing()
    
    # SD TODO::this below code is used for testing
    testingData = []
    processor.InitThreadsForProcessing(testingData=testingData)
    for sub_img in testingData:
        print("INFO::main::image shape:: ", sub_img.shape)
        processor.show_img(sub_img)
        
if __name__ == '__main__':
    main(sys.argv)