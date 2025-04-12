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

def main(args):
    
    P = Processor()
    
    return 

def show_img(img: np.ndarray) -> None:
    """
    Display the given image using OpenCV.

    Args:
    - img (np.ndarray): The image to display.
    """
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 