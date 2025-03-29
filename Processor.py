'''

    The processor spawns various threads to preprocess subimages og the image 
        received. 
    Ouputs a list of coordinates for each subimage to the planner to be 
        drawn over
        
    
    
'''
import cv2
from Planner import Planner
from PIL import Image
import image
import numpy as np
from typing import Tuple

NUM_ROBOTS = 2

# Constants for image processing and particle simulation
GAUSS_KERNAL_SIZE = 3         # Kernel size for Gaussian blur
GAUSS_SIGMA = 0               # Sigma value for Gaussian blur

CONTRAST = 3                  # Contrast adjustment factor
BRIGHTNESS = 50               # Brightness adjustment factor

class Processor:

    def __init__(self, image: np.ndarray):
        self.__image = image
        self.__planner = Planner()
        self.__num_rows = 4
        self.__num_cols = 5
        self.__num_threads = self.__num_rows * self.__num_cols
    
    
    def process_image(self):
        '''
            Spawn self.__num_threads of threads to process each chunk of the self.__image
            
            Each thread then communicates to the Planner object to add the list of coordinates
            onto the queue.

        '''
    
    def process_chunk(self, chunk_image_range : Tuple[Tuple[int, int], Tuple[int, int]], 
                    planner: Planner, image: np.ndarray):
        '''
            Function ran by a thread spawned from the main Processor thread
            
            Given the image range and the image itself, process the chunk into 
            a list of coordinates.
            
            This list of coordinates, is then sent to the Planner.
            
            Args
            -----
            self : Processor
            chunk_image_range :  Tuple[Tuple[int, int], Tuple[int, int]]
                The range of the image of the chunk in the form of (top_left_coord, bottom_right_coord)
            planner : Planner
            image : np.ndarray
                The full image in an np array
                
            Returns
            -------
            None
            
        '''
        (top_left, bottom_right) = chunk_image_range
        (top_right, bottom_left) = ((bottom_right[0], top_left[1]), (top_left[0], bottom_right[1]))
        
        chunk_width = bottom_right[0] - top_left[0]
        chunk_height = top_left[1] - bottom_right[1]
        
        
        return 
    
    def split_image(self):
       '''
        Split each full image into subimages 
        Input: an image object (cv2 image)
        Ouput: A list of image ranges
        [(top_left_coordinate, bottom_right_coordinate)]
    '''
        img_height, img_width, _ = self.__image.shape

        subimage_range_list = []
        for x in range(self.__num_cols):
            for y in range(self.__num_rows):
                left_x = x * (img_width // self.__num_cols)
                left_y = y * (img_height // self.__num_rows)

                right_x = min((x + 1) * (img_width // self.__num_cols), img_width)
                right_y = min((x + 1) * (img_height // self.__num_rows), img_height)

                subimage_range_list.append(((left_x, left_y), (right_x, right_y)))
        
        return subimage_range_list

    
    

    """
    A class for performing various image processing tasks including
    converting color to grayscale, adjusting contrast, applying Gaussian blur,
    detecting edges, and filtering contours.
    """

    @staticmethod
    def ColorToGrayscale(img):
        """
        Convert the given color image to grayscale.

        Args:
            img (np.array): Input image in BGR format.
        
        Returns:
            np.array: Grayscale version of the input image.
        """
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    
    @staticmethod
    def AdjustContrast(img):
        """
        Adjust the contrast and brightness of the given image.

        Args:
            img (np.array): Input image to adjust.
        
        Returns:
            np.array: Image with adjusted contrast and brightness.
        """
        adjusted_img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=BRIGHTNESS)
        return adjusted_img
    
    @staticmethod
    def GaussianBlur(img):
        """
        Apply Gaussian blur to the input image to reduce noise.

        Args:
            img (np.array): Input image to apply blur on.
        
        Returns:
            np.array: Blurred version of the input image.
        """
        blurred_img = cv2.GaussianBlur(img, (GAUSS_KERNAL_SIZE, GAUSS_KERNAL_SIZE), GAUSS_SIGMA)
        return blurred_img
    
    @staticmethod
    def EdgeDetection(img):
        """
        Perform edge detection on the input image using the Canny edge detector.

        Args:
            img (np.array): Input image for edge detection.
        
        Returns:
            np.array: Image with detected edges.
        """
        edge_img = cv2.Canny(image=img, threshold1=50, threshold2=150)
        return edge_img

    