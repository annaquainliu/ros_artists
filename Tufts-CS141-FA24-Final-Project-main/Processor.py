"""
File: processor.py
Authors: Sophonni Dy, Sam Youkeles
Date: 12-19-24

Description:
    This file contains the `Processor` class, which includes various image 
    processing methods such as color-to-grayscale conversion, contrast adjustment, 
    Gaussian blurring, edge detection, and contour analysis. These methods can 
    be used sequentially to prepare an image for further processing in computer vision 
    tasks, such as object detection and path planning.
    
    The `Processor` class provides static methods that operate on images (as numpy arrays), 
    making it easy to chain operations like contrast adjustment, Gaussian blurring, 
    and edge detection to extract contours or features from an image.

Usage:
    1. `ColorToGrayscale`: Converts a BGR image to grayscale.
    2. `AdjustContrast`: Adjusts the contrast and brightness of an image.
    3. `GaussianBlur`: Applies a Gaussian blur to the image to reduce noise.
    4. `EdgeDetection`: Detects edges in the image using the Canny edge detector.
    5. `GetContours`: Finds contours in a binary image.
    6. `FilterContours`: Filters out duplicate contours based on shape similarity and distance.
    7. `ApproxContours`: Approximates the contours to polygons based on the specified threshold.

Dependencies:
    - numpy: Used for handling images as numpy arrays.
    - cv2: OpenCV library used for image processing tasks.

Example:
    img = cv2.imread('image.jpg')
    grayscale_img = Processor.ColorToGrayscale(img)
    contrast_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(contrast_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)
    approx_contours = Processor.ApproxContours(filtered_contours, 5)
"""

import cv2
import numpy as np

# Constants for image processing
GAUSS_KERNAL_SIZE = 3  # Gaussian kernel size for blurring
GAUSS_SIGMA = 0        # Standard deviation for Gaussian kernel

CONTRAST = 3            # Contrast adjustment factor
BRIGHTNESS = 50         # Brightness adjustment value

# Particle info (used for visualization or path representation)
RED = 180
GREEN = 105
BLUE = 255
PARTICLE_THICKNESS = 2
PARTICLE_RADIUS = 1

# Scale factors for distance and angle
DIST_SCALE = 0.03125
ANGLE_SCALE = 1.45

class Processor:
    """
    A collection of image processing methods used for contour detection, 
    edge detection, contrast adjustment, and Gaussian blurring. These methods
    can be applied to images in a step-by-step manner to extract features 
    and perform image manipulations.
    """
    
    @staticmethod
    def ColorToGrayscale(img: np.ndarray) -> np.ndarray:
        """
        Convert an input image to grayscale.

        Args:
        - img (np.ndarray): The input image in BGR format.

        Returns:
        - np.ndarray: The grayscale version of the input image.
        """
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img
    
    @staticmethod
    def AdjustContrast(img: np.ndarray) -> np.ndarray:
        """
        Adjust the contrast and brightness of an image.

        Args:
        - img (np.ndarray): The input image in grayscale or color format.

        Returns:
        - np.ndarray: The contrast-adjusted image.
        """
        adjusted_img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=BRIGHTNESS)
        return adjusted_img
    
    @staticmethod
    def GaussianBlur(img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to the image to reduce noise and details.

        Args:
        - img (np.ndarray): The input image.

        Returns:
        - np.ndarray: The blurred image.
        """
        blurred_img = cv2.GaussianBlur(img, (GAUSS_KERNAL_SIZE, GAUSS_KERNAL_SIZE), GAUSS_SIGMA)
        return blurred_img
    
    @staticmethod
    def EdgeDetection(img: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
        """
        Detect edges in an image using the Canny edge detection method.

        Args:
        - img (np.ndarray): The input image (grayscale).
        - t1 (int, optional): The lower threshold for the Canny edge detection. Default is 50.
        - t2 (int, optional): The upper threshold for the Canny edge detection. Default is 150.

        Returns:
        - np.ndarray: The binary image representing edges.
        """
        edge_img = cv2.Canny(img, t1, t2)
        return edge_img
    
    @staticmethod
    def GetContours(img: np.ndarray) -> list:
        """
        Find contours in the input binary image.

        Args:
        - img (np.ndarray): The input binary image.

        Returns:
        - list: A list of contours found in the image. Each contour is represented as an array of points.
        """
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        return contours
    
    @staticmethod
    def FilterContours(contours: list, threshold: float, distance: int) -> list:
        """
        Filter out duplicate contours based on shape similarity and distance.

        Args:
        - contours (list): A list of contours to filter.
        - threshold (float): The maximum match value for contour similarity (lower is more similar).
        - distance (int): The maximum distance between contour centers for considering them duplicates.

        Returns:
        - list: A list of filtered contours, with duplicates removed.
        """
        filtered_contours = []

        # Iterate through all contours and filter out duplicates
        for c1 in contours:
            if cv2.contourArea(c1) > 8:  # Ignore small contours with area < 8
                is_duplicate = False
                for c2 in filtered_contours:
                    # Calculate shape similarity using matchShapes (lower values indicate higher similarity)
                    match_value = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)

                    # Get the center (moments) of both contours for distance comparison
                    M1 = cv2.moments(c1)
                    M2 = cv2.moments(c2)
                    c1X = int(M1["m10"] / M1["m00"])
                    c1Y = int(M1["m01"] / M1["m00"])
                    c2X = int(M2["m10"] / M2["m00"])
                    c2Y = int(M2["m01"] / M2["m00"])

                    # Check if contours are similar enough and close to each other
                    if match_value < threshold and abs(c2X - c1X) < distance and abs(c2Y - c1Y) < distance:
                        is_duplicate = True
                        break
                # If no duplicate is found, add the contour to the result list
                if not is_duplicate:
                    filtered_contours.append(c1)
        
        return filtered_contours
        
    @staticmethod
    def ApproxContours(contours: list, threshold: float) -> list:
        """
        Approximate each contour to a polygon using the specified threshold.

        Args:
        - contours (list): A list of contours to approximate.
        - threshold (float): The approximation accuracy. Larger values result in fewer vertices.

        Returns:
        - list: A list of approximated contours, where each contour is a polygon (array of points).
        """
        approx_contours = []

        # Approximate each contour to a polygon
        for c in contours:
            approx_contours.append(cv2.approxPolyDP(c, threshold, True))
            
        return approx_contours
