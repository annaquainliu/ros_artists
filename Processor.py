'''

    The processor spawns various threads to preprocess subimages og the image 
        received. 
    Ouputs a list of coordinates for each subimage to the planner to be 
        drawn over
    
'''
import cv2
from Planner import Planner
import math
import numpy as np
from typing import Tuple
import threading

NUM_ROBOTS = 2

# Constants for image processing and particle simulation
GAUSS_KERNAL_SIZE = 3         # Kernel size for Gaussian blur
GAUSS_SIGMA = 0               # Sigma value for Gaussian blur

CONTRAST = 3                  # Contrast adjustment factor
BRIGHTNESS = 50               # Brightness adjustment factor

class Processor:

    def __init__(self, image: np.ndarray, rows=4, cols=5):
        self.__image = image
        h, w, _ = self.__image.shape
        self.__planner = Planner(w, h)
        self.__num_rows = rows
        self.__num_cols = cols
        self.__num_threads = self.__num_rows * self.__num_cols
    
    
    @staticmethod
    def PathPlan(contours: list, init_coord: np.ndarray) -> np.ndarray:
        """
        Plan a path that visits points in a series of contours starting from an initial coordinate.

        Args:
        - contours (list): A list of contours, where each contour is a list of points.
        - init_coord (np.ndarray): The initial starting coordinate (as a 2D array).

        Returns:
        - np.ndarray: The planned path as a series of coordinates in the order they should be visited.
        """
        # Initialize the path with the starting coordinate
        path = np.array([init_coord])

        # While there are still contours to process, keep adding the closest point to the path
        while len(contours) != 0:
            # Find the next contour point closest to the current coordinate
            next_contour = Processor.GetNextContour(init_coord, contours)
            
            # Append the next contour point to the path
            path = np.vstack((path, next_contour))

            # Update the initial coordinate to the first point of the newly found contour
            init_coord = next_contour[0][0]

        return path

    @staticmethod
    def GetNextContour(coords: np.ndarray, contours: list) -> np.ndarray:
        """
        Find the closest point from the list of contours to the given coordinate.

        Args:
        - coords (np.ndarray): The current coordinate to find the nearest point from.
        - contours (list): A list of contours, where each contour is a list of points.

        Returns:
        - np.ndarray: The closest contour after rearranging so that the starting point is closest to the current coordinate.
        """
        # Initialize variables to track the closest distance and contour indices
        closest_distance = np.linalg.norm(contours[0][0] - coords)  # Start with the first point in the first contour
        closest_contour_idx = 0  # Index of the closest contour
        closest_coords_idx = 0   # Index of the closest point in the closest contour

        # Iterate over all contours and all points within each contour
        for ci, c in enumerate(contours):
            for pi, p in enumerate(c):
                # Calculate the Euclidean distance from the current coordinate to the point in the contour
                distance = np.linalg.norm(p - coords)

                # Update the closest contour if a shorter distance is found
                if distance < closest_distance:
                    closest_coords_idx = pi
                    closest_distance = distance
                    closest_contour_idx = ci

        # Rearrange the selected contour so that its closest point is at the start
        rearranged_contour = np.roll(contours[closest_contour_idx], (-1 * closest_coords_idx), axis=0)

        # Remove the selected contour from the list of contours
        del contours[closest_contour_idx]

        # Append the first point of the contour to the end of the rearranged contour to close the loop
        return np.append(rearranged_contour, rearranged_contour[0][np.newaxis, ...], axis=0)
    
    def process_image(self, testingData = None):
        '''
            Spawn self.__num_threads of threads to process each chunk of the self.__image
            
            Each thread then communicates to the Planner object to add the list of coordinates
            onto the queue.
        '''
        subimage_range_list = self.split_image()
        threads = []
        
        # Assumption: # thread = # sub-image chunks
        for i in range(len(subimage_range_list)):
            thread = threading.Thread(target=self.process_chunk,args=[subimage_range_list[i], testingData])
            threads.append(thread)
        
        # Start threads
        for t in threads:
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()            
    
    @staticmethod
    def show_img(img: np.ndarray) -> None:
        """
        Display the given image using OpenCV.

        Args:
        - img (np.ndarray): The image to display.
        """
        cv2.imshow(f'Image (Press 0 to Exit)', img)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
        
    @staticmethod
    def DrawPath(img, path):
        """
        Draw the path on the canvas.

        Args:
        - img (np.ndarray): The input image.
        - path (list): A list of points representing the planned path.

        Returns:
        - np.ndarray: The canvas with the drawn path.
        """
        height, width, _ = img.shape
        canvas = np.zeros((height, width), dtype=np.uint8)

        # Draw lines between consecutive path points
        for i in range(len(path) - 1):
            curr_x, curr_y = (path[i][0, 0], path[i][0, 1])
            next_x, next_y = (path[i + 1][0, 0], path[i + 1][0, 1])
            
            thickness = 1  # Line thickness
            cv2.line(canvas, (curr_x, curr_y), (next_x, next_y), 255, thickness)

        return canvas
    
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
    
    def process_chunk(self, chunk_image_range 
                            : Tuple[Tuple[int, int], Tuple[int, int]], 
                            testingData = None):
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
        ((left, top), (right, bottom)) = chunk_image_range
        
        # For the specified chunk between the columns [left, right] 
        # and the rows [top, bottom] create an image chunk
        image_chunk = self.__image[top : bottom + 1, left : right + 1, :]
        
        print(f"image dimensions: {self.__image.shape}")
        print(f"image chunk shape: {image_chunk.shape}")
        
        # Convert the image to grayscale
        grayscale_img = Processor.ColorToGrayscale(image_chunk)
        
        print(f"Grayscale image shape: {grayscale_img.shape}")
        
        # Adjust the contrast and brightness of the grayscale image
        adjusted_img = Processor.AdjustContrast(grayscale_img)
        
         # Apply Gaussian blur to reduce noise in the image
        blurred_img = Processor.GaussianBlur(adjusted_img)
        
         # Detect edges in the blurred image using Canny edge detection
        edge_img = Processor.EdgeDetection(blurred_img)

        # Find the contours in the edge-detected image
        contours = Processor.GetContours(edge_img)
        
         # Filter the contours based on shape similarity and distance
        filtered_contours = Processor.FilterContours(contours, 0.5, 35)

        # Approximate the contours to reduce the number of points per contour
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        
        # Plan a path through the approximated contours starting from the initial position
        print("approx_contours: ", approx_contours)
        # print("left, right", left, top)
        path = Processor.PathPlan(approx_contours, np.array([[left, top]]))
        
        if testingData is None:
            self.planner.AddTaskToQueue(path)
        else:
            # Add the path to the queue
            canvas = Processor.DrawPath(image_chunk, path)
            testingData.append(canvas)

    
    def split_image(self):
        '''
        Split each full image into subimages 
        Input: an image object (cv2 image)
        Ouput: A list of image ranges
        [(top_left_coordinate, bottom_right_coordinate)]
        '''
        img_height, img_width, _ = self.__image.shape
        
        section_width = math.ceil(img_width / self.__num_cols)
        section_height = math.ceil(img_height / self.__num_rows)
        
        subimage_range_list = []
        
        for x in range(self.__num_cols):
            for y in range(self.__num_rows):
                left_x = x * section_width
                left_y = y * section_height

                right_x = min(left_x + section_width, img_width)
                right_y = min(left_y + section_height, img_height)

                subimage_range_list.append(((left_x, left_y), (right_x, right_y)))
                # print(subimage_range_list[-1])
        
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

    