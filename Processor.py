'''
    The processor spawns various threads to preprocess subimages of the image 
    received. 
    Outputs a list of coordinates for each subimage to the planner to be 
    drawn over
'''

import cv2
from Planner import Planner
import math
import numpy as np
from typing import Tuple
import threading

# Constants for image processing and particle simulation
GAUSS_KERNAL_SIZE = 3         # Size of the Gaussian kernel for blurring
GAUSS_SIGMA = 0               # Standard deviation for Gaussian kernel (0 lets OpenCV auto-calculate)

CONTRAST = 3                  # Multiplier for image contrast
BRIGHTNESS = 50               # Value to increase brightness

class Processor:

    def __init__(self, images: list[np.ndarray]):
        # Initialize with a list of sub-images
        self.__images = images
        num_robots = len(self.__images)
        self.__planner = Planner(num_robots)  # Planner handles the path tasks
        self.__num_threads = num_robots       # One thread per image

    @staticmethod
    def PathPlan(contours: list, init_coord: np.ndarray) -> np.ndarray:
        """
        Plan a path through the contours, starting from a given coordinate.
        """

        path = np.array([init_coord])  # Initialize the path with the start point

        # Keep finding and appending the next closest contour until done
        while len(contours) != 0:
            next_contour = Processor.GetNextContour(init_coord, contours)
            path = np.vstack((path, next_contour))  # Add contour to the path
            init_coord = next_contour[0][0]         # Update current location

        return path

    @staticmethod
    def GetNextContour(coords: np.ndarray, contours: list) -> np.ndarray:
        """
        Find the contour whose starting point is closest to the given coordinate.
        """

        # Start by assuming the first contour's first point is the closest
        closest_distance = np.linalg.norm(contours[0][0] - coords)
        closest_contour_idx = 0
        closest_coords_idx = 0

        # Search for the closest point in all contours
        for ci, c in enumerate(contours):
            for pi, p in enumerate(c):
                distance = np.linalg.norm(p - coords)
                if distance < closest_distance:
                    closest_coords_idx = pi
                    closest_distance = distance
                    closest_contour_idx = ci

        # Reorder selected contour to start from the closest point
        rearranged_contour = np.roll(contours[closest_contour_idx], (-1 * closest_coords_idx), axis=0)

        # Remove the used contour from the list
        del contours[closest_contour_idx]

        # Close the loop by appending the first point again
        return np.append(rearranged_contour, rearranged_contour[0][np.newaxis, ...], axis=0)

    def InitThreadsForProcessing(self, testingData = None):
        '''
        Launch a thread for each sub-image to process it in parallel.
        Each thread will extract path coordinates and send them to the Planner.
        '''

        threads = []

        # Spawn threads to handle each image chunk
        for i in range(len(self.__images)):
            thread = threading.Thread(
                target=self.ProcessImageAndEnqueueTask,
                args=[self.__images[i], testingData]
            )
            threads.append(thread)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()            

    @staticmethod
    def show_img(img: np.ndarray) -> None:
        """
        Display an image using OpenCV GUI window.
        """
        cv2.imshow(f'Image (Press 0 to Exit)', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def DrawPath(img, path):
        """
        Draw the planned path as white lines on a black canvas.
        """

        height, width, _ = img.shape
        canvas = np.zeros((height, width), dtype=np.uint8)  # Blank canvas

        # Draw lines between consecutive points in the path
        for i in range(len(path) - 1):
            curr_x, curr_y = (path[i][0, 0], path[i][0, 1])
            next_x, next_y = (path[i + 1][0, 0], path[i + 1][0, 1])
            cv2.line(canvas, (curr_x, curr_y), (next_x, next_y), 255, thickness=1)

        return canvas

    @staticmethod
    def GetContours(img: np.ndarray) -> list:
        """
        Find contours in a binary image using OpenCV.
        """
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        return contours

    @staticmethod
    def FilterContours(contours: list, threshold: float, distance: int) -> list:
        """
        Filter out similar and nearby duplicate contours based on shape and position.
        """
        filtered_contours = []

        for c1 in contours:
            if cv2.contourArea(c1) > 8:  # Skip tiny contours
                is_duplicate = False
                for c2 in filtered_contours:
                    match_value = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)

                    # Calculate center of mass for both contours
                    M1 = cv2.moments(c1)
                    M2 = cv2.moments(c2)
                    c1X = int(M1["m10"] / M1["m00"])
                    c1Y = int(M1["m01"] / M1["m00"])
                    c2X = int(M2["m10"] / M2["m00"])
                    c2Y = int(M2["m01"] / M2["m00"])

                    # Check both similarity and proximity
                    if match_value < threshold and abs(c2X - c1X) < distance and abs(c2Y - c1Y) < distance:
                        is_duplicate = True
                        break

                # Keep only unique contours
                if not is_duplicate:
                    filtered_contours.append(c1)
        
        return filtered_contours

    @staticmethod
    def ApproxContours(contours: list, threshold: float) -> list:
        """
        Simplify each contour to a polygon using the Ramer-Douglas-Peucker algorithm.
        """
        approx_contours = []

        for c in contours:
            approx_contours.append(cv2.approxPolyDP(c, threshold, True))

        return approx_contours

    def ProcessImageAndEnqueueTask(self, image : np.ndarray, testingData = None):
        '''
        Process a single image to extract path data and send to planner or test output.

        Steps:
            1. Convert to grayscale
            2. Adjust contrast & brightness
            3. Apply Gaussian blur
            4. Detect edges
            5. Find & filter contours
            6. Approximate contours
            7. Plan path
            8. Send to planner or collect output
        '''

        # Convert color image to grayscale
        grayscale_img = Processor.ColorToGrayscale(image)

        # Adjust image brightness and contrast
        adjusted_img = Processor.AdjustContrast(grayscale_img)

        # Blur to reduce noise
        blurred_img = Processor.GaussianBlur(adjusted_img)

        # Use Canny edge detection to find edges
        edge_img = Processor.EdgeDetection(blurred_img)

        # Get all detected contours
        contours = Processor.GetContours(edge_img)

        # Remove duplicate or noisy contours
        filtered_contours = Processor.FilterContours(contours, 0.5, 35)

        # Approximate contours to simpler shapes
        approx_contours = Processor.ApproxContours(filtered_contours, 5)

        # Create a path starting from (0,0)
        init_coord = np.array([[0, 0]])
        path = Processor.PathPlan(approx_contours, init_coord)

        # Either add to planner task queue or store result for testing
        if testingData is None:
            self.__planner.AddTaskToQueue(path)
        else:
            canvas = Processor.DrawPath(image, path)
            testingData.append(canvas)

    # ========== IMAGE PROCESSING UTILITIES BELOW ==========

    @staticmethod
    def ColorToGrayscale(img):
        """
        Convert a color (BGR) image to grayscale.
        """
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayscale_img

    @staticmethod
    def AdjustContrast(img):
        """
        Adjust contrast and brightness of a grayscale image.
        """
        adjusted_img = cv2.convertScaleAbs(img, alpha=CONTRAST, beta=BRIGHTNESS)
        return adjusted_img

    @staticmethod
    def GaussianBlur(img):
        """
        Apply Gaussian blur to smooth image and reduce noise.
        """
        blurred_img = cv2.GaussianBlur(img, (GAUSS_KERNAL_SIZE, GAUSS_KERNAL_SIZE), GAUSS_SIGMA)
        return blurred_img

    @staticmethod
    def EdgeDetection(img):
        """
        Perform Canny edge detection to highlight object boundaries.
        """
        edge_img = cv2.Canny(image=img, threshold1=50, threshold2=150)
        return edge_img
