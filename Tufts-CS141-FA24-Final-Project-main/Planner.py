"""
File: planner.py
Authors: Sophonni Dy, Sam Youkeles
Date: 12-29-24

Description:
    This file contains the `Planner` class which is responsible for path planning
    based on contours extracted from an image. The `PathPlan` function generates a 
    sequence of coordinates that outlines a path starting from an initial position.
    The `GetNextContour` function helps find the closest contour to the current 
    position and rearranges it such that the closest point is at the start.
    
    The main purpose of this class is to process contours and generate a navigable 
    path for a robot or other agent. This can be used in robotics, computer vision, 
    and similar fields where path planning is needed.

Usage:
    1. `PathPlan` requires a list of contours (each contour being a sequence of points)
       and an initial coordinate. It will return a path as a series of coordinates.
    2. `GetNextContour` helps find and rearrange the next closest contour to the current position.

Dependencies:
    - numpy: For handling coordinate arrays and calculations.

Example:
    contours = [...]  # A list of contours (each a list of points)
    initial_coord = np.array([[0, 0]])  # Starting position
    path = Planner.PathPlan(contours, initial_coord)
"""

import numpy as np

class Planner:
    """
    A class to plan a path based on a list of contours and an initial starting coordinate.
    The path is generated by selecting the nearest point from the list of contours at each step.
    """

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
            next_contour = Planner.GetNextContour(init_coord, contours)

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