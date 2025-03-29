#!/user/bin/env python

"""
File: turtlebot_artist.py
Authors: Sophonni Dy, Sam Youkeles
Date: [Date]

Description:
    This script is designed to control a TurtleBot (or any similar robot) by combining image processing, 
    path planning, and ROS-based movement control. It processes an image to extract contours, plans a 
    path through those contours, and then uses the TurtleBot to follow the path based on odometry and 
    velocity commands.

    The script leverages several classes for different tasks:
    1. `Processor`: Performs image processing tasks such as converting to grayscale, applying Gaussian blur,
       detecting edges, finding and filtering contours.
    2. `Planner`: Plans a navigable path through a series of contours, starting from an initial position.
    3. `Odometer`: Handles odometry data from the `/odom` ROS topic to track the robot’s pose in real-time.
    4. `Artist`: Controls the movement of the robot by publishing velocity commands, and uses the `Odometer` 
       to track the robot's position and orientation as it moves.

    The main function of the script loads an image, processes it, generates a path from the contours, 
    and drives the robot along that path using ROS.

Usage:
    1. ROS setup should be complete, with the TurtleBot connected and running.
    2. The script starts by initializing the robot's position, then loads and processes an image.
    3. The robot's path is computed based on the contours found in the image, and it moves accordingly.

Dependencies:
    - rospy: For communication with ROS and publishing velocity commands.
    - numpy: For handling coordinate data and movement calculations.
    - opencv-python (cv2): For image processing tasks like edge detection, contour finding, and filtering.
    - math: For mathematical calculations (such as calculating Euclidean distance and angles).
    - threading: To handle the odometer listener in a separate thread.

Example:
    To run this script:
    1. Ensure the robot is powered on and connected via ROS.
    2. Place the script in the ROS workspace and run it to control the TurtleBot to navigate along the path.
    3. The robot will follow the contours of the image, moving and rotating as necessary.

Note:
    This script assumes that the ROS topic `/odom` is available for odometry data and that the TurtleBot is 
    set up to listen to the `/cmd_vel_mux/input/navi` topic for velocity commands.
"""


import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from threading import Thread, Event

import numpy as np
import math
import cv2

####################################################################################################################################
# Constants for image processing and particle simulation
GAUSS_KERNAL_SIZE = 3         # Kernel size for Gaussian blur
GAUSS_SIGMA = 0               # Sigma value for Gaussian blur

CONTRAST = 3                  # Contrast adjustment factor
BRIGHTNESS = 50               # Brightness adjustment factor

# Particle visualization constants
RED = 180
GREEN = 105
BLUE = 255
PARTICLE_THICKNESS = 2       # Particle outline thickness
PARTICLE_RADIUS = 1          # Particle radius

# Robot movement constants (scaling factors)
DIST_SCALE = 0.015625         # Scaling factor for distance (1/64)
ANGLE_SCALE = 1.45           # Scaling factor for angular movement

class Processor:
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
    
    @staticmethod
    def GetContours(img):
        """
        Find and return the contours in the given image.

        Args:
            img (np.array): Input image to extract contours from.
        
        Returns:
            list: List of contours found in the image.
        """
        _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def FilterContours(contours, threshold, distance):
        """
        Filter out duplicate contours based on shape matching and proximity.

        Args:
            contours (list): List of contours to filter.
            threshold (float): Threshold for shape matching (lower is stricter).
            distance (int): Maximum allowed distance between contours for them to be considered duplicates.
        
        Returns:
            list: Filtered list of contours with duplicates removed.
        """
        filtered_contours = []

        # Iterate through each contour and check for duplicates
        for c1 in contours:
            if cv2.contourArea(c1) > 8:  # Only consider contours with area greater than 8
                is_duplicate = False
                for c2 in filtered_contours:
                    # Compare the shape of contour c1 with contour c2 using matchShapes
                    match_value = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
                    
                    # Calculate the centroids of both contours
                    M1 = cv2.moments(c1)
                    M2 = cv2.moments(c2)
                    c1X = int(M1["m10"] / M1["m00"])
                    c1Y = int(M1["m01"] / M1["m00"])
                    c2X = int(M2["m10"] / M2["m00"])
                    c2Y = int(M2["m01"] / M2["m00"])

                    # Check if the contours are similar in shape and close in position
                    if match_value < threshold and abs(c2X - c1X) < distance and abs(c2Y - c1Y) < distance:
                        is_duplicate = True
                        break
                
                # If no duplicate is found, add contour to the filtered list
                if not is_duplicate:
                    filtered_contours.append(c1)
        
        return filtered_contours
    
    @staticmethod
    def ApproxContours(contours, threshold):
        """
        Approximate each contour to a polygon with fewer vertices using the Ramer-Douglas-Peucker algorithm.

        Args:
            contours (list): List of contours to approximate.
            threshold (float): The approximation accuracy. Lower values result in more vertices.
        
        Returns:
            list: List of approximated contours.
        """
        approx_contours = []

        # Approximate each contour in the list
        for c in contours:
            approx_contours.append(cv2.approxPolyDP(c, threshold, True))
        
        return approx_contours
####################################################################################################################################
class Planner:
    """
    A class for path planning, which determines the optimal path through a series of contours,
    starting from an initial coordinate.
    """

    @staticmethod
    def PathPlan(contours, init_coord):
        """
        Plan a path through a list of contours starting from an initial coordinate.

        Args:
            contours (list): List of contours, where each contour is a list of points (np.array).
            init_coord (np.array): The starting coordinate (x, y) for the path planning.
        
        Returns:
            np.array: A 2D array representing the planned path, where each row is a coordinate (x, y).
        """
        # Initialize the path with the starting coordinate
        path = np.array([init_coord])

        # Continue to find the next contour and append it to the path until all contours are processed
        while len(contours) != 0:
            # Get the next contour that is closest to the current position
            next_contour = Planner.GetNextContour(init_coord, contours)

            # Append the next contour to the path
            path = np.vstack((path, next_contour))

            # Update the current position to the first point of the next contour
            init_coord = next_contour[0][0]

        return path

    @staticmethod
    def GetNextContour(coords, contours):
        """
        Find the next contour that is closest to the given coordinate and rearrange the contour's points
        to start from the closest point.

        Args:
            coords (np.array): The current position (x, y).
            contours (list): A list of contours, where each contour is a list of points (np.array).
        
        Returns:
            np.array: A rearranged contour with the closest point to `coords` at the start.
        """
        # Initialize the closest distance with the distance to the first point in the first contour
        closest_distance = np.linalg.norm(contours[0][0] - coords)
        closest_contour_idx = 0
        closest_coords_idx = 0

        # Iterate over each contour and each point within the contour to find the closest one
        for ci, c in enumerate(contours):
            for pi, p in enumerate(c):
                # Calculate the Euclidean distance between the current point and the given coordinates
                distance = np.linalg.norm(p - coords)

                # Update the closest point information if the current point is closer
                if distance < closest_distance:
                    closest_coords_idx = pi
                    closest_distance = distance
                    closest_contour_idx = ci

        # Rearrange the points in the closest contour so that the closest point is at the start
        rearranged_contour = np.roll(contours[closest_contour_idx], (-1 * closest_coords_idx), axis=0)

        # Remove the used contour from the list of contours
        del contours[closest_contour_idx]

        # Return the rearranged contour, appending the first point at the end to close the loop
        return np.append(rearranged_contour, rearranged_contour[0][np.newaxis, ...], axis=0)
####################################################################################################################################
class Odometer:
    """
    A class for handling odometry data from the `/odom` ROS topic.
    It subscribes to the `/odom` topic, listens for messages, and stores the robot's pose.
    """

    def __init__(self):
        """
        Initializes the Odometer class, setting the initial pose and running status.
        
        Attributes:
            pose (Pose): The current pose of the robot (position and orientation).
            running (bool): A flag to control whether the odometer should continue running.
        """
        self.pose = None  # Initialize pose to None (no pose received yet)
        self.running = True  # Flag to indicate if the odometer is running

    def run(self):
        """
        Starts the odometer by subscribing to the `/odom` topic and listening for incoming odometry messages.

        The function loops continuously while the `running` flag is set to `True` and the ROS node is not shut down.
        The loop sleeps for 0.1 seconds to reduce CPU usage.

        The `odom_callback` method will be triggered every time a new Odometry message is received.
        """
        # Subscribe to the '/odom' topic to receive odometry data
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Continuously run the odometer while the node is not shutdown and `running` flag is True
        while self.running and not rospy.is_shutdown():
            rospy.sleep(0.1)  # Sleep briefly to reduce CPU usage

    def odom_callback(self, msg):
        """
        Callback function to handle incoming odometry messages from the `/odom` topic.

        This method is triggered whenever a new `Odometry` message is received. It stores the robot's pose
        from the message in the `pose` attribute.

        Args:
            msg (Odometry): The incoming Odometry message containing position and orientation data.
        """
        # If the odometer is stopped, do not update the pose
        if not self.running:
            return

        # Store the pose from the Odometry message (position and orientation)
        self.pose = msg.pose

    def stop(self):
        """
        Stops the odometer by setting the `running` flag to `False`.

        This will cause the `run` method to stop subscribing to the `/odom` topic and exit the loop.
        """
        self.running = False  # Set the running flag to False to stop the odometer
####################################################################################################################################
class Artist:
    """
    A class that controls the movement of a robot (such as TurtleBot) by publishing velocity commands
    to move it forward, rotate it, and calculate distances and angles. It also uses an Odometer class 
    to track the robot's pose in the environment.
    """

    def __init__(self, init_pos):
        """
        Initializes the Artist class, setting up the ROS node, publisher, and initial robot pose.

        Args:
            init_pos (np.array): Initial position of the robot (x, y) in the environment.
        """
        # Initialize the ROS node for the turtlebot_artist
        rospy.init_node('turtlebot_artist', anonymous=True)

        # Create a publisher to send velocity commands to the robot's navigation system
        self.velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)

        # Set looping rate (10 Hz)
        self.rate = rospy.Rate(10)

        # Initialize the velocity message object to send command velocities
        self.vel_msg = Twist()

        # Set the robot's current position and initial angle
        self.curr_pos = init_pos
        self.curr_angle = 0

        # Create an Odometer object to track the robot's pose
        self.listener = Odometer()

        # Start the odometer listener in a separate thread
        self.listener_thread = Thread(target=self.listener.run)
        self.listener_thread.start()

    def MoveForward(self, distance):
        """
        Move the robot forward by a given distance.

        Args:
            distance (float): The distance to move forward in meters.
        """
        rospy.loginfo("Info: Move Forward Started")

        # Set linear velocity in the x-direction to move forward
        self.vel_msg.linear.x = 0.2

        # Record the start time
        t0 = rospy.Time.now().to_sec()

        # Loop until the robot has moved the specified distance
        while (rospy.Time.now().to_sec() - t0) < distance * DIST_SCALE:
            # Publish the velocity message to move the robot
            self.velocity_publisher.publish(self.vel_msg)

            # Sleep to maintain the desired loop rate
            self.rate.sleep()

        # Stop the robot after moving
        self.StopRobot()

        rospy.loginfo("Info: Move Forward Completed")

    def Rotate(self, angle_radian):
        """
        Rotate the robot by a given angle in radians.

        Args:
            angle_radian (float): The angle to rotate in radians. Positive values rotate counterclockwise.
        """
        # Normalize the angle to be within [-pi, pi]
        if angle_radian > np.pi:
            angle_radian = -((2 * np.pi) - angle_radian)
        elif angle_radian < (-1 * np.pi):
            angle_radian = ((2 * np.pi) + angle_radian)

        rospy.loginfo("Info: Rotation Started")
        rospy.loginfo("Info::Rotation::Curr Radian::%s", angle_radian)

        # Set angular velocity to rotate the robot
        self.vel_msg.angular.z = angle_radian * ANGLE_SCALE

        # Record the start time
        t0 = rospy.Time.now().to_sec()

        # Loop for a fixed duration to rotate the robot
        while (rospy.Time.now().to_sec() - t0) < 1:
            # Publish the velocity message to rotate the robot
            self.velocity_publisher.publish(self.vel_msg)

            # Sleep to maintain the desired loop rate
            self.rate.sleep()

        # Stop the robot after rotation
        self.StopRobot()

        rospy.loginfo("Info: Rotation Completed")

    def StopRobot(self):
        """
        Stop the robot by setting linear and angular velocities to zero.
        """
        # Set both linear and angular velocities to zero
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0

        # Publish the stop command
        self.velocity_publisher.publish(self.vel_msg)

        # Sleep for 1 second to ensure the stop command is executed
        rospy.sleep(1)

    def GetEuclidianDistance(self, coords_init, coords_final):
        """
        Calculate the Euclidean distance between two points.

        Args:
            coords_init (np.array): Initial coordinates (x, y).
            coords_final (np.array): Final coordinates (x, y).
        
        Returns:
            float: The Euclidean distance between the initial and final coordinates.
        """
        # Use numpy's linear algebra norm function to calculate the Euclidean distance
        distance = np.linalg.norm(coords_final - coords_init)
        return distance

    def GetRotationAngle(self, coords_init, coords_final):
        """
        Calculate the angle required to rotate from the initial position to the final position.

        Args:
            coords_init (np.array): Initial coordinates (x, y).
            coords_final (np.array): Final coordinates (x, y).
        
        Returns:
            float: The angle (in radians) that the robot needs to rotate.
        """
        # Compute the difference between the final and initial coordinates
        diff = coords_final - coords_init

        # Calculate the angle using arctan2, which gives the angle between the x-axis and the point (dx, dy)
        angle = np.arctan2(diff[0, 1], diff[0, 0])

        # Return the difference between the desired rotation angle and the current robot orientation
        return angle - self.curr_angle

    def Move(self, coords_final):
        """
        Move the robot to a specified final position by first rotating and then moving forward.

        Args:
            coords_final (np.array): The target coordinates (x, y) to move to.
        """
        # Calculate the distance and rotation angle to the target coordinates
        distance = self.GetEuclidianDistance(self.curr_pos, coords_final)
        angle = self.GetRotationAngle(self.curr_pos, coords_final)

        # Rotate the robot to the desired angle
        self.Rotate(angle)

        # Move the robot forward by the calculated distance
        self.MoveForward(distance)

        # Update the current position and orientation after movement
        self.curr_pos = coords_final
        self.curr_angle += angle

        # Print out control estimated coordinates and current odometer pose for debugging
        print("Control estimated coords:", coords_final)
        print("Odometer pose:", self.listener.pose)
####################################################################################################################################
if __name__ == '__main__':
    """
    Main script to control the TurtleBot artist using image processing and path planning.
    This script loads an image, processes it, generates a path from contours, 
    and makes the robot follow that path.
    """

    # Set the initial position of the robot at the origin (0, 0)
    init_pos = np.array([[0, 0]])

    # Create an instance of the Artist class, which controls the robot
    artist = Artist(init_pos)

    # Load the image 'amg.png' using OpenCV
    img = cv2.imread('amg.png')

    # Extract image dimensions: height, width, and the number of channels
    height, width, channels = img.shape

    # Convert the image to grayscale
    grayscale_img = Processor.ColorToGrayscale(img)

    # Adjust the contrast and brightness of the grayscale image
    adjusted_img = Processor.AdjustContrast(grayscale_img)

    # Apply Gaussian blur to reduce noise in the image
    blurred_img = Processor.GaussianBlur(adjusted_img)

    # Detect edges in the blurred image using Canny edge detection
    edge_img = Processor.EdgeDetection(blurred_img)

    # Find the contours in the edge-detected image
    contours = Processor.GetContours(edge_img)

    # Filter the contours based on shape similarity and distance
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)

    # Approximate the contours to reduce the number of points per contour
    approx_contours = Processor.ApproxContours(filtered_contours, 5)

    # Plan a path through the approximated contours starting from the initial position
    path = Planner.PathPlan(approx_contours, init_pos)

    # Move the robot along the path by making the artist (robot) follow each point
    for p in path:
        artist.Move(p)  # Move to the next point on the path

    # Stop the odometer listener and wait for the listener thread to finish
    artist.listener.stop()
    artist.listener_thread.join()
