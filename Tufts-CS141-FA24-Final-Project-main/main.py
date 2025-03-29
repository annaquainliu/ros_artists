"""
File: processor.py
Authors: Sophonni Dy, Sam Youkeles
Date: 12-19-24

Description:
    This script performs a series of image processing and path planning operations, including:
        1. Loading and preprocessing an image (grayscale conversion, contrast adjustment, Gaussian blur).
        2. Performing edge detection and extracting contours from the image.
        3. Filtering, approximating, and path planning through the contours.
        4. Evaluating the effect of various processing thresholds on image complexity and accuracy.
        5. Visualizing the results through plots and displaying images using OpenCV.

Usage:
    1. Ensure all dependencies are installed (e.g., NumPy, OpenCV, Matplotlib, scikit-image).
    2. Place the input image (e.g., 'trumpet.jpeg') in the same directory.
    3. Run the script to generate processed images and analysis plots.
        
Dependencies:
    - Processor.py (custom module for image processing)
    - Planner.py (custom module for path planning)
    - NumPy (for numerical operations)
    - OpenCV (for image manipulation and display)
    - Matplotlib (for graphing results)
    - scikit-image (for Structural Similarity Index (SSIM) computation)
"""

from Processor import Processor
from Planner import Planner
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def main():
    """
    Main function that loads an image, processes it, performs contour-based path planning, 
    and generates plots to visualize the effects of different processing thresholds.
    """

    # Load the image (e.g., trumpet.png)
    img = cv2.imread('trumpet.png')

    # Get image dimensions (height, width, channels) and create a blank canvas
    height, width, channels = img.shape
    canvas = np.ones((height, width, channels)) * 255  # White canvas

    # Convert the image to grayscale
    grayscale_img = Processor.ColorToGrayscale(img)
    show_img(grayscale_img)

    # Adjust contrast and brightness of the grayscale image
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    show_img(adjusted_img)

    # Apply Gaussian blur to reduce noise
    blurred_img = Processor.GaussianBlur(adjusted_img)
    show_img(blurred_img)

    # Perform edge detection using Canny edge detection
    edge_img = Processor.EdgeDetection(blurred_img)
    show_img(edge_img)

    # Extract contours from the edge-detected image
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)
    approx_contours = Processor.ApproxContours(filtered_contours, 5)

    # Plan the path through the approximated contours
    path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))

    # Perform various threshold tests for complexity and accuracy
    low_bound = 0
    up_bound = 10
    num_points = 100
    keys, values = DpThresholdToComplexity(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Contour Approximation Threshold', 'Number of Points', 'Effect of Contour Approximation on Image Complexity')

    keys, values = DpThresholdToAccuracy(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Contour Approximation Threshold', "Accuracy", 'Effect of Contour Approximation on Image Accuracy')

    low_bound = 0
    up_bound = 1000
    num_points = 50
    keys, values = EdgeDetectionThresholdToComplexity(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Edge Detection Threshold', 'Number of Points', 'Effect of Edge Detection Threshold on Image Complexity')

    keys, values = EdgeDetectionThresholdToAccuracy(img, low_bound, up_bound, num_points)
    plot_graph_without_dots(keys, values, 'Edge Detection Threshold', "Accuracy", 'Effect of Edge Detection Threshold on Image Accuracy')

    keys, values = FilterContourThresholdToComplexity(img)
    plot_graph_without_dots(keys, values, 'Contour Similarity Threshold', 'Number of Points', 'Effect of Contour Filtering on Image Complexity', 'log')

    keys, values = FilterContourThresholdToAccuracy(img)
    plot_graph_without_dots(keys, values, 'Contour Similarity Threshold', 'Accuracy', 'Effect of Contour Filtering on Image Accuracy', 'log')


def show_img(img: np.ndarray) -> None:
    """
    Display the given image using OpenCV.

    Args:
    - img (np.ndarray): The image to display.
    """
    cv2.imshow(f'Image (Press 0 to Exit)', img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def plot_graph_without_dots(x, y, x_label_str, y_label_str, title_str, scale=None):
    """
    Plot a graph without any dots on the line.

    Args:
    - x (list or np.ndarray): X-axis values
    - y (list or np.ndarray): Y-axis values
    - x_label_str (str): Label for the x-axis
    - y_label_str (str): Label for the y-axis
    - title_str (str): Title of the graph
    - scale (str, optional): The scale for the x-axis ('log' or 'linear')
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linestyle='-', color='b')  # Line without dots

    if scale:
        plt.xscale(scale)  # Apply logarithmic scale if specified

    # Set the title and axis labels
    plt.title(title_str)
    plt.xlabel(x_label_str)
    plt.ylabel(y_label_str)

    plt.grid(True)
    plt.show()


def DpThresholdToComplexity(img, low_bound, up_bound, num_points):
    """
    Compute the effect of different contour approximation thresholds on image complexity.

    Args:
    - img (np.ndarray): The input image.
    - low_bound (float): The lower bound of the threshold range.
    - up_bound (float): The upper bound of the threshold range.
    - num_points (int): The number of threshold values to test.

    Returns:
    - (list, list): The threshold values and the corresponding complexity values.
    """
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)

    points = np.linspace(low_bound, up_bound, num_points)
    threshold_to_num_points = {}
    for p in points:
        approx_contours = Processor.ApproxContours(filtered_contours, p)
        threshold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threshold_to_num_points.keys())
    values_array = list(threshold_to_num_points.values())
    return keys_array, values_array


def EdgeDetectionThresholdToComplexity(img, low_bound, up_bound, num_points):
    """
    Compute the effect of different edge detection thresholds on image complexity.

    Args:
    - img (np.ndarray): The input image.
    - low_bound (float): The lower bound of the threshold range.
    - up_bound (float): The upper bound of the threshold range.
    - num_points (int): The number of threshold values to test.

    Returns:
    - (list, list): The threshold values and the corresponding complexity values.
    """
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)

    points = np.linspace(low_bound, up_bound, num_points)
    threshold_to_num_points = {}
    for p in points:
        edge_img = Processor.EdgeDetection(blurred_img, p)
        contours = Processor.GetContours(edge_img)
        filtered_contours = Processor.FilterContours(contours, 0.5, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        threshold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threshold_to_num_points.keys())
    values_array = list(threshold_to_num_points.values())
    return keys_array, values_array


def FilterContourThresholdToComplexity(img):
    """
    Compute the effect of contour filtering thresholds on image complexity.

    Args:
    - img (np.ndarray): The input image.

    Returns:
    - (list, list): The threshold values and the corresponding complexity values.
    """
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)

    points = np.array([0.001, 0.01, 0.1, 1, 10])
    threshold_to_num_points = {}
    for p in points:
        filtered_contours = Processor.FilterContours(contours, p, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        threshold_to_num_points[p] = GetNumPoints(approx_contours)

    keys_array = list(threshold_to_num_points.keys())
    values_array = list(threshold_to_num_points.values())
    return keys_array, values_array


def DpThresholdToAccuracy(img, low_bound, up_bound, num_points):
    """
    Compute the effect of different contour approximation thresholds on image accuracy.

    Args:
    - img (np.ndarray): The input image.
    - low_bound (float): The lower bound of the threshold range.
    - up_bound (float): The upper bound of the threshold range.
    - num_points (int): The number of threshold values to test.

    Returns:
    - (list, list): The threshold values and the corresponding accuracy values.
    """
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)
    filtered_contours = Processor.FilterContours(contours, 0.5, 10)

    points = np.linspace(low_bound, up_bound, num_points)
    y = []
    for p in points:
        approx_contours = Processor.ApproxContours(filtered_contours, p)
        path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
        canvas = DrawPath(img, path)
        score, _ = ssim(edge_img, canvas, full=True)  # Compute similarity score
        y.append(score)

    return points, y


def EdgeDetectionThresholdToAccuracy(img, low_bound, up_bound, num_points):
    """
    Compute the effect of different edge detection thresholds on image accuracy.

    Args:
    - img (np.ndarray): The input image.
    - low_bound (float): The lower bound of the threshold range.
    - up_bound (float): The upper bound of the threshold range.
    - num_points (int): The number of threshold values to test.

    Returns:
    - (list, list): The threshold values and the corresponding accuracy values.
    """
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)

    points = np.linspace(low_bound, up_bound, num_points)
    y = []
    for p in points:
        edge_img = Processor.EdgeDetection(blurred_img, p)
        contours = Processor.GetContours(edge_img)
        filtered_contours = Processor.FilterContours(contours, 0.5, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, 5)
        path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
        canvas = DrawPath(img, path)
        score, _ = ssim(edge_img, canvas, full=True)
        y.append(score)

    return points, y


def FilterContourThresholdToAccuracy(img):
    """
    Compute the effect of contour filtering thresholds on image accuracy.

    Args:
    - img (np.ndarray): The input image.

    Returns:
    - (list, list): The threshold values and the corresponding accuracy values.
    """
    grayscale_img = Processor.ColorToGrayscale(img)
    adjusted_img = Processor.AdjustContrast(grayscale_img)
    blurred_img = Processor.GaussianBlur(adjusted_img)
    edge_img = Processor.EdgeDetection(blurred_img)
    contours = Processor.GetContours(edge_img)

    points = np.array([0.001, 0.01, 0.1, 1, 10])
    y = []
    for p in points:
        filtered_contours = Processor.FilterContours(contours, p, 10)
        approx_contours = Processor.ApproxContours(filtered_contours, p)
        path = Planner.PathPlan(approx_contours, np.array([[0, 0]]))
        canvas = DrawPath(img, path)
        score, _ = ssim(edge_img, canvas, full=True)
        y.append(score)

    return points, y


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


def GetNumPoints(contours):
    """
    Count the total number of points in all contours.

    Args:
    - contours (list): A list of contours, where each contour is a list of points.

    Returns:
    - int: The total number of points in the contours.
    """
    num_points = 0
    for contour in contours:
        num_points += len(contour)  # Count points in each contour

    return num_points


if __name__ == '__main__':
    main() 