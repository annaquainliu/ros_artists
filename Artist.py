'''

    One artist (one machine) waits for and receives a list of coordinates 
    in a message from the Planner process.
    
'''

from threading import Thread, Lock, Condition
import socket
import rospy
from geometry_msgs.msg import Twist
import numpy as np
import sys

# Robot movement constants (scaling factors)
DIST_SCALE = 0.015625         # Scaling factor for distance (1/64)
ANGLE_SCALE = 1.45           # Scaling factor for angular movement

class Artist:
    """
    A class that controls the movement of a robot (such as TurtleBot) by publishing velocity commands
    to move it forward, rotate it, and calculate distances and angles. 
    
    
    """
    
    def __init__(self, init_pos):
        '''
            Initializes the Artist class, setting up the ROS node, publisher, and initial robot pose.

            Args:
                init_pos (np.array): Initial position of the robot (x, y) in the environment.
        '''
        self.taskList = []
        self.taskMutex = Lock()
        self.dataAvailable = Condition(self.taskMutex)
        
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
        
        self.producer = Thread(target=self.receive_messages, args=())
        self.producer.daemon = True  # Set the thread as a daemon
        self.producer.start()
        
        self.consumer = Thread(target=self.execute_tasks, args=())
        self.consumer.daemon = True  # Set the thread as a daemon
        self.consumer.start()
        
        while True:
            user_input = raw_input("enter char to quit: ")
            try:
                number = int(user_input)
                print("You entered:", number)
            except ValueError:
                print("Character entered. Quitting program.")
                sys.exit()
                break
        
        
    
    def receive_messages(self):
        '''
            receive_messages will be executed on another thread and 
            continuously listens for tasks and pushes each task onto its taskList.
            
        '''
        PLANNER_IP = "10.5.13.238"
        PLANNER_PORT = 22
        print("ENTER receive tasks")
        
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            print("Artist::Before Connect")
            s.connect((PLANNER_IP, PLANNER_PORT))
            
            while True:
                # print("Artist::Before Meta Data")
                metadata = s.recv(1024)
                
                if len(metadata) == 0:
                    continue
                
                print(metadata)
                print("Artist::Recived data size:: ", len(metadata))
                metadata_str = metadata.decode('utf-8')
        
                # print("Artist::After Meta Data Str: ", metadata_str)
                
                if metadata_str is None:
                    print("Arist::metadata is none")
                    break
                
                shape_str, dtype_str, message_len, _ = metadata_str.split(';')
                message_len = int(message_len)
                print("after split::", shape_str, "::", dtype_str, "::", message_len)
                
                shape = tuple(map(int, shape_str[1:-1].split(',')))
                dtype = np.dtype(dtype_str)
                
                # Receive the data
                data = b''
                curr_message_size = 0
                while curr_message_size < message_len:
                    chunk = s.recv(1024)
                    if not chunk:
                        break  # Connection closed or no more data
                    data += chunk
                    curr_message_size += len(chunk)
                   
                print("Artist::Receive data from Planner")
                
                # bytes_np_dec = data.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
                print("Artist::Before from buffer")
                # Create numpy array from received data
                task = np.frombuffer(data[:int(message_len)], dtype=dtype).reshape(shape)
                print("Received NumPy array:\n", task)
                
                with self.taskMutex:
                    self.taskList.append(task)
                    self.dataAvailable.notify()
                print("Data recieved through socket: ", data)
        except Exception as e:
            print("Artist::Receive Message::Exception:: ", e)
            pass
        finally:
            s.close()
    
    def execute_tasks(self):
        '''
            execute_task will wait for tasks on the queue and execute
            the closest task to its current position.
            
        '''
        print("ENTER execute tasks")
        with self.taskMutex:
            
            while (len(self.taskList) == 0):
                self.dataAvailable.wait()
            print("Artist::Execute new chunks")
                
            # find closest task to execute
            closest_task = self.taskList[0]
            curr_dist = self.GetEuclidianDistance(closest_task[0, 0], self.curr_pos)
            
            for idx, task in enumerate(self.taskList[1:]):
                dist = self.GetEuclidianDistance(self.curr_pos, task[0, 0])
                if dist < curr_dist: 
                    closest_task = task
                    curr_dist = dist
            
            self.taskList.remove(closest_task)
            
        # Move the robots in the order of the path for the task
        for next_coord in closest_task:
            self.Move(next_coord)
        
        self.execute_tasks()
    
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
        print("Artist::GetEuc::coor_fin::", coords_final, "::Type::", type(coords_final))
        print("Artist::GetEuc::coor_init::", coords_init, "::Type::", type(coords_init))
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
            

def main(args):
    init_x = int(args[1])
    init_y = int(args[2])
    
    
    artist = Artist((init_x, init_y))

if __name__ == '__main__':
    main(sys.argv)