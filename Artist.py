'''
    One artist (robot or machine) waits for and receives a list of coordinates 
    from the Planner process, then physically follows the path by moving 
    through those points.
'''

from threading import Thread, Lock, Condition
import socket
import rospy
from geometry_msgs.msg import Twist
import numpy as np
import sys

# Robot movement constants (scaling factors)
DIST_SCALE = 0.015625  # Scaling factor for linear distance (1/64)
ANGLE_SCALE = 1.45     # Scaling factor for angular velocity

class Artist:
    """
    Represents a robot that listens for coordinate paths and executes them 
    by rotating and moving forward accordingly.
    """

    def __init__(self):
        '''
        Initializes the Artist:
        - Sets up ROS communication and robot control loop.
        - Spawns two threads: one for receiving tasks, one for executing them.
        '''
        self.taskList = []
        self.taskMutex = Lock()
        self.dataAvailable = Condition(self.taskMutex)

        # Initialize ROS node for controlling the robot
        rospy.init_node('turtlebot_artist', anonymous=True)

        # Publisher for sending velocity commands
        self.velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)

        # Rate at which to loop (10 Hz)
        self.rate = rospy.Rate(10)

        # Initialize velocity command message
        self.vel_msg = Twist()

        # Robot's initial position and orientation
        self.curr_pos = (0, 0)
        self.curr_angle = 0

        # Start background thread for receiving coordinate tasks
        self.producer = Thread(target=self.receive_messages)
        self.producer.daemon = True
        self.producer.start()

        # Start background thread for executing coordinate tasks
        self.consumer = Thread(target=self.execute_tasks)
        self.consumer.daemon = True
        self.consumer.start()

        # Simple CLI to allow exit
        while True:
            user_input = raw_input("enter char to quit: ")
            try:
                number = int(user_input)
            except ValueError:
                sys.exit()
                break

    def receive_messages(self):
        '''
        Continuously listens for incoming tasks from the Planner.
        Deserializes the received coordinate path and adds it to the task queue.
        '''
        PLANNER_IP = "10.5.13.238"
        PLANNER_PORT = 22

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((PLANNER_IP, PLANNER_PORT))

            while True:
                # Receive metadata first (includes shape, dtype, and message size)
                metadata = s.recv(1024)
                if len(metadata) == 0:
                    continue

                metadata_str = metadata.decode('utf-8')
                if metadata_str is None:
                    break

                # Extract shape and data type info
                shape_str, dtype_str, message_len, _ = metadata_str.split(';')
                message_len = int(message_len)
                shape = tuple(map(int, shape_str[1:-1].split(',')))
                dtype = np.dtype(dtype_str)

                # Receive raw data based on expected length
                data = b''
                curr_message_size = 0
                while curr_message_size < message_len:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                    curr_message_size += len(chunk)

                # Convert received bytes into a numpy array
                task = np.frombuffer(data[:message_len], dtype=dtype).reshape(shape)

                # Add task to queue
                with self.taskMutex:
                    self.taskList.append(task)
                    self.dataAvailable.notify()

        except Exception as e:
            print("Artist::Receive Message::Exception:: ", e)
            pass
        finally:
            s.close()

    def execute_tasks(self):
        '''
        Waits for available tasks and executes the one closest to the robot's current position.
        Executes tasks by moving through each point in the path.
        '''
        with self.taskMutex:
            while len(self.taskList) == 0:
                self.dataAvailable.wait()

            # Find the task closest to current robot position
            closest_task = self.taskList[0]
            curr_dist = self.GetEuclidianDistance(closest_task[0, 0], self.curr_pos)

            for idx, task in enumerate(self.taskList[1:]):
                dist = self.GetEuclidianDistance(self.curr_pos, task[0, 0])
                if dist < curr_dist:
                    closest_task = task
                    curr_dist = dist

            self.taskList.remove(closest_task)

        # Follow the path in the selected task
        for next_coord in closest_task:
            self.Move(next_coord)

        # Recursively call to process next task
        self.execute_tasks()

    def MoveForward(self, distance):
        """
        Moves the robot forward by a given distance.

        Args:
            distance (float): Distance in meters.
        """
        rospy.loginfo("Info: Move Forward Started")
        self.vel_msg.linear.x = 0.2  # Set forward speed

        t0 = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - t0 < distance * DIST_SCALE:
            self.velocity_publisher.publish(self.vel_msg)
            self.rate.sleep()

        self.StopRobot()
        rospy.loginfo("Info: Move Forward Completed")

    def Rotate(self, angle_radian):
        """
        Rotates the robot by a given angle (in radians).

        Args:
            angle_radian (float): Rotation angle in radians.
        """
        # Normalize angle to [-π, π]
        if angle_radian > np.pi:
            angle_radian = -((2 * np.pi) - angle_radian)
        elif angle_radian < -np.pi:
            angle_radian = (2 * np.pi) + angle_radian

        rospy.loginfo("Info: Rotation Started")
        rospy.loginfo("Info::Rotation::Curr Radian::%s", angle_radian)

        self.vel_msg.angular.z = angle_radian * ANGLE_SCALE

        t0 = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - t0 < 1:
            self.velocity_publisher.publish(self.vel_msg)
            self.rate.sleep()

        self.StopRobot()
        rospy.loginfo("Info: Rotation Completed")

    def StopRobot(self):
        """
        Stops the robot's movement by setting both velocities to zero.
        """
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0
        self.velocity_publisher.publish(self.vel_msg)
        rospy.sleep(1)

    def GetEuclidianDistance(self, coords_init, coords_final):
        """
        Computes Euclidean distance between two points.

        Args:
            coords_init (np.array): Starting point (x, y).
            coords_final (np.array): Ending point (x, y).

        Returns:
            float: Euclidean distance.
        """
        return np.linalg.norm(coords_final - coords_init)

    def GetRotationAngle(self, coords_init, coords_final):
        """
        Computes the angle the robot must rotate to face the destination.

        Args:
            coords_init (np.array): Current position.
            coords_final (np.array): Target position.

        Returns:
            float: Rotation angle in radians.
        """
        diff = coords_final - coords_init
        angle = np.arctan2(diff[0, 1], diff[0, 0])
        return angle - self.curr_angle

    def Move(self, coords_final):
        """
        Rotates and moves the robot to the given target coordinates.

        Args:
            coords_final (np.array): Final destination (x, y).
        """
        distance = self.GetEuclidianDistance(self.curr_pos, coords_final)
        angle = self.GetRotationAngle(self.curr_pos, coords_final)

        self.Rotate(angle)
        self.MoveForward(distance)

        # Update internal robot state
        self.curr_pos = coords_final
        self.curr_angle += angle

def main(args):
    artist = Artist()

if __name__ == '__main__':
    main(sys.argv)
