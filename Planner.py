'''
    Essentially acts as the distributor that assigns tasks to artists (bots).

    - Delegates a list of coordinates (representing one image chunk) to an artist.
    - Maintains a queue of tasks received from the Processor.
    - Continuously checks for available artists and assigns them the next task in queue.
'''

import queue
from threading import Lock, Condition
import socket
import threading
import math

class Planner:
    
    def __init__(self, num_artists: int):
        self.__num_artists = num_artists
        
        # __assigned_task[i] tracks whether the ith artist currently has a task
        self.__assigned_task = [False] * self.__num_artists
        
        # IP and port settings for accepting connections from artists
        ARTIST_IP = "0.0.0.0"   # Accept connections from any IP
        ARTIST_PORT = 22        # NOTE: Needs to use a non-privileged port in production

        # SD TODO::comment below code (connection code) for wanting to look at images after they have been processed
        # Create and bind server socket for artists to connect
        # self.sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.sever_socket.bind((ARTIST_IP, ARTIST_PORT))
        # self.sever_socket.listen(self.__num_artists)

        # # Accept connections from all expected artists
        # self.connections = [self.sever_socket.accept() for _ in range(self.__num_artists)]

        # # Thread-safe queue for storing paths to distribute
        # self.__queue = queue.Queue()

        # # Lock and condition variable for task distribution
        # self.distributeTaskLock = Lock()
        # self.dataAvailable = Condition(self.distributeTaskLock)

        # # Start background thread to listen for and distribute tasks
        # thread = threading.Thread(target=self.DistributeTask)
        # thread.start()

    def AddTaskToQueue(self, path):
        '''
        Adds a new path (list of coordinates) to the distribution queue.
        Notifies the distributor thread that a new task is available.
        '''
        with self.distributeTaskLock:
            self.__queue.put(path)
            self.dataAvailable.notify()  # Wake up distributor thread if waiting

    def DistributeTask(self):
        '''
        Continuously distributes tasks to available artists.

        - Waits for new tasks to be available in the queue.
        - Sends the task to the first available artist.
        - Marks that artist as busy.
        '''
        try:
            with self.distributeTaskLock:
                # Wait until a task becomes available in the queue
                while self.__queue.empty():
                    self.dataAvailable.wait()
                
                # Get the next task (list of coordinates)
                task_to_distribute = self.__queue.get()

                # SD TODO::used to show the weird dimension of how we represent our coordinate
                first_x_coordinate = task_to_distribute[0, 0, 0]
                print(f"first_x_coordinate: {first_x_coordinate}")

                # Find the first available artist and send the task
                for i in range(self.__num_artists):
                    if not self.__assigned_task[i]:
                        self.__assigned_task[i] = True
                        print(f"Planner::Send task to artist {i}")
                        self.send_messages(task_to_distribute, i)
                        break
                    else: 
                        print("Not distributing task to artist because already assigned")

        except Exception as e:
            print(f"Exception in DistributeTask: {e}")
        
        # Recursively call to continue listening for new tasks
        self.DistributeTask()

    def send_messages(self, msg, artist_index):
        '''
        Sends a serialized numpy array to the specified artist over TCP socket.
        
        Message format:
            - First 1024 bytes: metadata (shape, dtype, data length)
            - Remaining bytes: actual data buffer, padded to 1024 bytes

        Args:
            msg (np.ndarray): Numpy array of path coordinates to send
            artist_index (int): Index of the artist (connection) to send the message to
        '''
        # Convert numpy array to bytes
        message_to_send = msg.tobytes()
        msg_len = str(len(message_to_send))

        # Prepare metadata string including shape, dtype, and length
        metadata_str = str(msg.shape) + ";int64;" + msg_len + ";"
        metadata_to_send = metadata_str.encode('utf-8')

        # Pad metadata to 1024 bytes to ensure fixed header size
        padded_metadata = metadata_to_send.ljust(1024, b'\0')

        # Send metadata first
        self.connections[artist_index][0].sendall(padded_metadata)

        # Pad the array bytes to the next multiple of 1024 for safe transmission
        padded_array = message_to_send.ljust((math.ceil(len(message_to_send) / 1024)) * 1024, b'\0')

        # Send the actual data
        self.connections[artist_index][0].sendall(padded_array)
