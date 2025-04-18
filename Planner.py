

'''
    Essentially act as the distributor that gives task to bots

    Delegates list of coordinates of one chunk of an image to an artist.
    
    Contains a queue of a list of coordinates that are received from the 
    Processor.
    
    While there is an available artist, a list of coordinates is popped
    from the queue and passed as a message to an artist.

'''
import queue
from threading import Lock, Condition
import socket
import threading
import math

class Planner:
    
    
    def __init__(self, img_w, img_h):
        self.__num_artists = 2
        self.__artist_x_boundaries = []
        
        # Each boundary represents the [start_x, end_x)
        for i in range(self.__num_artists):
            self.__artist_x_boundaries.append((i * (img_w // self.__num_artists), (i + 1) * (img_w // self.__num_artists)))
            
        ARTIST_IP = "0.0.0.0"   # accpet connetion from any IP
        ARTIST_PORT = 22     # need to used some non-priviledge port
        
        print("Before Planner Socket Bind")
        self.sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sever_socket.bind((ARTIST_IP, ARTIST_PORT))
        self.sever_socket.listen(self.__num_artists)
        print("After Planner Socket Bind")
        
        # make sure to secure connection with artist
        self.connections = [self.sever_socket.accept() for _ in range(self.__num_artists)]
        print("Planner::Accecpted Connections")
        
        # Threadsafe queue
        self.__queue = queue.Queue()
        self.distributeTaskLock = Lock()
        self.dataAvailable = Condition(self.distributeTaskLock)
        
        # Spawn thread to set up Distribute task listener
        thread = threading.Thread(target=self.DistributeTask,args=[])
        thread.start()
    

    def AddTaskToQueue(self, path):
        '''
            Add pre-processed list of coordinates to queue
        '''
        
        with self.distributeTaskLock:
            print("Planner::Add task to queue")
            self.__queue.put(path)
            self.dataAvailable.notify()
    
    def DistributeTask(self):
        '''
            Pick 
        '''
        try:
            with self.distributeTaskLock:
                
                while self.__queue.empty():
                    self.dataAvailable.wait()
                
                print(f"len of planner before get: {self.__queue.qsize()}")
                task_to_distribute = self.__queue.get()
                first_x_coordinate = task_to_distribute[0, 0, 0]
                
                print(f"first_x_coordinate: {first_x_coordinate}")
                
                for i in range(self.__num_artists):
                    if (first_x_coordinate >= self.__artist_x_boundaries[i][0] 
                        and first_x_coordinate < self.__artist_x_boundaries[i][1]):
                        # Send robot i the path
                        print(f"len of planner after get: {self.__queue.qsize()}")
                        print("Planner::Send task to artist")
                        self.send_messages(task_to_distribute, i)
                    else: 
                        print("not distrbuting task to any artists bc of out of range")
            
        except Exception as e:
            print(e)
        
        self.DistributeTask()
        
        
    def send_messages(self, msg, artist_index):
        message_to_send = msg.tobytes()
        msg_len = str(len(message_to_send))
        
        metadata_str = str(msg.shape) + ";int64;" + msg_len + ";"
        metadata_to_send = metadata_str.encode('utf-8')
        padded_metadata = metadata_to_send.ljust(1024, b'\0')
        
        # print(f"metadata_str: {metadata_str}")
        # print(metadata_to_send.decode())
        
        print(f"message_to_send is of size: {len(padded_metadata)}")
        self.connections[artist_index][0].sendall(padded_metadata)
        
        # # Sending Actual Data... 
        
        # # message_to_send = (str(msg.tobytes())).encode()
        padded_array = message_to_send.ljust((math.ceil(len(message_to_send) / 1024)) * 1024, b'\0')

        print(f"message_to_send is of size: {len(padded_array)} bytes")
        self.connections[artist_index][0].sendall(padded_array)
        
        # # response after the message is sent
        # data = self.sever_socket.recv(1024)
        # print("Server says:", data.decode())

