

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
    
    
    def __init__(self, num_artists : int):
        self.__num_artists = num_artists
        # __assigned_task[i] is true if the ith artist has been assigned an image
        self.__assigned_task = [False] * self.__num_artists
            
        ARTIST_IP = "0.0.0.0"   # accpet connetion from any IP
        ARTIST_PORT = 22     # need to used some non-priviledge port
        
        # SD TODO::comment below code (connection code) for wanting to look at images after they have been processed
        self.sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sever_socket.bind((ARTIST_IP, ARTIST_PORT))
        self.sever_socket.listen(self.__num_artists)
        
        # make sure to secure connection with artist
        self.connections = [self.sever_socket.accept() for _ in range(self.__num_artists)]
        
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
                
                task_to_distribute = self.__queue.get()
                
                # SD TODO::used to show the weird dimension of how we represent our coordinate
                first_x_coordinate = task_to_distribute[0, 0, 0]
                print(f"first_x_coordinate: {first_x_coordinate}")
                
                for i in range(self.__num_artists):
                    if not self.__assigned_task[i]:
                        self.__assigned_task[i] = True
                        # Send robot i the path
                        print(f"Planner::Send task to artist {i}")
                        self.send_messages(task_to_distribute, i)
                        break
                    else: 
                        print("not distrbuting task to artist bc already assigned")
        
        except Exception as e:
            print(e)
        
        self.DistributeTask()
        
        
    def send_messages(self, msg, artist_index):
        message_to_send = msg.tobytes()
        msg_len = str(len(message_to_send))
        
        metadata_str = str(msg.shape) + ";int64;" + msg_len + ";"
        metadata_to_send = metadata_str.encode('utf-8')
        padded_metadata = metadata_to_send.ljust(1024, b'\0')
        
        self.connections[artist_index][0].sendall(padded_metadata)
        
        padded_array = message_to_send.ljust((math.ceil(len(message_to_send) / 1024)) * 1024, b'\0')

        self.connections[artist_index][0].sendall(padded_array)
