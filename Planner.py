

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
from socket import socket
import threading

class Planner:
    
    
    def __init__(self, img_w, img_h):
        self.__num_artists = 2
        self.__artist_x_boundaries = []
        
        # Each boundary represents the [start_x, end_x)
        for i in range(self.__num_artists):
            self.__artist_x_boundaries.append((i * (img_w // self.__num_artists), (i + 1) * (img_w // self.__num_artists)))
            
        ARTIST_IP = "0.0.0.0"   # accpet connetion from any IP
        ARTIST_PORT = 65432     # need to used some non-priviledge port
        
        self.sever_socket = socket(socket.AF_INET, socket.SOCK_STREAM)
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
        queue.put(path)
        
        with self.distributeTaskLock:
            self.dataAvailable.notify()
    
    def DistributeTask(self):
        '''
            Pick 
        '''
        try:
            with self.distributeTaskLock:
                
                while self._queue.empty():
                    self.dataAvailable.wait()
                
                task_to_distribute = self.__queue.get()
                first_x_coordinate = task_to_distribute[0, 0, 0]
                
                print(f"first_x_coordinate: {first_x_coordinate}")
                
                for i in range(self.__num_artists):
                    if (first_x_coordinate >= self.__artist_x_boundaries[i][0] 
                        and first_x_coordinate < self.__artist_x_boundaries[i][1]):
                        # Send robot i the path
                        self.send_messages(task_to_distribute, i)
        except Exception:
            print("TODO")
        
        
    def send_messages(self, msg, artist_index):
        metadata_str = f"{msg.shape}, int64"
        self.connections[artist_index].send(metadata_str.encode())
        self.connections[artist_index].send(msg.encode())
        
        # # response after the message is sent
        # data = self.sever_socket.recv(1024)
        # print("Server says:", data.decode())

