

'''
    Essentially act as the distributor that gives task to bots

    Delegates list of coordinates of one chunk of an image to an artist.
    
    Contains a queue of a list of coordinates that are received from the 
    Processor.
    
    While there is an available artist, a list of coordinates is popped
    from the queue and passed as a message to an artist.

'''
import queue

class Planner:
    
    
    def __init__(self):
        # Threadsafe queue
        self.__queue = queue.Queue()
        
        # List of pid for each artist; use for communicating between machines
        self.__artists = []
        
    def AddTaskToQueue(self, path):
        '''
            Add pre-processed list of coordinates to queue
        '''
        queue.put(path)
    
    def DistributeTask(self, ):
        '''
            Pick 
        '''
        
        raise Exception("Not Yet Implemented")