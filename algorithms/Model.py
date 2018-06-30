'''
Created on 13.04.2018

@author: malte
'''
import abc
from abc import abstractmethod

class Model(abc.ABC):
    '''
    classdocs
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
    @abstractmethod
    def init(self, train, test):
        pass
    
    @abstractmethod
    def train(self, train):
        pass
    
    @abstractmethod
    def predict(self, name=None, tracks=None):
        pass
        
        