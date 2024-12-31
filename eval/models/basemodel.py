import os
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def score(self,**kwargs):
        pass
    
    @abstractmethod
    def generate(self,**kwargs):
        pass
    
    def instruct_template(self,sentence_list,**kwargs):
        return sentence_list

        