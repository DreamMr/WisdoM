import os
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def eval(self,cfg,builder):
        pass
    
    def solve_sample(self,data=None):
        pass
    
    def initialize(self,cfg):
        if hasattr(cfg,'prompt'):
            self.logger.info('Using prompt: {}'.format(cfg.prompt))
            self.prompt_template = cfg.prompt
        else:
            self.prompt_template = '{}'
    
    def wrapper_prompt(self,text):
        n_text = self.prompt_template.format(text)
        return n_text