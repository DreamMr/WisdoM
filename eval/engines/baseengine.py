import os
from abc import ABC, abstractmethod
import numpy as np


class BaseEngine(ABC):
    def __init__(self, use_wisdom=False, alpha=0.85, theta=0.79):
        self.use_wisdom = use_wisdom
        self.alpha = alpha
        self.theta = theta
    
    def initialize(self):
        pass
    
    @abstractmethod
    def run(self,cfg,builder,**kwargs):
        pass
    
    def build_model(self,cfg,builder):
        model = builder.build_model(cfg)
        return model
    
    def build_dataset(self,cfg,builder):
        dataset = builder.build_dataset(cfg)
        return dataset
    
    def cal_high(self,prob):
        sorted_prob = sorted(prob,reverse=True)
        dis = sorted_prob[0] - sorted_prob[1]
        
        return dis
    
    def fuse(self, prob_A, prob_B):
        diff = self.cal_high(prob_A)
        if diff > self.theta:
            return prob_A
        A_array = np.array(prob_A)
        B_array = np.array(prob_B)
        fuse = A_array + self.alpha * (B_array - A_array)
        return fuse