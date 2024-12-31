import os
import importlib
from models.basemodel import BaseModel
from engines.baseengine import BaseEngine
from datasets.basedataset import BaseDataset


class Builder:
    def __init__(self):
        pass
    
    def build_model(self,cfg):
        if cfg.models['model_name'] is None:
            raise ValueError('Not define model !')
        
        model_name = 'models.' + cfg.models['model_name']
        modellib = importlib.import_module(model_name)
        model = None
        target_model_name = cfg.models['model_name']
        
        for name,cls in modellib.__dict__.items():
            if name.lower() == target_model_name.lower() and issubclass(cls,BaseModel):
                model = cls
                break
                
        if model is None:
            raise ValueError('In %s.py, there is not a subclass of BaseModel with class name that matches %s in lower case.'%(model_name,target_model_name))
        
        params = cfg.models.get('params',{})
        instance = model(**params)
        
        return instance
    
    
    def build_engine(self,cfg):
        if cfg.engine['engine_name'] is None:
            raise ValueError("Not define engine")
        
        engine_name = 'engines.' + cfg.engine['engine_name']
        enginelib = importlib.import_module(engine_name)
        engine = None
        target_engine_name = cfg.engine['engine_name']
        
        for name,cls in enginelib.__dict__.items():
            if name.lower() == target_engine_name.lower() and issubclass(cls, BaseEngine):
                engine = cls
                break
            
        if engine is None:
            raise ValueError(
                'In %s.py, there is not a subclass of BaseEngine with class name that matches %s in lower case.'%(engine_name,target_engine_name)
            )
            
        if 'params' in cfg.engine and cfg.engine['params'] is not None:
            params = cfg.engine['params']
            instance = engine(**params)
        else:
            instance = engine()
            
        return instance
    
    def build_dataset(self,cfg):
        if cfg.dataset['dataset_name'] is None:
            raise ValueError("Not define dataset")
        
        dataset_name = 'datasets.' + cfg.dataset['dataset_name']
        datasetlib = importlib.import_module(dataset_name)
        dataset = None
        target_dataset_name = cfg.dataset['dataset_name']
        
        for name,cls in datasetlib.__dict__.items():
            if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
                dataset = cls
                break
            
        if dataset is None:
            raise ValueError(
                'In %s.py, there is not a subclass of BaseDataset with class name that matches %s in lower case.'%(dataset_name,target_dataset_name)
            )
            
        if 'params' in cfg.dataset and cfg.dataset['params'] is not None:
            params = cfg.dataset['params']
            instance = dataset(**params)
        else:
            instance = dataset()
            
        return instance