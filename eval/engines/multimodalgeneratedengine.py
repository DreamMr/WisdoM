from engines.baseengine import BaseEngine
import torch
import tqdm
from threading import Thread
from datasets.common import collate_fn,ABSADataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy
import numpy as np
MAPPING = {0:'A).',1:'B).',2:'C).'}
MAPPING_LABEL = {0:2,1:1,2:0}

class MultiModalGeneratedEngine(BaseEngine):
    def __init__(self,use_wisdom=False, alpha=0.85, theta=0.79):
        BaseEngine.__init__(self,use_wisdom, alpha, theta)
        
    def run(self,dataset,cfg,**kwargs):
        if cfg.use_distributed:
            print('use distributed')
            dataset = ABSADataset(dataset)
            sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=False)
            dataloader = DataLoader(dataset, collate_fn=collate_fn, sampler=sampler, batch_size=1, num_workers=8)
        else:
            dataloader = dataset
        resp_dataset = []
        auto_add_image_token = kwargs.get('auto_add_image_token',True)
        for data in tqdm.tqdm(dataloader):
            input_text = data['text']
            image_path = data.get('image',None)
            context_text = data.get('context_text', None)
            cur_resp = copy.deepcopy(data)
            
            resp = self.model.generate(
                input_text=input_text,
                image_path=image_path,
                max_new_tokens=cfg.max_new_tokens,
                auto_add_image_token=auto_add_image_token
            )
            
            prob = resp[1]
            resp_text = resp[0]
            
            if self.use_wisdom and context_text is not None:
                #print("Using WisdoM")
                resp = self.model.generate(
                    input_text=context_text,
                    image_path=image_path,
                    max_new_tokens=cfg.max_new_tokens,
                    auto_add_image_token=auto_add_image_token
                )
            
                context_prob = resp[1]
                context_resp_text = resp[0]
                
                fuse_prob = self.fuse(prob, context_prob)
                index = np.argmax(fuse_prob)
                response = MAPPING.get(index,'nan')
                prob = fuse_prob
                resp_text = response
                
            cur_resp['prob'] = {'A':prob[0],'B':prob[1],'C':prob[2]}
            cur_resp['response'] = resp_text
            resp_dataset.append(cur_resp)
        
        if cfg.use_distributed:
            response_list_distributed = [None for _ in range(cfg.world_size)]
            torch.distributed.all_gather_object(response_list_distributed, resp_dataset)
            response_list_distributed = [item for _resp in response_list_distributed for item in _resp]
            resp_dataset = response_list_distributed

        def dedup(dataset):
            idx_set = set()
            
            dedup_dataset = []
            for data in dataset:
                idx = data['idx']
                if idx not in idx_set:
                    dedup_dataset.append(data)
                    idx_set.add(idx)
            
            return dedup_dataset
        
        resp_dataset = dedup(resp_dataset)
        sorted_resp_dataset = sorted(resp_dataset,key=lambda x:x['idx'])
            
        return {'idx_response_list':sorted_resp_dataset}