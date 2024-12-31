import os
import numpy as np
import torch
import json
import string
import re
from datasets.basedataset import BaseDataset
import pandas as pd
from tools.utils import to_device, read_txt, save_txt, read_pickle,save_pickle, read_json, save_json
import os
from metrics.task_metrics import cal_f1_score,cal_accuracy,cal_precision,cal_recall

class MSEDDataset(BaseDataset):
    LABEL_MAPPING = {
        'sentiment':{
            'negative':0,
            'neutral':1,
            'positive':2
        },
        'emotion': {
            'happiness':0,
            'sad':1,
            'neutral':2,
            'disgust':3,
            'anger':4,
            'fear':5
        }
    }
    def __init__(self,path,preffix,preffix_context,image_root,context_path=None,label_type='sentiment'):
        BaseDataset.__init__(self)
        self.path = path
        self.preffix = preffix
        self.preffix_context = preffix_context
        self.label_type = label_type
        self.context_path = context_path
        self.image_root = image_root
    
    def read_file(self,path):
        tsv_data = read_json(path)
        
        if self.context_path is not None:
            context_dataset = read_json(self.context_path)
        else:
            context_dataset = None
        
        dataset = {}
        for i in range(len(tsv_data)):
            
            text = tsv_data[i]['caption']
            image_path = tsv_data[i]['image']
            image_path = os.path.join(self.image_root, image_path)
            label = tsv_data[i][self.label_type]
            
            context_text = None
            if context_dataset is not None:
                image_id = image_path.split('/')[-1].split('.')[0]
                context = context_dataset[image_id]
                context_text = self.preffix_context.format(text,context)
                text = self.preffix.format(text)
            else:
                text = self.preffix.format(text)
            
            
            sample = dataset.get(image_path,{})
            sample['text'] = text
            sample['context_text'] = context_text
            
            sample['label'] = label
            sample['idx'] = i
            if context_dataset is not None:
                sample['context'] = context
            else:
                sample['context'] = ""
            
            dataset[image_path] = sample
        
        return dataset
    
    def instruct_dataset(self,dataset):
        
        data_list = []
        for image_path,sample in dataset.items():
            dic = {'text':sample['text'], 'context_text': sample['context_text'],'image':image_path,'label':sample['label'],'idx':sample['idx'],'context':sample['context']}
            data_list.append(dic)
        
        for i in range(10):
            print(data_list[i]['text'])
        
        return data_list
    
    def eval(self,cfg,builder):
        
        file_list = [
            ('test','test.json')
        ]
        
        for mode,file_name in file_list:
            print("started process {}".format(file_name))
            dataset = self.read_file(os.path.join(self.path,mode,file_name))
            data_list = self.instruct_dataset(dataset=dataset)
            
            # engine run
            resp_dataset = self.engine.run(data_list,cfg)['idx_response_list']
            
            if not cfg.use_distributed or cfg.rank == 0:
                sorted_resp_dataset = sorted(resp_dataset,key=lambda x: x['idx'])
                save_json(sorted_resp_dataset,os.path.join(cfg.out,"{}_{}_result.json".format(cfg.experiment_name,mode)))
        
                # post process
                def get_label(text):
                    text = text.strip()
                    text = text.split('Answer: ')[-1]
                    
                    if text[0] == 'A':
                        return 2
                    elif text[0] == 'B':
                        return 1
                    elif text[0] == 'C':
                        return 0
                    else:
                        return -1
                    
                MAPPING = {'negative':0,'neutral':1,'positive':2}
                for dic in sorted_resp_dataset:
                    if 'response' not in dic:
                        continue
                    prediction = dic['response']
                    pred_label = get_label(prediction)
                    dic['pred_index'] = pred_label
                    dic['label_index'] = MAPPING[dic['label']]
                    

                save_json(sorted_resp_dataset,os.path.join(
                    cfg.out,'result_post_procrss_samples.json'
                ))
                # metric
                def cal(data,out_root):
                    preds = []
                    labels = []
                    for dic in data:
                        if 'all_f1_score' in dic:
                            continue
                        preds.append(int(dic['pred_index']))
                        labels.append(int(dic['label_index']))
                        
                    f1_score = cal_f1_score(labels,preds)
                    accuracy = cal_accuracy(labels,preds)
                    recall = cal_recall(labels,preds)
                    precision = cal_precision(labels,preds)
                    
                    print("MAC F1: {}, Acc: {}, Recall: {}, Precision: {}".format(f1_score,accuracy,recall,precision))
                    dic = {"all_f1_macro":f1_score,'accuracy':accuracy,'recall': recall,'precision':precision}
                    save_json(dic,os.path.join(
                        cfg.out,out_root
                    ))
                    
                                    
                out_root = os.path.join(cfg.out,'result_macro_choices.json')
                cal(sorted_resp_dataset,out_root)