import yaml
from tools.config import Config
import json
import pickle
import time
import os
import jsonlines
import gzip

# read config file & load the param into the 'config'
def read_config(args,sub_task_config=None):
    config_path = sub_task_config if sub_task_config is not None else args.config_path
    # read config file
    with open(config_path,'r') as imf:
        yaml_config = yaml.safe_load(imf.read())
    
    config = Config()
    
    # load args param
    for k,v in sorted(vars(args).items()):
        setattr(config, k, v)
    
    # load config param
    for k in yaml_config.keys():
        setattr(config,k,yaml_config[k])
     
    return config

# write args into file
def write_args(args):
    
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(args.out, 'config.log')
    with open(file_name, 'wt') as config_file:
        config_file.write(message)
        config_file.write('\n')

# get local time format: year_month_day_hour_minute_second
def get_timestamp():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime()))

# copy from mmlu
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

# copy from mmlu
def read_txt(file: str):
    with open(file, 'r',errors='ignore') as f:
        return f.readlines()
        
# copy from mmlu
def save_txt(data: list, file: str):
    with open(file, 'w') as f:
        f.writelines(data)
    print(f'Save to {file}')

# copy from mmlu
def read_pickle(file: str) -> list:
    with open(file, 'rb') as f:
        return pickle.load(f)

# copy from mmlu
def save_pickle(data, file: str):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')

# copy from mmlu
def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

# copy from mmlu
def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f'Saved to {file}.')
    return

# dedup
def dedup(l:list):
    out = {}
    for i in l:
        out[i[0].item()] = i[1]
    return list(out.items())    

def read_jsonl(file):
    datalist=[]
    with open(file, "r+", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            datalist.append(item)
    return datalist

def save_jsonl(data, file):
    with jsonlines.open(file, 'w') as w:
        for item in data:
            w.write(item)
    print(f'Save to {file}')
    

def stream_jsonl(filename):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)