import os
os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import argparse
from tools.utils import read_config, write_args,save_txt
from tools.builder import Builder
import torch
from transformers import set_seed
from loggers.logger import Logger
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import traceback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",type=int,default=0)
    parser.add_argument("--local-rank",type=int,default=0)
    parser.add_argument("--config_path",type=str,required=True)
    parser.add_argument("--distributed_method", type=str, default="torch_run",
        help="distributed method, deepspeed or torch_run or None.",
    )
    parser.add_argument("--seed", type=int, default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument('--out', type=str, default='out',
        help="specify the name of the output folder",
    )
    
    args = parser.parse_args()
    return args


def main():
    # create config
    args = parse_args()
    config = read_config(args)
    local_rank = config.local_rank
    # set distributed
    if config.use_distributed and config.distributed_method =="torch_run":
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(local_rank)
        config.local_rank = local_rank
        config.rank = rank
        config.world_size= world_size
        torch.cuda.set_device(local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', 
                                            init_method='env://',
                                            world_size=world_size,
                                            rank=rank)
    else:
        config.rank = config.local_rank
    device = torch.device("cuda", local_rank)    
    config.device = device
    set_seed(config.seed)
    
    # create builder
    builder = Builder()
    
    # load model !
    model_hidden_size = 8192
    ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
    }
    
    
    # create Logger
    logger = Logger()
    
    # create model
    model = builder.build_model(config)
    # use distributed must do dschf = HfDeepSpeedConfig first
    if hasattr(config,'use_distributed') and config.use_distributed:
        ds_engine = deepspeed.initialize(model=model.model, config_params=ds_config)[0]
        ds_engine.module.eval()
    
    # read task config
    for dataset_name,task_config_path in config.task_list.items():
        try:
            task_config = read_config(args,task_config_path)
            task_config.out = os.path.join(task_config.out,dataset_name,task_config.experiment_name)
            task_config.rank = config.rank
            task_config.model_name = config.models['model_name']
            task_config.use_distributed = config.use_distributed
            if hasattr(config,'use_distributed') and config.use_distributed:
                task_config.world_size= config.world_size
                task_config.device = config.device
            # create result dir
            if config.rank == 0:
                if not os.path.exists(task_config.out):
                    os.makedirs(task_config.out)
                write_args(task_config)
            
            if hasattr(task_config,'n_shot') and isinstance(task_config.n_shot, int):
                task_config.n_shot = [task_config.n_shot]
            
            # create engine
            engine = builder.build_engine(task_config)
            
            # create dataset
            data_process = builder.build_dataset(task_config)
            
            # combine
            engine.model = model
            engine.data_process = data_process
            engine.logger = logger
            
            data_process.engine = engine
            data_process.model = model
            data_process.logger = logger
            
            model.logger = logger
            
            # start eval
            engine.initialize()
            #model.model.eval()
            data_process.initialize(task_config)
            data_process.eval(task_config,builder)
        except Exception as e:
            logger.info(type(e))
            logger.info(str(e))
            if config.rank == 0:
                traceback.print_exc()


if __name__ == '__main__':
    main()
    