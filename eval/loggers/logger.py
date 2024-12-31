import logging
import os
import time
# from tools.utils import get_time

'''
function:
1. record config file
2. logging 
'''
class Logger:
    def __init__(self,rank=0):
        self.logger = logging.getLogger("logger")
        self.rank = rank
        
    # write args into file
    def write_args(self,args):
        if not self.validate():
            return
        
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
    
    def info(self,message):
        if not self.validate():
            return
        self.logger.info(message)
        
    def validate(self):
        '''
        just allow 1 process
        '''
        return True if self.rank == 0 else False
    