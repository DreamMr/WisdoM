from torch.utils.data import Dataset

def collate_fn(data_list):
    key_set = list(data_list[0].keys())
    
    res_dic = {}
        
    for data in data_list:
        for key in key_set:
            res_dic[key] = data[key]
    
    return res_dic

class ABSADataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
