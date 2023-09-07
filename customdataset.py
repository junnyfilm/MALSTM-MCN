import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


#read data

with open(path+'cross_val_train_set_'+str(cv)+'_'+str(cond)+'.pickle', 'rb') as f:
    cross_val_train_set = pickle.load(f)
with open(path+'cross_val_train_label_'+str(cv)+'_'+str(cond)+'.pickle', 'rb') as f:
    cross_val_train_label = pickle.load(f)
with open(path+'cross_val_test_set_'+str(cv)+'_'+str(cond)+'.pickle', 'rb') as f:
    cross_val_test_set = pickle.load(f)
with open(path+'cross_val_test_label_'+str(cv)+'_'+str(cond)+'.pickle', 'rb') as f:
    cross_val_test_label = pickle.load(f)
    
    
    
#depthwise version
class customdataset(Dataset):
    def __init__(self, data, label): 
        super().__init__()
        self.data=torch.tensor(data)
        self.label=torch.tensor(label)
        
    def __len__(self):
        return len(self.label)
  
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.label[idx] 
                
        return  x.to(device).float(), label.to(device).long()

def loaders(tr_data,tr_label,ts_data,ts_label,random_state):
    train_set, valid_set, train_label, valid_label = train_test_split(tr_data, tr_label,random_state=random_state, train_size=0.75)
    
    traindataset = customdataset(train_set, train_label)
    validdataset = customdataset(valid_set, valid_label)
    testdataset = customdataset(ts_data, ts_label)
    
    traindataloader = DataLoader(traindataset, batch_size=32, shuffle=True, drop_last=True )
    validdataloader = DataLoader(validdataset, batch_size=32, shuffle=True, drop_last=True )
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False, drop_last=False )
    
    return traindataloader,validdataloader,testdataloader