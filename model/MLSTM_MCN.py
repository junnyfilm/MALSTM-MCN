import torch
import torch.nn as nn
from Dilated_Module import Dilated_Module
from seblock import SELayer
    
    
class MLSTMfcn_MSN(nn.Module):
    def __init__(self, *, num_classes, num_features,
                 num_lstm_out, num_lstm_layers=1, 
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn_MSN, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.dilated_1=Dilated_Module(self.num_features, self.conv1_nf)
        self.dilated_2=Dilated_Module(self.conv1_nf, self.conv2_nf)
        self.dilated_3=Dilated_Module(self.conv2_nf, self.conv3_nf)

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_classes)
    
    def forward(self, x):
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''
        x=x.transpose(2,1)
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''       
        # x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens, 
        #                                        batch_first=True, 
        #                                        enforce_sorted=False)
        x1, (ht,ct) = self.lstm(x)
        x1 = self.lstmDrop(x1)
        # x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
        #                                          padding_value=0.0)
        # print(x1.size())
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.relu(self.bn1(self.dilated_1(x2))))
        # print(x2.size())
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.dilated_2(x2))))
        # print(x2.size())
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.dilated_3(x2))))
        # print(x2.size())
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out,x_all