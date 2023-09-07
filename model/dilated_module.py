import torch
import torch.nn as nn

class Dilated_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Module,self).__init__()
        self.conv1=nn.Conv1d(in_channels, int(out_channels/4), kernel_size = 5, stride = 1, padding = ((5)//2), dilation=1, bias=False)        
        self.conv2=nn.Conv1d(in_channels, int(out_channels/4), kernel_size = 5, stride = 1, padding = ((5+4*2)//2), dilation=3, bias=False)        
        self.conv3=nn.Conv1d(in_channels, int(out_channels/4), kernel_size = 5, stride = 1, padding = ((5+4*4)//2), dilation=5, bias=False)
        self.conv4=nn.Conv1d(in_channels, int(out_channels/4), kernel_size = 5, stride = 1, padding = ((5+4*6)//2), dilation=7, bias=False)
        
    def forward(self,x):
        ''' input x should be in size [B,F,T], where 
            B = Batch size
            F = features
            T = Time samples
        '''
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        x4=self.conv4(x)
        
        y=torch.cat([x1,x2,x3,x4],dim=1)
        
        return y