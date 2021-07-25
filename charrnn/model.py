#coding:utf_8
import numpy as np
from typing import Dict,Text
import torch as t
import torch.functional as f
from torch import nn
from torch.autograd import Variable


class Poetrymodel(nn.Module):

    def __init__(self,vocab_size,embedding_size,hidden_size):
        super(Poetrymodel,self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(vocab_size,embedding_size)  #独热向量维度映射到编码维度
        self.lstm=nn.LSTM(embedding_size,self.hidden_size,num_layers=2)
        self.dropout=nn.Dropout(0.1)
        self.linear1=nn.Linear(self.hidden_size,vocab_size)

    def forward(self, x,HID=None):
        batch_szie,seq_len=x.size() #124,32
        #batch_szie,seq_len = x.size()
        self.out=self.embedding(x)
        if HID is None:
            # h_0=x.data.new_zeros(2,batch_szie,self.hidden_size).fill_(0).float()
            # c_0=x.data.new_zeros(2,batch_szie,self.hidden_size).fill_(0).float()
            h_0 = t.zeros(2, seq_len, self.hidden_size).double().cuda()  #一般细胞状态和隐藏状态初始化为0
            c_0 = t.zeros(2, seq_len, self.hidden_size).double().cuda()  #batch_size
            # h_0=Variable(h_0)
            # c_0=Variable(c_0)
        else:
            h_0,c_0=HID
        embed=self.embedding(x)
        #(seq*batch,embed_size)
        out,hidden=self.lstm(embed,(h_0,c_0)) #seq*batch,hidden)
        output=self.linear1(out)
        #output.transpose(1,0)
        return output,hidden



if __name__=='__main__':
    target = t.empty(128, 20, dtype=t.long).random_(8293)
    print(target[10].tolist())
    print(target.reshape(128 * 20).shape)
    target = target.view(1, -1).squeeze(0)
    print(target.shape)
    """考虑cuda"""
    a = t.tensor(np.arange(128 * 20, dtype=np.int32).reshape(128, 20), dtype=t.long)#.cuda()
    print(a)
    e=nn.Embedding(8293,50)#.cuda()  #embedding_siez=50
    lstm = nn.LSTM(50,256, num_layers=2,batch_first=True)
    a1=e(a)
    print(a1.shape)  #torch.Size([128, 20, 50])
    a2,_=lstm(a1)
    print(a2.shape)  #torch.Size([128, 20, 256])

    linear1 = nn.Linear(256, 8293)
    a3=linear1(a2)
    print(a3.shape)  #torch.Size([128, 20, 8293])
    criterion = nn.CrossEntropyLoss()
    target=t.empty(128,20,dtype=t.long).random_(8293)
    print(target.shape)
    print(a3.view(-1,8293).shape)
    a3=a3.view(-1,8293)
    target=target.view(1, -1).squeeze(0)
    """测试损失函数"""
    loss = criterion(a3, target)
    """检查模型"""
    a = t.tensor(np.arange(128 * 20, dtype=np.int32).reshape(128, 20), dtype=t.long).cuda()  # b*seq  12*20
    p = Poetrymodel(8293, 50, 256).cuda()
    print(p(a)[0].shape)  #torch.Size([128, 20, 8293])

    # print('a3shape:',a3.shape)
    # print('target:',target.shape)
    # loss=criterion(a3,target)
    # print(loss)
    """不能计算三维的交叉熵"""




