import numpy as np
from typing import Dict,Text
import torch as t
import torch.functional as f
from torch import nn
from torch.autograd import Variable

rnn=nn.LSTM(10,20,2)
h0=t.randn(2,3,20)
c0=t.randn(2,3,20)
input=t.randn(5,3,10)
#output,(cn,hn)=rnn(input,(h0,c0))
#print(output.size())

print(input.shape)

def get_data(inputfile='data/tang.npz'):
    datas=np.load(inputfile,allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()

    return data,ix2word,word2ix

if __name__=='__main__':
    data, ix2word, word2ix = get_data('data/tang.npz')
    print(''.join([ix2word[_] for _ in data[10]]))

#<START>夏景已难度，怀贤思方续。乔树落疎阴，微风散烦燠。伤离枉芳札，忻遂见心曲。蓝上舍已成，田家雨新足。讬邻素多欲，残帙犹见束。日夕上高斋，但望东原绿。<EOP>