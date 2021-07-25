import numpy as np
from typing import Dict,Text
import torch as t
import torch.functional as f
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import   Config
from model import Poetrymodel
from data import get_data
import tqdm
import glob
opt=Config()



def train(epoch):

    data,ix2word,word2ix=get_data(opt.data_path)
    data=t.from_numpy(data)
    modellist=glob.glob('checkpoints/*.pth')
    print(modellist)

    model=Poetrymodel(len(word2ix),opt.embedding_size,opt.hidden_size)
    try:
        model.load_state_dict(modellist[0])    #检查是否可以继续上次的训练
    except:
        pass

    optimizer=t.optim.Adam(model.parameters(),lr=opt.lr)
    dataloder=DataLoader(data,batch_size=opt.batch_size,shuffle=True,num_workers=0,drop_last=True)  #用dataloader加载数据
    #num_worker设置为会报错,如果非整批数据输入报错则需要将drop——last设置为TRUE
    #0530

    cretion=nn.CrossEntropyLoss()       #使用该函数时无需将输出经过softmax层，直接将网络输出用来计算标签即可
    # loss_meter=meter.
    model.cuda()
    cretion.cuda()
    for i in range(epoch):
        pbar=tqdm.tqdm(enumerate(dataloder))
        for ii,data_ in pbar:
            pbar.set_description(f"加载第{ii}批次数据:")
            print(data_[0])  #data_:batch_size,seq_len ,进入embedding后
            print()
            #typea=type(data_)
            #print('typea:',typea)
            data_=data_.long().transpose(1,0) #必须转换为longtensor类型，也可以用LongTensor初始化
            #data_ = data_.long()

            #如果是float则会发生RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
            input=data_[:-1,:].cuda()    #转置的原因是为了方便这里
            target=data_[1:].cuda()
            optimizer.zero_grad()
            output,_=model(input)

            outputsize=output.size()
            #(seq*batch,hidden_size)
            loss=cretion(output.view(-1,outputsize[-1]),target.view(-1))  #(seq*batch,vocab),(batch_size,seq_len)要满足计算条件
            loss.backward()
            optimizer.step()
            if (ii % opt.plot_every):
                pass
        try:
            t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, i))
            print("成功保存")
        except:
            pass














train(opt.epochs)









