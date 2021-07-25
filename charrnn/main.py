# coding:utf8
import sys, os
import torch as t
from data import get_data
from model import *
from torch import nn
from utils import Visualizer
import tqdm
from config import Config
from torch.utils.data import DataLoader
import ipdb
import torchnet
#--------------------
print(t.cuda.is_available())
from generate import generate


'==============================================================='
opt = Config()
#def train(**kwargs):
def train():
    # for k, v in kwargs.items():
    #     setattr(opt, k, v)

    opt.device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    device = opt.device
    #vis = Visualizer(env=opt.env)
    data, ix2word,word2ix = get_data()
    # 获取数据
    print(len(word2ix),len(ix2word))
    data = t.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=opt.batch_size,
                            shuffle=True)

    # 模型定义
    model = Poetrymodel(vocab_size=len(word2ix),embedding_size=128,hidden_size=256)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.model_path:
        try:
            model.load_state_dict(t.load(opt.model_path))
        except:
            pass
    model.to(device)
    print('success')
    #loss_meter = meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        #loss_meter.reset()
        losses=[]
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            #print('data_.shape',data_.shape) # 32*125
            #print(data_)
            # 训练
            #data_ = data_.long().transpose(1, 0)#.contiguous()
            data_ = data_.long()
            data_ = data_.to(device)

            optimizer.zero_grad()
            input_, target = data_[:,:-1], data_[ :,1:]  #32*124
            batch,seq_len=input_.size()
            #print('inputshape',type(input_),input_.shape,target.shape) #
            output, _ = model(input_)  #torch.Size([32, 124, 8293])

            #target=target.view(1,-1).squeeze(0) #报错RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            target=target.reshape(opt.batch_size*124)
            output=output.view(-1,len(word2ix))
            #print(target.shape,output.shape)
            loss = criterion(output, target)
            #loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #loss_meter.add(loss.item())
            try:
                losses.append(loss.cpu().detach().numpy())
            except:
                print('error')
            # 可视化
            if (1 + ii) % opt.plot_every == 0:

                #if os.path.exists(opt.debug_file):
                    #ipdb.set_trace()

                #vis.plot('loss', loss_meter.value()[0])

                # 诗歌原文
                # for _iii in range(input_.shape[0]):
                #     print(input_[ _iii].tolist())
                #     for _word in input_[_iii].tolist():
                #         print(ix2word[_word])

                poetrys = [[ix2word[_word] for _word in input_[ _iii].tolist()] for _iii in range(input_.shape[0])][:16]

                #print(poetrys)
                #vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')

                gen_poetries = []
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                    print(f'{word}开头的诗',gen_poetry)
                #vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')

        t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))

train()


