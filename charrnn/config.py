import torch

class Config:
    data_path='data/tang.npz'
    pickle_path='data/tang.npz'
    epochs=10
    use_gpu=True if torch.cuda.is_available() else False
    embedding_size=128
    hidden_size=256
    lr=1e-3
    batch_size=64
    max_len=125
    prefix_word='床前明月光'  #激发词
    start_word=''
    acrostic=False
    model_prefix='checkpoints/tang'
    model_path='poetrymodel.pth'
    plot_every=2
    env = 'poetry'
    epoch = 20
    max_gen_len=128
