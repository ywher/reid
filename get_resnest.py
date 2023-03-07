import torch
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)