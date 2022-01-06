import torch.nn as nn
import torch
import torch_geometric as tg



def weights_init_1(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv1') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def weights_init_2(m):
    if isinstance(m, tg.nn.ChebConv):
        print(m)
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()

def save_model(net,path, name_net):

  # This fucntion is used to save a specific model

    path_net =  path + '/' + name_net + '.pth'
    torch.save(net.cpu().state_dict(), path_net)
    net.cuda()

def load_model(net, path, name_net):

  # This function is used to load a specific model we saved before

    path_net =  path + '/' + name_net + '.pth'
    net.load_state_dict(torch.load(path_net))
    return net
