import torch
from torch import nn                                #导入神经网络模块（neural network）
import torch.nn.functional as F                     #神经网络的数学函数
import math                                         #python自带的数学模块


#词嵌入层
class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)
    
#位置嵌入


      



