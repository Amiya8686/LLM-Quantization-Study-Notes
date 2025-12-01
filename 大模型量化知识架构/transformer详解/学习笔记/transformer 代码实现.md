# transformer 代码实现

## 导入组件

```python
import torch
from torch import nn                                #导入神经网络模块（neural network）
import torch.nn.functional as F                     #神经网络的数学函数
import math                                         #python自带的数学模块
```





## token embedding（词嵌入）

将token索引序列，转为词向量序列（矩阵）

```python
class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)
```

- **nn.embedding：**pytorch的词嵌入层。本质上一张token索引到对应词向量的映射表。
- **padding_idx=1：**使用索引为1的token来填充句子。我们有同时处理多个句子的需求，填充以让它们长度一致。









## position encoding（位置编码）



