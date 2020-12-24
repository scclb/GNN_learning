# python 3
# Author: Scc_hy
# Create date: 2020-12-25
# Func: DGL框架实现GCN
# reference: https://mp.weixin.qq.com/s/wbN0WdNxKBCH3cFz4VLtcA


""" DGL 核心——消息传递
- 消息函数( message function ) : 传递消息的目的是将节点计算时需要的信息传递给它，因此对每条边来说，
每个源节点将会将自身的Embedding(e.src.data)和边的Embedding(egde.data)传递到目的节点；
对于每个目的节点来说，它可能会受到多个源节点传过来的消息，它会将这些消息存储在“邮箱”中。

- 聚合函数(reduce function) : 聚合函数的目的是根据邻居传过来的消息更自身节点的Embedding，对每个节点来说，
它先从邮箱(v.mailbox['m'])中汇聚消息函数所传递过来的消息(message)，并清空邮箱(v.mailbox['m'])内消息；然后
该节点结合汇聚后的结果和该节点原Embedding，更新节点Embedding。

"""

""" 
1- 在GCN中每个节点都有属于自己的表示 h_i;
2- 根据消息传递(message passing)的范式，每个节点将会受到来自邻居节点发送的Embedding
3- 每个节点将会对来自邻居节点的Embedding进行汇聚得到 中间表示 \hat{h_i}
4- 对中间节点表示 \hat{h_i} 进行线性变换，然后在利用非线性函数f进行计算： h_u^new = f(W_u \hat{h_u})
5- 利用新的节点表示  h_u^new 对该节点表示 h_u进行更新

目的节点的reduce函数很简单，因为按照GCN的数学定义，邻接矩阵和特征矩阵相乘，以为这更新后的特征矩阵的每一行是原特征矩阵某几行相加的形式，"某几行"是由邻接矩阵选定的，即对应节点的邻居所在的行。
因此目的节点reduce只需要通过sum将接受到的信息相加就可以了
"""

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.fuctional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

## step3 定义全连接层 来表示对中间节点表示  \hat{h_i}进行线性变换
class NodeApplyModule(nn.Module):
  def __init__(self, in_feats, out_feats, activation):
    super(NodeApplyModule, self).__init__()
    self.linear = nn.Linear(in_feats, out_feats)
    self.activation = activation

  def forward(self, node):
    h = self.linear(node.data['h'])
    h = self.activation(h)
    return {'h' : h}

## setp4 定义Embedding 更新层，以实现所有节点上进行消息传递，并利用NodeApplyModule对节点信息进行计算更新

class GCN(nn.Module):
  def __init__(self, in_feats, out_feats, activation):
    super(GCN, self).__init__()
    self.apply_mod = NodeApplyMoudle(in_feats, out_feats, activation)
  
  def forward(self, g, feature):
    g.ndata['h'] = feature
    g.update_all(gcn_msg, gcn_reduce)
    g.apply_nodes(func=self.apply_mod)
    return g.ndata.pop('h')
  
 ## step5 最后定义了一个包含两个GCN层的图神经网络分类器。我们通过向该分类器出入大小为1433的训练样本
 #### 以获得该样本所属的类别
 
 class Net(nn.Module):
  def
 


