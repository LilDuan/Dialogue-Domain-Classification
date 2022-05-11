# coding: UTF-8
import torch.nn as nn
import torch.nn.functional as F

from Models.TextCNN.dataset import *


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_weights, num_filters, filter_sizes, output_dim, dropout, padding_idx):
        super(Model, self).__init__()

        # 外部的embedding层
        outer_embedding = torch.tensor(embedding_weights).long()
        words, dim = outer_embedding.shape

        self.pretrained_embedding = nn.Embedding(words, dim, padding_idx) # padding_idx表示需要padding的位置，这里用params.pad表示，
                                                                                # 即单词列表中所有padding_idx的位置全部用0替换
        self.pretrained_embedding.weight.data.copy_(outer_embedding)
        self.pretrained_embedding.weight.requires_grad = False


        # 随机初始化的embedding层
        self.random_embedding = nn.Embedding(vocab_size,
                                        embedding_dim,
                                        padding_idx)
        self.random_embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size = (fs, embedding_dim)) for fs in filter_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim) # 最后一层，max_pooling拼接后 => 输出到各个的类别

    def conv_and_pool(self, x, conv):

        out = conv(x)   # x =  (batch_size, num_filters, (seq_len, embedding_dim))
        out = F.relu(out)   # out = batch_size * num_filters * (seq_len - filter_size + 1) * 1 , 因为卷积核的宽和embedding的长度一样，所以最后一个维度是1
        out = out.squeeze(3)  # out = # out = batch_size * num_filters * (seq_len - filter_size + 1)
        max_pooled = F.max_pool1d(out, out.size(2)) # max_pooled = batch_size * num_filters * 1

        out = max_pooled.squeeze(2) # out = batch_size * num_filters (一个卷积核对应一个max_pool得到的数字，一共有bs * num_filters个卷积核)

        return out



    def forward(self, x):
        # 经过两个embedding层
        out = self.pretrained_embedding(x) + self.random_embedding(x)

        # out = self.dropout(x)

        out = out.unsqueeze(1)  # 输入数据格式为(batch_size, 1(in_channel), Height, Width)， channel就是这里加上的维度

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # 拼接后为 batch_size* (num_filters * len(filter_sizes))

        out = self.dropout(out)

        out = self.fc(out)
        return out

