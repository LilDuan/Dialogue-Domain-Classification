from torch.utils.data import DataLoader

from Models.TextCNN.dataset import *
from Models.TextCNN.model import Model
from Models.TextCNN.parameters import Params
import Models.TextCNN.trainer as trainer

if __name__ == '__main__':
    datasets = ['CrossWOZ', 'RiSAWOZ', 'SGD']
    dataset = datasets[1]
    params = Params(dataset)

    vocab = params.idx2word # params中调用了build_vocab方法用于构建<idx2word> 和<word2idx>

    embedding_weights = load_pretrained_embedding(params)

    train_data = MyDataset(params, params.train_path, vocab)
    dev_data = MyDataset(params, params.dev_path, vocab)
    test_data = MyDataset(params, params.test_path, vocab)

    train_iter = DataLoader(train_data, batch_size=params.batch_size, collate_fn=collate_fn)
    dev_iter = DataLoader(dev_data, batch_size=params.batch_size, collate_fn=collate_fn)
    test_iter = DataLoader(test_data, batch_size=params.batch_size, collate_fn=collate_fn)


    model = Model(vocab_size = params.vocab_size,
                  embedding_dim=params.embedding_dim,
                  embedding_weights=embedding_weights,
                  num_filters=params.num_filters,
                  filter_sizes=params.filter_sizes,
                  output_dim=params.class_num,
                  dropout=params.dropout,
                  padding_idx=params.pad).to(params.device)

    trainer.init_network(model)
    print(model.parameters)

    trainer.train(params, model, train_iter, dev_iter, test_iter)













