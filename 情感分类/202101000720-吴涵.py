import paddle
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn as nn

# cpu/gpu环境选择，在paddle.set_device()输入对应运行设备
device = paddle.set_device('gpu')

# 加载训练数据
train_dataset = paddle.text.datasets.Imdb(mode='train')
# 加载测试数据
eval_dataset = paddle.text.datasets.Imdb(mode='test')

# 获取数据集的词表 并且已经排序
word_dict = train_dataset.word_idx
# 在词表中添加特殊字符，用于对序列进行补齐
word_dict['<pad>'] = len(word_dict)

pad_id = word_dict['<pad>']
classes = ['negative', 'positive']
# 生成句子列表
def ids_to_str(ids):
    words = []
    for k in ids:
        # list(word_dict)将的字典的key转为list
        w = list(word_dict)[k]
        words.append(w if isinstance(w, str) else w.decode('utf-8'))
    return ' '.join(words)
    

seq_len = 200
batch_size = 32
# 读取数据归一化处理
def create_padded_dataset(dataset):
    padded_sents = []
    labels = []
    for batch_ids, data in enumerate(dataset):
        sent, label = data[0], data[1]
        padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))]).astype('int32')
        padded_sents.append(padded_sent)
        labels.append(label)
    return np.array(padded_sents), np.array(labels)

# 对train, test进行实例化
train_sents, train_labels = create_padded_dataset(train_dataset)
test_sents, test_labels = create_padded_dataset(eval_dataset)

# 用Dataset和DataLoader进行数据集加载
class IMDBDataset(paddle.io.Dataset):
    '''
    继承paddle.io.Dataset类进行封装数据
    '''
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]
        return data, label
    
    def __len__(self):
        return len(self.sents)

train_dataset = IMDBDataset(train_sents, train_labels)
test_dataset = IMDBDataset(test_sents, test_labels)

train_loader = paddle.io.DataLoader(train_dataset, return_list=True, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = paddle.io.DataLoader(test_dataset, return_list=True, shuffle=False, batch_size=batch_size, drop_last=True)


# 参数设置
vocab_size = len(word_dict) + 1
emb_size = 256
epochs = 2
learning_rate = 0.001

# 简单RNN模型
# class MyRNN(paddle.nn.Layer):
#     def __init__(self):
#         super(MyRNN, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_size)# 嵌入层
#         self.rnn = nn.SimpleRNN(emb_size,emb_size,1,dropout=0.5)# 循环神经网络层
#         self.linear = nn.Linear(in_features=emb_size, out_features=2)# 线性层
#         self.dropout = nn.Dropout(0.5)# Dropout层

#     def forward(self, inputs):
#         # 嵌入层输出
#         x = self.embedding(inputs)
#         # RNN 层输出
#         x, _ = self.rnn(x)
#         # 取 RNN 最后时刻的输出
#         x = x[:, -1, :]
#         # 应用 dropout
#         x = self.dropout(x)
#         # 通过线性层得到最终输出
#         output = self.linear(x)
#         return output

# 双向LSTM模型
class MyRNN(paddle.nn.Layer):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)# 嵌入层
        self.rnn = nn.LSTM(emb_size, emb_size, num_layers=2, direction='bidirectional', dropout=0.5) # LSTM层
        self.linear = nn.Linear(in_features=emb_size*2, out_features=2)# 全连接层
        self.dropout = nn.Dropout(0.5)# Dropout层

    def forward(self, inputs):
        emb = self.dropout(self.embedding(inputs))# 嵌入层
        output, (hidden, _) = self.rnn(emb)# LSTM层
        hidden = paddle.concat((hidden[-2,:,:], hidden[-1,:,:]), axis=1) # 合并前向和后向 LSTM 的隐藏状态  
        hidden_t = output[:,-1,:]
        output = self.dropout(output[:,-1,:])
        return self.linear(output)
    

# 可视化定义
def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel('iter', fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()

# 对模型进行封装
def train(model):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    steps = 0
    Iters, total_loss, total_acc = [], [], []
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            steps += 1
            sent, label = data[0], data[1]
            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)

            # 500个batch输出一次结果
            if batch_id % 500 == 0:
                Iters.append(steps)
                total_loss.append(loss.numpy()[0])
                total_acc.append(acc.numpy()[0])
                print('epoch: {}, batch_id: {}, loss is: {}'.format(epoch, batch_id, loss.numpy()))
            
            loss.backward()
            opt.step()
            opt.clear_grad()

        # 每个epoch后对模型进行评估
        model.eval()
        accuracies, losses = [], []
        for batch_id, data in enumerate(test_loader):
            sent, label = data[0], data[1]
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print('[validation]accuracy: {}, loss: {}'.format(avg_acc, avg_loss))
        model.train()
        # 可视化查看，绘制每一轮迭代完成后，模型的loss值和acc值的结果
        draw_process('training loss', 'red', Iters, total_loss, 'training loss')
        draw_process('training acc', 'green', Iters, total_acc, 'training acc')
        
    # 保存模型
    paddle.save(model.state_dict(), str(epoch) + '_model_final.pdparams')
model = MyRNN()
train(model)

# 导入模型
model_state_dict = paddle.load('1_model_final.pdparams')
model = MyRNN()
model.set_state_dict(model_state_dict)
model.eval()
label_map = {0:'negative', 1:'positive'}
accuracies, losses, samples, predictions = [], [], [], []
for batch_id, data in enumerate(test_loader):
    sent, label = data[0], data[1]
    logits = model(sent)

    for idx, probs in enumerate(logits):
        label_idx = np.argmax(probs)
        labels = label_map[label_idx]
        predictions.append(labels)
        samples.append(sent[idx].numpy())

    loss = paddle.nn.functional.cross_entropy(logits, label)
    acc = paddle.metric.accuracy(logits, label)

    accuracies.append(acc.numpy()[0])
    losses.append(loss.numpy()[0])

avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
print('[validation] accuracy: {}, loss: {}'.format(avg_acc, avg_loss))
print('数据: {}\n情感: {}'.format(ids_to_str(samples[0]), predictions[0]))