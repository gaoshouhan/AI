# 导入相关包
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt

# 设置paddle默认的全局数据类型为float64
paddle.set_default_dtype('float64')
# 加载数据
train_dataset = paddle.text.datasets.UCIHousing(mode='train')  # 加载训练数据集
eval_dataset = paddle.text.datasets.UCIHousing(mode='test')    # 加载测试数据集
# 封装训练数据
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练数据的 DataLoader
eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=8, shuffle=False)     # 创建测试数据的 DataLoader

# 模型配置
class Regressor(paddle.nn.Layer):  # 定义一个继承自paddle.nn.Layer的模型类
    def __init__(self):
        super(Regressor, self).__init__()
        # 定义神经网络结构
        self.linear1 = paddle.nn.Linear(13, 13)  # 输入大小为13，输出大小为13的线性层
        self.relu1 = paddle.nn.ReLU()            # ReLU激活函数
        self.linear2 = paddle.nn.Linear(13, 13)  # 输入大小为13，输出大小为13的线性层
        self.relu2 = paddle.nn.ReLU()            # ReLU激活函数
        self.linear3 = paddle.nn.Linear(13, 1)   # 输入大小为13，输出大小为1的线性层

    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        return x


# 模型训练
Batch = 0
# 记录批次
Batchs = []
# 记录损失值，用于后续绘图
all_train_loss = []

# 模型实例化
model = Regressor()  # 创建一个回归模型的实例
# 训练模型
model.train()  # 设置模型为训练模式
# 均方误差损失函数
mse_loss = paddle.nn.MSELoss()  # 定义均方误差损失函数
# 优化函数，随机梯度下降
opt = paddle.optimizer.SGD(learning_rate=0.0005, parameters=model.parameters())  # 随机梯度下降优化器

# 迭代次数
epochs_num = 180  # 训练的总迭代次数
step = 8           # 每多少步记录一次损失
for pass_num in range(epochs_num):
    for batch_id, data in enumerate(train_loader()):  # 遍历训练数据
        # data包括input, label
        image = data[0]   # 输入数据
        label = data[1]   # 标签数据
        # 前向计算
        predict = model(image)  # 使用模型进行预测
        loss = mse_loss(predict, label)  # 计算损失

        # 每隔一定步数记录一次损失
        if batch_id != 0 and batch_id % step == 0:
            Batch = Batch + step
            Batchs.append(Batch)
            # 记录损失值并打印
            all_train_loss.append(loss.numpy()[0])  # 记录损失值
            print('epoch:{}, step:{}, train_loss:{}'.format(pass_num, batch_id, loss.numpy()[0]))  # 打印损失值

        # 反向传播
        loss.backward()  # 反向传播计算梯度
        # 更新参数
        opt.step()       # 更新模型参数
        # 重置梯度
        opt.clear_grad()  # 清空梯度信息

# 保存模型
paddle.save(model.state_dict(), 'Regressor')  # 保存模型参数

# 模型评估
# 绘制损失值随模型迭代次数的变化过程
def draw_train_loss(Batchs, train_losses):
    title = 'training loss'
    plt.title(title, fontsize=24)
    plt.xlabel('batch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.plot(Batchs, train_losses, color='green', label='training loss')
    plt.legend()
    plt.grid()
    plt.show()
draw_train_loss(Batchs, all_train_loss)

# 验证模型
# 从保存的模型中加载模型
para_state_dict = paddle.load('Regressor')  # 加载保存的模型参数
model = Regressor()  # 创建一个新的回归模型实例
# 模型加载参数
model.set_state_dict(para_state_dict)  # 加载保存的模型参数
# 模型切换为验证模型
model.eval()  # 设置模型为评估模式

losses = []
infer_results = []
ground_truths = []
# 遍历测试集
for batch_id, data in enumerate(eval_loader()):
    image = data[0]   # 输入数据
    label = data[1]   # 标签数据
    ground_truths.extend(label.numpy())  # 记录真实标签
    predict = model(image)  # 使用模型进行预测
    infer_results.extend(predict.numpy())  # 记录预测结果
    loss = mse_loss(predict, label)  # 计算损失
    losses.append(loss.numpy()[0])    # 记录损失值
    avg_loss = np.mean(losses)        # 计算平均损失值
print('当前模型在验证集上的损失值为：', avg_loss)  # 打印验证集上的损失值

# 绘制真实值和预测值对比图
def draw_infer_result(ground_truths, infer_results):
    title = 'Boston'
    plt.title(title, fontsize=24)
    x = np.arange(1, 30)
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(ground_truths, infer_results, color='red', label='trainingcost')
    plt.grid()
    plt.show()
draw_infer_result(ground_truths, infer_results)
