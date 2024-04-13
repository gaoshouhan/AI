from sklearn.model_selection import train_test_split
import pandas as pd  

def readData(filename):
    dataset = pd.DataFrame(pd.read_csv(filename, encoding="utf-8"))
    dataset.drop(labels=["编号"], axis=1, inplace=True)
    dataset["好瓜"].replace(to_replace=["是", "否"], value=["好瓜", "坏瓜"], inplace=True)
    return dataset

# 读取数据
dataset = readData("D:\\VScode\\AI\\决策树实验\\xigua_data3.0.csv") 

# 特征列表
featureList = dataset.columns[:-1]

# 将数据集划分为训练集和测试集，其中测试集占比为20%，随机种子设置为42
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# 打印训练集和测试集的样本数量
print("训练集样本数量:", len(train_data))
print(train_data)
print("测试集样本数量:", len(test_data))
print(test_data)