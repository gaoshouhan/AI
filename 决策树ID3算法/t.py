import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split 

# 读取数据
def readData(filename):
    dataset = pd.DataFrame(pd.read_csv(filename, encoding="utf-8"))
    dataset.drop(labels=["编号"], axis=1, inplace=True)  # 删除编号列
    dataset["好瓜"].replace(to_replace=["是", "否"], value=["好瓜", "坏瓜"], inplace=True)  # 将“是”替换为“好瓜”，“否”替换为“坏瓜”
    return dataset

# 计算信息熵
def calEnt(dataset):
    frequency = dataset["好瓜"].value_counts() / len(dataset["好瓜"])  # 计算每个类别的频率
    entropy = -sum(pk * np.log2(pk) for pk in frequency)  # 计算信息熵
    return entropy

# 根据离散特征划分数据集
def splitDiscrete(D, feature): 
    splitD = []
    for Dv in D.groupby(by=feature, axis=0):
        splitD.append(Dv)
    return splitD

# 根据连续特征划分数据集
def splitContinuous(D, feature, splitValue):
    splitD = []
    splitD.append(D[D[feature] <= splitValue])
    splitD.append(D[D[feature] > splitValue])
    return splitD

# 计算信息增益（离散特征）
def calculateGainDiscrete(D, feature):
    gain = calEnt(D) - sum(len(Dv[1]) / len(D) * calEnt(Dv[1]) for Dv in splitDiscrete(D, feature))
    return gain

# 计算信息增益（连续特征）
def calculateGainContinuous(D, feature):
    max_gain = 0
    splitValue = 0
    T = {}
    for f in featureList[-2:]:
        T1 = dataset[f].sort_values() 
        T2 = T1[:-1].reset_index(drop=True) 
        T3 = T1[1:].reset_index(drop=True)  
        T[f] = (T2 + T3) / 2
    for t in T[feature].values:  
        temp = calEnt(D) - sum(len(Dv) / len(D) * calEnt(Dv) for Dv in splitContinuous(D, feature, t))
        if max_gain < temp:
            max_gain = temp
            splitValue = t
    return max_gain, splitValue

# 选择最佳划分特征
def chooseBestFeature(D, A):
    information_gain = {}
    for feature in A:
        if feature in ["密度", "含糖率"]:  
            ig, splitValue = calculateGainContinuous(D, feature)
            information_gain[feature+"<=%.3f"%splitValue] = ig
        else:
            information_gain[feature] = calculateGainDiscrete(D, feature)
    information_gain = sorted(information_gain.items(), key=lambda ig:ig[1], reverse=True)  
    return information_gain[0][0]

# 多数表决法确定类别
def majorityCnt(D): 
    return D["好瓜"].mode().iloc[0]

# 递归构建决策树
def treeGenerate(D, A):
    if len(splitDiscrete(D, "好瓜")) == 1:  # 如果所有样本属于同一类别，则停止划分，返回类别标签
        return D["好瓜"].iloc[0]

    if len(A) == 0 or len(splitDiscrete(D, A.tolist())) == 1:  # 如果所有特征已经用完或者所有样本在所有特征上取值相同，则停止划分，返回多数类别标签
        return majorityCnt(D)

    bestFeature = chooseBestFeature(D, A)  # 选择最佳划分特征
    if "<=" in bestFeature:  # 如果是连续特征
        bestFeature, splitValue = bestFeature.split("<=")
        myTree = {bestFeature+"<="+splitValue:{}}  # 构建节点
        [D0, D1] = splitContinuous(D, bestFeature, float(splitValue))  # 根据最佳划分特征和划分值划分数据集
        A0 = pd.Index(A)
        A1 = pd.Index(A)
        myTree[bestFeature+"<="+splitValue]["是"] = treeGenerate(D0, A0)  # 递归构建左子树
        myTree[bestFeature+"<="+splitValue]["否"] = treeGenerate(D1, A1)  # 递归构建右子树
    else: 
        myTree = {bestFeature:{}}
        for bestFeatureValue, Dv in splitDiscrete(D, bestFeature):  # 如果是离散特征
            if len(Dv) == 0:
                return majorityCnt(D)
            else:
                A2 = pd.Index(A)
                A2 = A2.drop([bestFeature])
                Dv = Dv.drop(labels=[bestFeature], axis=1)
                myTree[bestFeature][bestFeatureValue] = treeGenerate(Dv, A2)
    return myTree

def prePruning(D, A, threshold=0.95):
    # 如果当前节点的样本全部属于同一类别，或者特征集为空，则停止划分
    if len(set(D['好瓜'])) == 1 or len(A) == 0:
        return majorityCnt(D)

    # 计算当前节点的信息熵
    current_entropy = calEnt(D)

    # 选择最佳划分特征
    best_feature = chooseBestFeature(D, A)

    # 计算信息增益
    if "<=" in best_feature: 
        best_feature, split_value = best_feature.split("<=")
        _, D1 = splitContinuous(D, best_feature, float(split_value))
        _, D2 = splitContinuous(D, best_feature, float(split_value))
        new_entropy = len(D1) / len(D) * calEnt(D1) + len(D2) / len(D) * calEnt(D2)
    else:
        _, D1 = splitDiscrete(D, best_feature)
        _, D2 = splitDiscrete(D, best_feature)
        new_entropy = len(D1) / len(D) * calEnt(D1) + len(D2) / len(D) * calEnt(D2)

    # 如果信息增益小于阈值或者划分后的数据集样本比例小于阈值，则停止划分，返回多数类别
    if current_entropy - new_entropy < threshold or len(D1) / len(D) < threshold:
        return majorityCnt(D)
    else:
        if "<=" in best_feature: 
            best_feature, split_value = best_feature.split("<=")
            myTree = {best_feature+"<="+split_value:{}}
            [D0, D1] = splitContinuous(D, best_feature, float(split_value))
            A0 = pd.Index(A)
            A1 = pd.Index(A)
            myTree[best_feature+"<="+split_value]["是"] = prePruning(D0, A0, threshold)
            myTree[best_feature+"<="+split_value]["否"] = prePruning(D1, A1, threshold)
        else: 
            myTree = {best_feature:{}}
            for best_feature_value, Dv in splitDiscrete(D, best_feature):
                if len(Dv) == 0:
                    return majorityCnt(D)
                else:
                    A2 = pd.Index(A)
                    A2 = A2.drop([best_feature])
                    Dv = Dv.drop(labels=[best_feature], axis=1)
                    myTree[best_feature][best_feature_value] = prePruning(Dv, A2, threshold)
    return myTree

def postPruning(D, myTree, valData):
    if isinstance(myTree, dict):
        keys = list(myTree.keys())
        for key in keys:
            if isinstance(myTree[key], dict):
                myTree[key] = postPruning(D, myTree[key], valData)

    # 如果当前节点是叶子节点，则返回其类别
    if not isinstance(myTree, dict):
        return myTree

    # 计算当前子树在验证集上的准确率
    acc_without_prune = test(myTree, valData)

    # 尝试剪枝
    leaf_label = majorityCnt(D)
    myTree_acc = test(myTree, valData)
    if acc_without_prune > myTree_acc:
        return leaf_label
    else:
        return myTree

def test(myTree, valData):
    correct = 0
    total = len(valData)

    for index, row in valData.iterrows():
        prediction = classify(myTree, row)
        if prediction == row['好瓜']:
            correct += 1

    accuracy = correct / total
    return accuracy

def classify(myTree, sample):
    if not isinstance(myTree, dict):
        # 如果当前节点是叶子节点，返回其类别
        return myTree
    else:
        # 获取当前节点的特征名称和取值
        feature, value = list(myTree.items())[0]

        # 如果样本中的特征值在当前节点的取值范围内，则继续向下遍历
        if feature.endswith('='):
            if sample[feature[:-1]] == value:
                return classify(value, sample)
            else:
                return classify(value, sample)
        else:
            if sample[feature] <= float(feature.split('<=')[1]):
                return classify(value['是'], sample)
            else:
                return classify(value['否'], sample)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei' 

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
							xytext=centerPt, textcoords='axes fraction', \
							va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = getTreeDepth(secondDict[key]) + 1
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
	yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = list(myTree.keys())[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalw, plotTree.yOff)
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			plotTree(secondDict[key], cntrPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalw
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalw = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5 / plotTree.totalw
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	plt.show()



if __name__ == "__main__":
    # 数据准备
    dataset = readData("D:\\VScode\\AI\\决策树实验\\xigua_data3.0.csv") 
    featureList = dataset.columns[:-1]  # 特征列表
    # 将数据集划分为训练集和测试集，其中测试集占比为20%，随机种子设置为42
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    # 决策树生成
    myTree = treeGenerate(dataset, featureList)  
    createPlot(myTree) 
