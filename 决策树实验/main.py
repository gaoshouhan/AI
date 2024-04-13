import pandas as pd  
import numpy as np 
import treePlotter 

def readData(filename):
    dataset = pd.DataFrame(pd.read_csv(filename, encoding="utf-8"))
    dataset.drop(labels=["编号"], axis=1, inplace=True)
    dataset["好瓜"].replace(to_replace=["是", "否"], value=["好瓜", "坏瓜"], inplace=True)
    return dataset

def calEnt(dataset):
    frequency = dataset["好瓜"].value_counts() / len(dataset["好瓜"])  
    entropy = -sum(pk * np.log2(pk) for pk in frequency)  
    return entropy

def splitDiscrete(D, feature): 
    splitD = []
    for Dv in D.groupby(by=feature, axis=0):
        splitD.append(Dv)
    return splitD

def splitContinuous(D, feature, splitValue):
    splitD = []
    splitD.append(D[D[feature] <= splitValue])
    splitD.append(D[D[feature] > splitValue])
    return splitD

def calculateGainDiscrete(D, feature):

    gain = calEnt(D) - sum(len(Dv[1]) / len(D) * calEnt(Dv[1]) for Dv in splitDiscrete(D, feature))
    return gain

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

def majorityCnt(D): 
    return D["好瓜"].mode().iloc[0]

def treeGenerate(D, A):
    if len(splitDiscrete(D, "好瓜")) == 1: 
        return D["好瓜"].iloc[0]

    if len(A) == 0 or len(splitDiscrete(D, A.tolist())) == 1: 
        return majorityCnt(D)

    bestFeature = chooseBestFeature(D, A) 
    if "<=" in bestFeature: 
        bestFeature, splitValue = bestFeature.split("<=")
        myTree = {bestFeature+"<="+splitValue:{}}
        [D0, D1] = splitContinuous(D, bestFeature, float(splitValue))
        A0 = pd.Index(A)
        A1 = pd.Index(A)
        myTree[bestFeature+"<="+splitValue]["是"] = treeGenerate(D0, A0)
        myTree[bestFeature+"<="+splitValue]["否"] = treeGenerate(D1, A1)
    else: 
        myTree = {bestFeature:{}}
        for bestFeatureValue, Dv in splitDiscrete(D, bestFeature):
            if len(Dv) == 0:
                return majorityCnt(D)
            else:
                A2 = pd.Index(A)
                A2 = A2.drop([bestFeature])
                Dv = Dv.drop(labels=[bestFeature], axis=1)
                myTree[bestFeature][bestFeatureValue] = treeGenerate(Dv, A2)
    return myTree

def testSplit(dataset, feature, split_value):
    sub_data0 = dataset[dataset[feature] <= split_value]
    sub_data1 = dataset[dataset[feature] > split_value]
    if calEnt(sub_data0) + calEnt(sub_data1) > calEnt(dataset):
        return False
    return True

def prune(tree, dataset, validate_data):
    if not isinstance(tree, dict) or not dataset.shape[0]:
        return tree

    node = list(tree.keys())[0]
    if "<=" in node:
        feature, split_value = node.split("<=")
        split_value = float(split_value)
        sub_data0 = dataset[dataset[feature] <= split_value]
        sub_data1 = dataset[dataset[feature] > split_value]
        sub_validate0 = validate_data[validate_data[feature] <= split_value]
        sub_validate1 = validate_data[validate_data[feature] > split_value]
        tree[node]["是"] = prune(tree[node]["是"], sub_data0, sub_validate0)
        tree[node]["否"] = prune(tree[node]["否"], sub_data1, sub_validate1)
        if testSplit(validate_data, feature, split_value):
            preds0 = tree[node]["是"] if isinstance(tree[node]["是"], str) else majorityCnt(sub_data0)
            preds1 = tree[node]["否"] if isinstance(tree[node]["否"], str) else majorityCnt(sub_data1)
            validate_data.loc[validate_data[feature] <= split_value, 'Prediction'] = preds0
            validate_data.loc[validate_data[feature] > split_value, 'Prediction'] = preds1
            acc_before = np.mean(validate_data['Prediction'] == validate_data['好瓜'])
            acc_after = np.mean(majorityCnt(validate_data) == validate_data['好瓜'])
            if acc_after >= acc_before:
                return majorityCnt(validate_data)
    else:
        for feature_value in tree[node].keys():
            sub_data = dataset[dataset[node] == feature_value]
            sub_validate = validate_data[validate_data[node] == feature_value]
            tree[node][feature_value] = prune(tree[node][feature_value], sub_data, sub_validate)
    return tree

validate_data = pd.DataFrame({
    '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '浅白'],
    '根蒂': ['蜷缩', '蜷缩', '蜷缩', '稍蜷', '硬挺', '硬挺'],
    '敲声': ['浊响', '浊响', '浊响', '浊响', '清脆', '清脆'],
    '纹理': ['清晰', '清晰', '稍糊', '清晰', '模糊', '模糊'],
    '脐部': ['凹陷', '凹陷', '凹陷', '平坦', '平坦', '平坦'],
    '触感': ['硬滑', '硬滑', '硬滑', '软粘', '硬滑', '硬滑'],
    '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403],
    '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237],
    '好瓜': ['好瓜', '好瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜']
})

if __name__ == "__main__":

    # 数据准备
    dataset = readData("D:\\VScode\\AI\\决策树实验\\xigua_data3.0.csv") 
    featureList = dataset.columns[:-1] 

    # 决策树
    myTree = treeGenerate(dataset, featureList)  
    treePlotter.createPlot(myTree) 
    # 树的后剪枝
    prunedTree = prune(myTree, dataset, validate_data)
    # 绘制处理后的树
    treePlotter.createPlot(prunedTree) 

