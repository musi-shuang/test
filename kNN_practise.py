from __future__ import print_function
from numpy import *
from itertools import islice
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import operator

# ----------------------------------------------------------------------------------
# 对于每一个在数据集中的数据点: 
#     计算目标的数据点（需要分类的数据点）与该数据点的距离
#     将距离排序: 从小到大
#     选取前K个最短距离
#     选取这K个中最多的分类类别
#     返回该类别来作为目标数据点的预测值
# ----------------------------------------------------------------------------------


# 将文本记录转化为numpy解析程序
def file2matrix(filename):
    """
    Desc：导入数据集
    parameters：数据文件路径
    return：数据矩阵 returnMat和对应的classLabelVector
    """
    fr = open(filename)
    # 获得文件的行数
    numberOflines = len(fr.readlines())
    # 生成对应的空矩阵
    returnMat = zeros((numberOflines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in islice(fr, 1, None):   #fr.readlines(hint: 1 ):
        # str.strip([chars]) --返回每行头尾指定字符所生成的新字符串
        line = line.strip() # 移除空白
        listFromLine = line.split(',')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        # print(type(listFromLine[-1])) 默认读取进来的数据都是str
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 分析数据：使用matplotlib画二维散点图查看数据
def useplt():
    x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
    plt.plot(x, np.sin(x))       # Plot the sine of each x point
    plt.show()   

def autoNorm(dataSet):
    """
    Desc：
        归一化特征，消除特征之间量级不同导致的影响
        常用方式：
            线性转化：(x-min)/(max-min) 
            log函数转换
            反余切函数转换
    parameter：
        数据集
    return:
        归一化的数据集
    """
    minVals = dataSet.min()
    maxVals = dataSet.max()
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet,ranges,minVals

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]    # 返回的是行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #将距离排序: 从小到大
    sortedDistIndicies = distances.argsort()
    #选取前K个最短距离， 选取这K个中最多的分类类别
    classCount={}
    for i in range(k): 
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    """
    Desc:
        对约会网站的测试方法
    parameters:
        none
    return:
        错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('/Users/musi/Documents/musi/learning/da/dataset/knn/hailun.csv')  # load data setfrom file
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = 0.9 # float(input("花在游戏上的时间百分比："))
    ffMiles = 3000 # float(input("飞机里程数："))
    iceCream = 90 # float(input("吃冰淇淋的公升数："))
    datingDataMat, datingLabels = file2matrix('/Users/musi/Documents/musi/learning/da/dataset/knn/hailun.csv')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])



######################################### 手写数字识别系统 ##################################################33
# 收集数据: 提供文本文件。
# 准备数据: 编写函数 img2vector(), 将图像格式转换为分类器使用的向量格式
# 分析数据: 在 Python 命令提示符中检查数据，确保它符合要求
# 训练算法: 此步骤不适用于 KNN
# 测试算法: 编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的
#          区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，
#          则标记为一个错误
# 使用算法: 本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取
#          数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统




if __name__ == '__main__':
    # print('hello,python')
    classifyPerson() 









