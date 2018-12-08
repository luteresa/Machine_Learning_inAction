
# coding: utf-8

# In[1]:


from math import  log
import operator

from IPython.display import Latex


# In[2]:


def createDataSet():
    dataSet = [[1,1,1,'yes'],
               [1,1,0,'yes'],
               [1,0,1,'no'],
               [0,1,0,'no'],
               [0,1,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels    


# # 信息的定义：$f(x)=-\log_2 p(x_i)$
# # 熵：信息的期望值
# $H=-\sum_{i=1}^np(x_i)\log_2p(x_i)$
# # 

# In[3]:


'''
func: 计算数据集的香农熵

dataSet: 传入数据集

return: 香农熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)#获取数据集条目数量

    #建立标签字典
    labelCounts = {}

    # 计算每个标签出现的次数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0


    #使用标签发生频率计算类别出现的概率
    #p(xi) = labelCounts[key]/numEntries
    #信息定义l(xi) = -log(p(xi),2)
    #print('labelCounts:',labelCounts)
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2) # 所有类别信息期望就是熵
    return shannonEnt


# In[4]:


'''
myData,labels = createDataSet()
calcShannonEnt(myData)
'''


# In[5]:


'''#熵越高，则数据越混乱
myData[0][-1]="maybe"
calcShannonEnt(myData)
'''


# In[6]:


'''
func: 划分数据集(按给定的特征及对应特征值，返回数据子集)

dataSet: 待划分数据集
axis: 划分数据集的特征序列号
value: 需要返回的特征值(符合该条件则筛选出，返回)

return: 根据传入参数划分的数据子集
'''
def splitDataSet(dataSet, axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #print(featVec)
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# In[7]:


'''
print(myData)
splitDataSet(myData,0,1)
'''


# In[8]:


'''
func: 选择最好的数据集划分方式,

input:数据集

return:最佳划分特征值序号
'''
def chooseBestFeatureToSplit(dataSet):
    #获取数据集的特征值个数
    numFeatures = len(dataSet[0])-1
    #计算原始香农熵，保存最初的无序度量值
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1

    #遍历数据集中所有特征
    for i in range(numFeatures):
        #获取dataSet中所有元素的第i个特征
        featList = [example[i] for example in dataSet]
        #去重,获取所有特征类型
        uniqueVals = set(featList)
        #print('---index_', i, uniqueVals)
        newEntroy =0.0

        #遍历所有唯一特征值，
        for value in uniqueVals:
            #获得特征值value的子序列划分
            subDataSet = splitDataSet(dataSet,i,value)

            #求得子集的概率
            prob = len(subDataSet)/float(len(dataSet))
            #求出dataSet在第i个特征上的条件熵
            newEntroy += prob*calcShannonEnt(subDataSet)

        #信息增益 = 熵-条件熵
        infoGain = abs(baseEntropy - newEntroy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


# In[9]:


'''print(myData)
chooseBestFeatureToSplit(myData)
'''


# In[10]:


'''
func: 统计类列表各类个数

return: 出现最多的类别
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classCount:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems,key=operator.itemgetter(1),reversed=True)

    return sortedClassCount[0][0]


# In[11]:


'''
func: 创建决策树

dataSet:数据集
labels:标签列表

return:树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #print('-'*88)
    #print(classList)
    #classList[0]元素的个数，与列表长度相等，即类别完全相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    #print(featValues)
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)

    return myTree


# In[12]:


def classify(inputTree, featLabel, testVec):
    firstStr = list(inputTree.keys())[0]
    print(firstStr)
    secondDict = inputTree[firstStr]  #inputTree的子节点列表
    featIndex = featLabel.index(firstStr) #获取特征标签对应索引
    print(secondDict.keys())
    for key in secondDict.keys():
        print(key)
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],featLabel,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# In[13]:


'''myData,labels = createDataSet()
print(myData)
print(labels)
print(mytree)
labels_bk = labels[:]
mytree = createTree(myData,labels)
print('*'*88)
classify(mytree,labels_bk,[0,1,0])
'''


# In[14]:


'''
构造决策树会很耗时，数据集很大时，会耗费大量时间，但是若执行分类时，运用提前训练好的决策树，则会节省很多计算时间
python模块pickle，可以序列化对象，将训练好的决策树保存在磁盘上，并在需要时读取出来
'''
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    
    return pickle.load(fr)


# In[15]:


'''
storeTree(mytree,'classifierStorage.txt')
grabTree('classifierStorage.txt')
'''


# In[16]:


def main():
    myDat,labels1 = createDataSet()
    labels2 = labels1[:]
    #myDat[0][-1] = 'maybe'
    print(myDat)
    print('labels1:', labels1)
    print('Shannon:',calcShannonEnt(myDat))

    #print(splitDataSet(myDat,2,1))
    #print(chooseBestFeatureToSplit(myDat))
    yuTree = createTree(myDat,labels1)
    print('labels2:', labels2)
    print(yuTree)
    print('='*88)
    testVec = [0,1,0]
    print('labels2:',labels2)

    label = classify(yuTree,labels2,testVec)
    print('testVec:{}-->{}'.format(testVec,label))
    print('*'*88)
    print(yuTree)
    storeTree(yuTree,'yuTree.txt')




def test2():
    myTree = grabTree('yuTree.txt')
    print(myTree)


if __name__ == '__main__':
    main()
    #test2()

