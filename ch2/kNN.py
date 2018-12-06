
# coding: utf-8

# In[2]:


import numpy as np
import operator
from os import listdir


# In[11]:


def createDataSet():
    #四组数据，每组数据含两个特征值
    group = np.array([
                    [1.1,1.1],
                    [1.0,1.0],
                    [0,0],
                    [0,0.1]
                    ])
    #group数据对应标签信息，大小与group行数相同
    labels = ['A','A','B','B']
    return group,labels


# In[18]:


group,labels = createDataSet()


# In[16]:


group


# In[19]:


labels


# \begin{equation}
# distance=\sqrt{(x1-x2)^{2}+(y1-y2){^2}}
# \end{equation}

# In[42]:


'''
inX: 待预测的输入向量
dataSet: 样本训练集
labels: 标签向量
k: 最近邻居个数
'''
def classify0(inX, dataSet,labels,k):
    '''
    step1: 计算输入向量inX与所有样本点的欧式距离
    '''
    dataSetSize = dataSet.shape[0]
    print(np.tile(inX,(dataSetSize,1)))
    #行向重复1次，列向重复dataSetSize次，扩展得到新矩阵与原矩阵相减
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    print(diffMat)
    sqDiffMat = diffMat**2
    print(sqDiffMat)
    #平方再求和，sum(axis=0)-->行相加，sum(1)-->列相加
    sqDistances = sqDiffMat.sum(axis=1)
    #print("sqDistances:", sqDistances)

    # 再开方，得到与样本集所有元素的欧式距离
    distances = sqDistances**0.5
    #print("distances:",distances)

    #distances保存测试向量与样本集所有元素欧式距离，返回排序后的索引值
    sortedDistIndicies = distances.argsort()
    #print("sortedDistIndicies:", sortedDistIndicies)

    
    '''
    step2:选择距离最小的k个点
    '''
    classCount={}
    for i in range(k):
        #按序取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        #dict.get(key,dafault=None)字典返回指定key的值，若不存在，创建一个新键值
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    '''
    step3:选出k个点里，出现频率最高的类别
    '''
    #print('classCount:',classCount)
    #classCount.items():将字典分解为元祖列表
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    print(classCount)
    print(classCount.items())
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #print('sortedClassCount:', sortedClassCount)
    
    #返回预测的类别
    return sortedClassCount[0][0]


# In[43]:


classify0([0,0],group,labels,3)


# In[50]:


def file2matrix(filename):
    '''
    step1: 得到文件行数
    '''
    fr = open(filename)
    arrayOfLines = fr.readlines() #读取所有内容
    #得到文件内容行数
    numberOFLines = len(arrayOfLines)

    '''
    step2: 创建返回的NumPy矩阵
    '''
    #分别是解析数据的矩阵和分类标签向量
    returnMat = np.zeros((numberOFLines, 3))
    # print('numberOFLines:', numberOFLines)
    classLabelVector = []

    '''
    step3: 解析文件内容到列表
    '''
    index = 0
    for line in arrayOfLines:
        line = line.strip() #参数为空时，默认删除空白符('\n','\t','\r',' ')
        #对字符串line根据'\t'进行切片
        listFromLine = line.split('\t')
        #print(listFromLine)
        #提取前n-1列，存放到returnMat特征矩阵中
        returnMat[index, :] = listFromLine[0:-1]

        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat,classLabelVector


# In[52]:


datingDateMat,datingLabels=file2matrix('./datingTestSet2.txt')


# In[53]:


datingDateMat,datingLabels


# 使用Matplotlib创建三点图

# In[61]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDateMat[:,1],datingDateMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.xlabel('game time')
plt.ylabel('gabiqie time')
plt.show()


# In[65]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDateMat[:,0],datingDateMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.xlabel('flight miles')
plt.ylabel('game time')
plt.show()


# In[69]:


def autoNorm(dataSet):
    #获取数据集每列的最小，最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #获取每列的数据值范围
    diffs = maxVals - minVals
    print(diffs)

    #创建返回的归一化数据矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    print(minVals)
    print(normDataSet.shape)
    #获取dataSet的行数
    m = dataSet.shape[0]

    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals,(m,1)) #将一行多列向量minVals，扩展成m行多列矩阵
    #再除以最大/最小值差，得到归一化数据
    normDataSet = normDataSet/np.tile(diffs,(m,1))

    #返回归一化数据矩阵，数据范围，最小值
    return normDataSet,diffs,minVals


# In[70]:


normDataSet,diffs,minVals = autoNorm(datingDateMat)
#print(normDataSet,diffs,minVals)


# In[7]:


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        linrStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(linrStr[j])

    return returnVect


# In[71]:


def handwritingClassTest():
    hwLabels = []
    #获取目录内容
    dirName = './digits/trainingDigits/'
    trainingFileList = listdir(dirName)
    print(trainingFileList)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNmuStr = int(fileStr.split('_')[0])
        hwLabels.append(classNmuStr)
        trainingMat[i,:] = img2vector(dirName+'/'+fileNameStr)
    dirName = 'digits/testDigits'
    testFileList = listdir(dirName)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(dirName+'/'+fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with:%d,the real answer is:%d'%(classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0

    print('\nthe total number of errors is:%d'%errorCount)
    print('\nthe total error rate is:%f'%(errorCount/float(mTest)))



# In[9]:


if __name__ == '__main__':
    #filename = 'datingTestSet2.txt'
    #datingDataMat,datingLabels = file2matrix(filename)
    #normDataSet, ranges, minVals = autoNorm(datingDataMat)
    #print(datingDataMat)
    #print(normDataSet)
    #print(datingLabels)

    #handwrite
    #testVector = img2vector('digits/trainingDigits/0_13.txt')
    #print(testVector)
    handwritingClassTest()

