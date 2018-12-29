以梯度下降为例对逻辑回归进行求解，其迭代公式的推导过程如下：

∂J(ω)∂ωj=−1m∑im[yi(1−hω(xi))⋅(−xi,j)+(1−yi)hω(xi)⋅(xi,j)]=−1m∑im(−yi⋅xi,j+hω(xi)⋅xi,j)=−1m∑im(hω(xi)−yi)xi,j(12)
(12)∂J(ω)∂ωj=−1m∑im[yi(1−hω(xi))⋅(−xi,j)+(1−yi)hω(xi)⋅(xi,j)]=−1m∑im(−yi⋅xi,j+hω(xi)⋅xi,j)=−1m∑im(hω(xi)−yi)xi,j
上述中xi,jxi,j表示第ii个样本的第jj个属性的取值。 
于是，ωω的更新方式为：

ωj+1=ωj−α∑i=1m(hω(xi）−yi)xx,j(13)
(13)ωj+1=ωj−α∑i=1m(hω(xi）−yi)xx,j
对于随机梯度下降，每次只取一个样本，则ωω的更新方式为：

ωj+1=ωj−α(hω(x）−y)xj(13)
(13)ωj+1=ωj−α(hω(x）−y)xj
其中xx为这个样本的特征值，yy为这个样本的真实值，xjxj为这个样本第jj个属性的值。