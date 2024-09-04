### 感知机（perceptron）

#### defination

接收多个信号，神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出1。这也称为 **神经元被激活**。  
这个界限值称为 **阀值**，用符号θ表示。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eb75aec6cd563f676d22ded96080e5f8.png)

#### AND，NAND, OR，XOR gate

根据数学公式可知，实现与门的参数选择有无数个。  
只要把实现与门的参数值的符号取反，就可以实现非门。

##### 感知机的实现（导入weight 和bias）

b为 bias, w1, w2 为weights

**AND的实现：**

```python
def AND(x1,x2):
	x = np.array([x1,x2])
	w = np.array([0.5,0.5])
	b = -0.7
	tmp = np.sum(x*w) + b
	if tmp<=0:
		return 0
	else:
		return 1
```

**NAND的实现：**

```python
def NAND(x1,x2):
	x = np.array([x1,x2])
	w = np.array([-0.5,-0.5])#只有权重和偏置与AND不同！其他都一样
	b = 0.7
	tmp = np.sum(x*w) + b
	if tmp<=0:
		return 0
	else:
		return 1
```

**OR的实现：**

```python
def OR(x1,x2):
	x = np.array([x1,x2])
	w = np.array([0.5,0.5])#只有权重和偏置与AND不同！其他都一样
	b = -0.2
	tmp = np.sum(x*w) + b
	if tmp<=0:
		return 0
	else:
		return 1

```

w1,w2是控制输入信号的重要性的参数，b是调整神经元激活的容易程度的参数

**XOR异或门**  
仅当x1或x2中的一方为1时，才会输出1  
用前文介绍的感知机无法实现这个异或门

#### 线性与非线性（linear and non-linear）

用曲线分割而成的空间称为**非线性空间**  
用直线分割而成的空间称为**线性空间**

#### 多层感知机（multi-layered perceptron）

XOR可以通过组合前面实现的AND，OR，NAND门来实现，异或门是一种多层结构的神经网络。  
**XOR的实现：**

```python
def XOR(x1,x2):
	s1 = NAND(x1,x2)
	s2 = OR(X1,X2)
	y = AND(s1,s2)
```

单层感知机只能表示线性空间，而多层感知机可以表示非线性空间。  
多层感知机在理论上可以表示计算机。

### 神经网络(neural network)

#### defination

一个重要性质是：神经网络可以自动地从数据中学到合适的权重参数  
用图来表示的话，最左边一列称为**输入层**，最右边的一列称为**输出层**，中间的一列称为**中间层**。中间层有时也被称为**隐藏层**。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8ad8b4cc7e4a388fbc84e8b36c4b6a71.png)

#### 激活函数（activation function）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f62a00e17103807036e95ce09adb7b5c.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cd94cd50b13c3bbbb2b01ea6360ded20.png)  
在这里，输入信号的总和会被函数h(x)转换，转换后的值就是输出信号y，这种函数一般称为**激活函数(activation function)**。  
激活函数的作用在于决定如何来激活输入信号的总和。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3a7f761dbaa109aee78efd1b911d6471.png)

##### 阶跃函数

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a30af3e2eb0141ec4fd7b48f3d31c41a.png)  
h(x)表示的激活函数以阀值为界，一旦输入超过阀值，就切换输出。这样的函数称为**阶跃函数**。如果把激活函数从阶跃函数换成其他的函数，就可以进入神经网络的世界了。

**阶跃函数的实现：**

```python
def step_function(x):
	y = x>0
	return y.astype(np.int)
```

y是一个布尔类型的数组，x大于0的元素被转换成True，小于等于0的元素被转换为False。  
astype()通过参数指定期望的类型，这个代码中是np.int型。Python将布尔型转为int型后，True会转换为1,False会转换成0,从而得到了阶跃函数的输出。

**阶跃函数的图形：**

```python
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	return np.array(x > 0, dtype = np.int)

x = np.array(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # limit the scale of y
plt.show()
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3ac7a9a3c89240759d243ae094a3c91f.png)

##### Sigmoid函数

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/443960c7fd7a1a38a66657cd62e15ed8.png)  
**Sigmoid 的实现:**

```python
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
```

**Sigmoid的图形:**

```python
x = np.array(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # limit the scale of 
plt.show()
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5c81ad0dcbb693b84c7268008bea9b29.png)

##### Sigmoid 函数与阶跃函数的不同与相同

**不同:**  
平滑性不同，Sigmoid是一条平滑的曲线，输出随着输入发生连续性的变化。  
感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号。  
**相同:**  
两者都属于非线性函数.

**PS: 神经网络的激活函数必须使用非线性函数**

##### ReLU函数

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/492c3e2e390563fef79e2b3405d3d819.png)  
ReLU函数在输入大于0时，直接输出该值；在输入小于等于0时，输出0。

**ReLU的实现:**

```python
def relu(x):
	return np.maximum(0,x)
```

**ReLU的图形:**  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3670e729ca0d913df9df6cf33a5aab6a.png)

#### 3层神经网络的实现

给出一个3层神经网络如下：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2e00ba3d35d9abdf5c45c8ab3614b885.png)

##### 各层间信号传递的实现

###### 从输入层到第1层

![从输入层到第1层的信号传递](https://i-blog.csdnimg.cn/blog_migrate/42d98ac19c871ce51141fd9259a41200.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3ad66201dfb7c94e73ad9618e82d793d.png)  
下面我们把输入信号，权重，偏置设置为任意值，并进行实现

```python
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
```

###### 从第1层到第2层

![从第1层到第2层的信号传递](https://i-blog.csdnimg.cn/blog_migrate/9a174386d39185663b052c5edce59e58.png)

```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
```

###### 从第2层到输出层

![从第2层到输出层的信号传递](https://i-blog.csdnimg.cn/blog_migrate/5ff4bd8be4ce3055ceba9bbf47d762e5.png)

```python
def identity_function(x):
	return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2,W3) + B3
Y = identity_function(A3)#用这个函数或者直接写Y = A3
```

这里的identity\_function()函数又称为**恒等函数**，它会把输入按照原样输出，我们把它用作输出层的激活函数  
输出层的激活函数用\*\*σ()\*\*表示，不同于隐藏层的激活函数h()

**PS:** 输出层所采用的激活函数要根据求解问题的性质决定。一般地，回归问题用恒等函数，二元分类问题用Sigmoid函数，多元分类问题可以用Softmax函数

#### 代码实现小结

```python
def init_network():
	network = {}
	network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	network['b1'] = np.array([0.1, 0.2, 0.3])
	network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
	network['b2'] = np.array([0.1, 0.2])
	network['W3'] = np.array([0.1, 0.3], [0.2, 0.4])
	network['b3'] = np.array([0.1, 0.2])

	return network

def forward(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x,W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(x,W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(x,W3) + b3
	y = identity_function(a3)

	return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network,x)
print(y) #[0.31682708, 0.69627909]
```

**forward**一词表示从输入到输出方向的传递处理  
**backward**一词表示从输出方向到输入方向的处理

#### 输出层的设计

一般地，回归问题用恒等函数，二元分类问题用Sigmoid函数，多元分类问题可以用Softmax函数  
输出层的激活函数用\*\*σ()\*\*表示，不同于隐藏层的激活函数h()

##### 恒等函数

它会把输入按照原样输出

##### softmax函数

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e0101787dd74b966acfc534ff1227f17.png)  
yk表示第k个神经元的输出  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8cc6ae7f300d9fe2107a557b61c7f574.png)  
**softmax函数的实现**

```python
def softmax(a):
	c= np.max(a)
	exp_a = np.exp(a - c) #让输入信号最大值a参与计算是为了避免溢出
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y
```

**softmax函数的特征**  
softmax函数的输出是**0.0到1.0**之间的实数  
softmax函数的**输出值总和是1**，所以我们可以把它的输出解释为**概率**  
即使使用了softmax函数，各个元素之间的大小关系也不会改变。一般而言，神经网络只把输出之最大的神经元所对应的类别作为识别结果，所以神经网络在分类时，输出层的softmax函数可以**省略**

##### 机器学习问题的步骤

**一 学习**  
学习又称为训练，是为了强调从数据中学习模型。  
**二 推理**  
用学到的模型对未知的数据进行推理。

#### 手写数字识别

##### MNIST数据集

它是由0到9的数字图像构成的。训练图像有6万张，测试图像有1万张，这些图像可以用于学习和推理。MNIST图像数据是28×28像素的灰度图像（1通道），各个像素的取值在0到255之间。每个图像数据都相应的标有“7”，“2”，“1”等标签。

可以按照以下方式读入MNIST数据：

```python
import sys, os
sys.path.append(os.pardir) #为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#输出各个数据的形状
print(x_train.shape) #(60000,784)
print(t_train.shape) #(60000,)
print(x_test.shape) #(10000,784)
print(t_test.shape) #(10000,)
```

load\_mnist函数以（训练图像，训练标签），（测试图像，测试标签）的返回形式读入的MNIST数据。其中可以设置三个参数。

1.  **normalize**设置是否将输入图像正规化为0.0到1.0的值。若设置为False，则输入图像的像素会保持原来的0~255。
2.  **flatten**设置是否展开输入图像（即是否将其展开变为一维数组），若设置为False，则输入图像为1×28×28的三维数组；若设置为 True，则输入图像会保存为由784个元素构成的一维数组。
3.  **one\_hot\_label**设置是否将标签保存为one-hot表示（one-hot representation），它的意思是仅正确解标签为1，其余皆为0的数组，就像\[0,0,1,0,0,0,0,0,0,0\]这样。当one\_hot\_label为false时，只是像7、2这样简单的保存正确解标签；True时，则保存为one-hot表示。

尝试显示训练图像的第一张：

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image # PIL是Python Image Library 模块

def img_show(img):
	pil_img = Image.fromarray(np.uint8(img)) #需要把Numpy保存的图像数据转换为PIL用的数据对象，这个转换用Image.fromarray()来完成
	pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label) #5

print(image.shape) #（784，）
img = img.reshape(28,28) #把图像变为原来的尺寸,reshape()方法的参数制定期望的形状
print(img.shape) #(28,28)

img_show(img)
```

##### 神经网络的推理处理

该神经网络的输入层有784个神经元，输出层有10个神经元。2个隐藏层，第1个隐藏层有50个神经元，第2个隐藏层有100个神经元。

```python
#先定义3个函数
def get_data():
	(x_train, t_train), (x_test, t_test) = \ #\是换行的意思
		load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test

def init_network():
	with open("sample_weight.pkl", 'rb') as f:
		network = pickle.load(f)
	return network
#init_network会读入保存在pickle文件sample_weight_pkl中的学习到的权重参数

def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x,W1) +b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1,W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2,W3) + b3
	y = softmax(a3)

	return y


#评价它的识别精度（accuracy），即能在多大程度上分类
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y) #获取概率最高的元素的索引
	if p == t[i]:
		accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
```

把数据限定到某个范围内的处理称为**正规化（normalization）**  
将数据整体的分布形状均匀化的方法称为**数据白化（whitening）**  
对神经网络的输入数据进行某种既定的转换称为**预处理（pre-processing）**，正规化属于预处理的一种

##### 批处理

打包式的输入数据称为**批(batch)**，批处理可以使计算机计算的更快

以下是基于批处理的代码实现：

```python
x, t = get_data()
network = init_network()

batch_size = 100 #批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
	x_batch = x[i:i+batch_size]
	y_batch = predict(network, x_batch)
	p = np.argmax(y_batch, axis=1)
	accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
```

这里采用argmax()获取值最大的元素的索引。  
axis=1表示在100×100的数组中，沿着第1维方向找到值最大的元素的索引。  
\*\*PS：\*\*矩阵的第0维是列方向，第1维是行方向。

### 神经网络的学习

**“学习”**指的是从训练数据中*自动*获取最优权重参数的过程，而**学习的目的**就是以*损失函数*为基准，找出能使它的值达到*最小*的权重参数。

如果从0想出一个可以识别数字5的算法是很难的，不如考虑如何有效利用数据来解决这个问题。一种方案是：先从图像中提取特征量，再用机器学习技术学习这些特征量的模式。

**特征量：**指可以从输入数据中准确地提取本质数据（即重要数据）的转换器。常用的特征量有SIFT、SURF、HOG等。使用这些特征量将图像数据转换为向量，然后对转换后的向量 使用机器学习中的SVM、KNN等**分类器**进行学习。对于不同的问题必须使用合适的特征量才能得到好的结果。在神经网络中，连数据中包含的重要特征量也都是由机器来进行学习的，不存在人为介入。

#### 训练数据与测试数据

机器学习中，一般将数据分为**训练数据（也称为监督数据）和测试数据**两部分来进行学习和实验。  
首先使用训练数据进行学习，寻找最优的参数；然后使用测试数据评价训练得到的模型的实际能力。  
我们所追求的最终目标是模型的**泛化能力**，泛化能力是指处理未被观察过的数据（即不包含在训练数据中的数据）的能力。

#### 损失函数（loss function）

**损失函数**表示当前的神经网络对监督数据在多大程度上不拟合，即在多大程度上不一致。一般使用\*\*均方误差（mean squared error）和交叉熵误差（cross entrophy error）\*\*等。

##### 均方误差（mean squared error）

![yk表示神经元的输出，tk表示监督数据，k表示数据的维度](https://i-blog.csdnimg.cn/blog_migrate/7f58ba26862b88dbfc5c9b9f712cc782.png)  
实现代码：

```python
def mean_squared_error(y, t):
	return 0.5 * np.sum((y-t)**2)
```

##### 交叉熵误差（cross entrophy error）

![yk表示神经元的输出，tk表示监督数据，k表示数据的维度](https://i-blog.csdnimg.cn/blog_migrate/adee7e90f88ff05f78351939c881cab9.png)  
交叉熵误差的值是由正确解标签所对应的输出结果决定的。  
代码实现：

```python
def cross_entrophy_error(y, t):
	delta = 1e-7 #这里加上一个微小值delta可以防止负无限大的发生
	return -np.sum(t * np.log(y+delta))
```

##### mini-batch学习

机器学习的任务是针对训练数据计算损失函数的值，找出使该值尽可能小的参数。  
**PS:** 计算损失函数必须把**所有**的训练数据作为对象，即若训练数据有100个的话，我们就要把这**100个损失函数的总和**作为学习的指标。最后要除以N进行正规化，通过除以N，可以求单个数据的\*\*“平均损失函数”\*\*。

\*\*mini-batch学习：\*\*从训练数据中选出一批数据称为mini-batch数据，然后对每个mini-batch进行学习。

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \ #\是换行的意思
		load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0] #0行1列
batch_size = 10
#使用np.random.choice()随机抽取数据
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

##### mini-batch版交叉熵误差的实现

实现一个可以同时处理单个数据和批量数据（数据作为batch集中输入）两种情况的函数如下，y是神经网络的输出，t是监督数据。y的维度为1时，即求单个数据的交叉熵误差时，需要改变数据的形状。并且当输入为mini-batch时，要用batch的个数进行正规化，计算单个数据的平均交叉熵误差：

```python
def cross_entrophy_error(y, t):
	if y.ndim ==1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

当监督数据是标签形式（非one-hot表示，而是像“2”，“7”这样的标签）时，交叉熵误差可以通过如下代码实现：

```python
def cross_entrophy_error(y, t):
	if y.ndim ==1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

对权重参数的损失函数求导，表示的是\*“如果稍微改变这个权重参数的值，损失函数将如何变化”\*。如果导数的值为负，通过使该权重参数向正方向改变，可以减小损失函数的值；反过来，如果导数的值为正，则通过使该权重参数向负方向改变，可以减小损失函数的值。

#### 数值微分（numerical differentiation）

\*\*舍入误差：\*\*指因为省略小数的精细部分的数值，如小数点后第8位以后的数值，而造成最终的计算结果上的误差。  
\*\*中心差分：\*\*以x为中心，计算函数 f 在（x+h）和（x-h）之间的差分  
\*\*前向差分：\*\*计算 x 和 （x+h）之间的差分

#### 梯度（gradient）

由全部变量的偏导数汇总而成的向量称为**梯度（gradient）**，梯度指示的方向是各点处的*函数值减小最多*的方向，但是无法保证梯度所指的方向就是函数的最小值或者真正应该前进的方向。

梯度的实现如下：

```python
#基于数值微分计算参数的梯度
def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x) #生成和x相同形状的数组

	for idx in range(x.size):
		tmp_val = x[idx]
		#f(x+h)的计算
		x[idx] = tmp_val + h
		fxh1 = f(x)
		
		#f(x-h)的计算
		x[idx] = tmp_val - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val #还原值
		
```

##### 梯度法

**最优参数：**损失函数取得最小值时的参数  
**梯度法（gradient method）：**函数的取值从当前位置沿着梯度方向前进一定距离，然后在新的地方重新求梯度 ，再沿着新梯度的方向前进，如此反复，不断地沿着梯度方向前进。像这样不断地沿梯度方向前进，逐渐减小函数值的过程就是梯度法。  
寻找最小值的梯度法称为**梯度下降法（gradient descent method）**，寻找最大值的梯度法称为**梯度上升法（gradient ascend method）**。

用数学式来表示梯度法如下：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b2091dcbc6f57558d56a3d3743272322.png)  
这个式子代表更新一次，这个算式会反复执行。

η是**学习率（learning rate）**，它决定在一次学习中，应该学习多少，以及在多大程度上更新参数。  
学习率一般会事先确认为某个值，这个值过大过小都无法抵达一个“好位置”。学习过程中，一般会一边改变学习率的值，以便确认学习是否正确进行了。

梯度下降法的实现如下：

```python
def gradient_desecnd(f, init_x, lr=0.01, step_num=100): 
	x = init_x # init_x是初始值
	for i in range(step_num): #step_num是要重复的次数
		grad = numerical_gradient(f, x) #f是要进行最优化的函数
		x -= lr * grad
	return x
```

\*\*超参数：\*\*机器学习中在训练模型前需要设置的参数，而不是通过训练学习到的参数。需要人工进行设定。一般来说，超参数需要尝试多个值，以便找到一种可以使学习顺利进行的设定。常见的超参数有：学习率，批处理大小等。

##### 学习算法的实现

神经网络学习的步骤：

1.  mini-batch：从训练数据中随机选出一部分数据，这部分数据称为mini-batch，我们的目标是减小mini-batch的损失函数的值。
2.  计算梯度：为了减小mini-batch的损失函数的值，需要计算出各个权重参数的梯度。梯度表示损失函数的值减小最多的方向。
3.  更新参数：将权重参数沿着梯度方向进行微小更新。
4.  重复以上3个步骤。

**随机梯度下降法/SGD（stochastic gradient descend）**：对随机选择的数据进行的梯度下降法。

实现2层神经网络TwoLayerNet类如下：

```python
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		#初始化权重
		self.params = {}
		self.params['W1'] = weight_init_std * \
							np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * \
							np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
	
	def predict(self, x):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']

		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y

	#x是输入数据，t是监督数据
	def loss(self, x, t):
		y = self.predict(x)

		return cross_entropy_error(y, t)

	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)

		accuracy = np.sum(y==t) / float(x.shape[0])
		return accuracy

	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t) #这个 lambda 表达式接受一个参数 W，并调用类中的 self.loss(x, t) 方法，其中 x 和 t 是该类中的成员变量。这个 lambda 表达式的目的是创建一个与权重矩阵 W 相关的损失函数。
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads
```

##### mini-batch的实现

\*\*过拟合（over-fitting）：\*\*指虽然训练数据中的数字图像能被正确识别，但是不在训练数据中的数字图像却无法被识别的现象  
\*\*epoch：\*\*是一个单位，一个epoch是指学习中 所有训练数据均被使用过一次时的更新次数。如：对于10000笔训练数据，用大小为100笔数据的mini-batch进行学习时，重复随机梯度下降法100次，所有的数据就都被“看过”了。此时，100次就是一个epoch。

mini-batch实现如下：

```python
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
#平均每个epoch的重复次数！！
iter_per_epoch = max(train_size / batch_size, 1)

#超参数
iters_num = 10000
batch_size = 100
learning_rate = 0.1

network TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
	#获取mini-batch
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	#计算梯度
	grad = network.numerical_gradient(x_batch, t_batch)
	#更新参数
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)
	#计算每一个epoch的识别精度
	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc, test acc | " + str(train_acc)+ ", " + str(test_acc))

```

### 误差反向传播

#### 反向传播

将信号E乘以节点的局部导数，然后将结果传递给下一个节点。这里说的局部导数是指正向传播中y=f(x)的导数。  
比如：假设y=f(x)=x^2，则局部导数为2x。把这个局部导数乘以上游传过来的值（本例子中为E），然后传递给前面的节点。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e980afb1cbb3b662899d85222ac8afc4.png)

##### 加法节点的反向传播

因为加法节点的反向传播只乘以1，所以输入的值会原封不动的流向下一个节点。  
例子：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/833e50eebd135f6cb27d6e708b59c251.png)

##### 乘法节点的反向传播

乘法的反向传播会将上游的值乘以正向传播时的输入信号的“翻转值”后传给下游。翻转值代表一种翻转关系。例如，正向传播时信号是x的话，反向传播时则是y；正向传播时信号是y的话，反向传播则是x。因此，在实现乘法节点的反向传播时，要保存正向传播的输入信号。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bcc1c6a003dbfdf3a741faadb48fbe03.png)  
例子：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/df79d19ae8c9c271c7f1f09c140bd4a4.png)

##### 乘法层和加法层的实现

**乘法层的实现如下：**

```python
class MulLayer:
	def __init__(self):
		self.x = None
		self.y = None

	def forward(self, x, y):
		self.x = x
		self.y = y
		out = x * y
		return out

	def backward(self, dout):
		dx = dout * self.y
		dy = dout * self.x
		return dx, dy
```

**加法层的实现如下：**

```python
class AddLayer:
	def __init__(self):
		pass

	def forward(self, x, y):
		out = x + y
		return out

	def backward(self, dout):
		dx = dout * 1
		dy = dout * 1
		return dx, dy
```

举例实现以下计算图的过程：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0a9e2f8c5d171f6a2ff40768d3851f51.png)

```python
apple = 100
apple_num  = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_price_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_price_layer.backward(dapple_price)

print(price)
print(dapple_num, dapple, dorange_num, dorange, dtax)
```

##### 激活函数层反向传播的实现

###### ReLU的实现

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e3c12e4537c91df38fc2cf612c1115de.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4a231b5f115c9e96fd5f3e99e81d97c6.png)

###### Sigmoid的实现

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/00f7c3e660d4b0a5b31d57b5dac73273.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e309fcfb25b19a3911d2133f19a86385.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/14cc65b078661e999cc669d44261eabf.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/08ae3bb7a78f21937cb201252bdad46a.png)

##### Affine层 / Softmax层的实现

###### Affine层的实现

Affine层即全连接层，实现过程如下：

```python
class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None
	def forward(self, x):
		self.x = x
		out = np.dot(x, self.W) + self.b
	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)

		return dx
```

###### Soft-With-Loss层

softmax函数会将输入值正规化（即将输出值的和调整为1,所以它的输出也可以看作看作概率）后再输出。例子如下：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5c786da36d5558e90e098599175c3352.png)  
神经网络的推理通常不使用Softmax层，因为当神经网络的推理只需要给出一个答案的情况下，因为此时只对得分最大值感兴趣，所以不需要Softmax层。但是神经网络的学习阶段则需要Softmax层。

简易版Softmax-with-Loss层的计算图如下：  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/27404cb192ce1c801b2f6f6f3915583c.png)  
(y1, y2, y3)是softmax层的输出，(t1, t2, t3)是监督数据。所以(y1-t1, y2-t2, y3-t3)是Softmax层的输出和标签的差分。  
神经网络的反向传播会把这个差分表示的误差传递给前面的层，这是神经网络学习中的重要性质。  
神经网络学习的目的就是通过调整权重参数，使神经网络的输出，即Softmax的输出接近教师标签。

Softmax-with-Loss层的实现过程如下：

```python
class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None #损失
		self.y = None #softmax的输出
		self.t = None #监督数据(one-hot vector)

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_extropy_error(self.y, self.t)

		return self.loss

	def backward(self, x, t):
		batch_size = self.t.shape[0]#shape[0]返回的是第一维度（行）的数量
		dx = (self.y - self.t) / batch_size

		return dx
```

#### 误差反向传播法的实现

##### 神经网络学习的步骤

前提：神经网络中有合适的权重和偏置，调整权重和偏置这些参数以拟合训练数据的过程叫做学习。过程如下：

1.  mini-batch：从训练数据中随机选择一部分数据
2.  计算梯度：计算损失函数关于各个权重参数的梯度
3.  更新参数：将权重参数沿着梯度方向进行微小的更新
4.  重复以上3个步骤

以下的代码使用了层，通过使用层，获得识别结果的处理（predict()）和计算梯度的处理（gradient()）只需要通过层之间的传递就能完成。下面是TwoLayer的代码实现：

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		#初始化权重
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

		#生成层
		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.lastLayer = SoftmaxWithLoss()
	
	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)

		return x

	#x是输入数据，t是监督数据
	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis = 1)
		if t.ndim != 1 : np.argmax(t, axis=1)
		accuracy = np.sum(y==t) / float(x.shape[0])
		return accuracy

	#x是输入数据，t是监督数据
	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads

	def gradient(self, x, t):
		#forward
		self.loss(x, t)

		#backward
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers = reverse()
		for layer in layers:
			dout =layer.backward(dout)

		grads = {}
		grads['W1'] = self.layers['Affine1'].dW
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db

		return grads
```

OrderedDict是有序字典，意思就是它可以记住向字典里添加元素的顺序。因此，神经网络的正向传播只需要按照添加元素的顺序调用各层的forward方法就可以完成处理，而反向传播只需要按照相反的顺序调用各层即可。

##### 梯度确认

目前知道两种求梯度的方法。基于数值微分的方法和使用误差反向传播的方法。  
数值微分很费时间，但是在确认误差反向传播法的实现是否正确时，需要用到数值微分。

\*\*梯度确认：\*\*确认数值微分求出的梯度结果和误差反向传播求出的结果是否一致的操作叫做梯度确认。

梯度确认的代码实现如下：

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
	diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
	print(key + ":" + str(diff))
```

##### 使用误差反向传播法的学习

实现代码如下：

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iter_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	#通过误差反向传播法求梯度
	grad = network.gradient(x_batch, t_batch)

	#更新
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.aprmas[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print(train_acc, test_acc)
```

### 与学习相关的技巧

#### 参数的更新

\*\*最优化（optimization）：\*\*神经网络学习的目的是找到使得损失函数的值尽可能小的参数。解决这个问题的过程叫做最优化。

具体使用哪种方法更好，需要具体情况具体分析。

##### 随机梯度下降法（SGD）

使用参数的梯度，沿梯度的方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，这个过程成为SGD。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/91e4eca30cf7b93e96ff8b913f6846e1.png)  
W是需要更新的权重参数，L是损失函数，η是学习率。  
将SGD定义为一个Python类如下：

```python
class SGD:
	def __init__(self, lr=0.01):
		self.lr = lr

	def update(self, params, grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key]
```

\*\*SGD的缺点：\*\*虽然它很简单容易实现，但是在解决某些问题时可能没有效率。低效的根本原因是：梯度的方向并没有指向最小值的方向。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9fccf8b2575b93ac3bea5cad562bf794.png)

##### Momentum

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4c08f45f53c2619be906f4dc46d6d3a2.png)  
v对应物理上的速度，代码实现如下：

```python
class Momentum:
	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None

	def update(self, params, grads):
		if self.v is None:
			self.v = {}
			for key, val in params.items():
				self.v[key] = np.zeros_like(val)

		for key in params.keys():
			self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
			params[key] += self.v[key]
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9f9ebffa5bfc09084980fbc30ed8d12b.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c6a9cb6b23a7cfc56d467919348820a5.png)

##### AdaGrad

\*\*学习率衰减（learning rate decay）：\*\*随着学习的进行，使学习率逐渐减小。  
学习率η的值很重要。学习率过小，会导致学习花费过多时间；反过来，学习率过大，则会导致学习发散而不能正确进行。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/53a8d5ed4f7bb5c04bb504d20b1f477a.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/27ded121a720553086e0d537b56e7004.png)  
AdaGrad会记录过去所有梯度的平方和。因此，学习越深入，更新幅度就越小。  
实际上，若无止境的学习，更新量就会变成0，完全不再更新。为了改善这个问题，可以使用RMSProp方法，这个方法并不是将过去所有的梯度一视同仁的相加，而是逐渐的遗忘过去的梯度，在做加法运算时将新梯度的信息更多的反映出来。

实现过程如下：

```python
class AdaGrad:
	def __init__(self, lr=0.01):
		self.lr = lr
		self.h = None

	def update(self, params, grads):
		if self.h is None:
		self.h = {}
		for key, val in params.item():
			self.h[key] = np.zeros_like(val)

	for key in params.keys():
		self.h[key] += grads[key] * grads[key]
		params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
		#最后一行加上了1e-7是为了防止当self.h[key]中有0时，将0用作除数的情况
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0645b0704b8771f17c1ea96a4c1e660c.png)

##### Adam

Adam将Momentum和AdaGrad融合起来了。Adam一般有3个参数，学习率、momemtum系数β1和二次momemtum系数β2  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/be01f3dfe1abfad5c0940b7f3bcb1c15.png)

#### 权重的初始值

抑制过拟合、提高泛化能力的技巧——**权值衰减（weight decay）**，它是一种以减小权重参数的值为目的进行学习的方法，通过减小权重参数的值来抑制过拟合的发生。

如何想减小权重的值，一开始就将初始值设置为较小的值才是正途。

\*\*梯度消失（gradient vanishing）：\*\*偏向0和1的数据分布会造成反向传播中梯度的值不断变小，最后消失。

作为激活函数的函数最好具有关于原点对称的性质。tanh函数是关于原点（0, 0）对称的S型曲线，sigmoid函数是关于（x, y） =（0, 0.5）对称的s型曲线。

当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，也称为“He初始值”。

#### Batch Normalization算法

**Batch Norm的优点：**  
可以增大学习率  
不那么依赖初始值  
抑制过拟合

\*\*Batch Norm的思路：\*\*调整各层的激活值分布使其拥有适当的广度，为此，向神经网络中插入对数据分布进行正规化的层，即Batch Normalization层。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4820167fcf6e5b0242a83e5dd0b09ae4.png)  
Batch Norm顾名思义，以进行学习时的mini-batch为单位，按照mini-batch进行正规化。即进行使数据分布的均值为0，方差为1的正规化。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/18764e97cc46cb15c9c141fddbecbdeb.png)  
通过将这个处理插入到激活函数的前面或者后面，可以减小数据分布的偏向。  
在不使用Batch Norm的情况下，如果不赋予一个尺度好的初始值，学习将完全无法进行。

#### 正则化

\*\*过拟合：\*\*指的是只能拟合训练数据，但不能很好地拟合不包含在训练数据中的其他数据的状态。

**过拟合的原因：**  
训练模型拥有大量参数、表现力强  
训练数据少

##### 权值衰减

这是一种用来抑制过拟合的办法，该方法通过在学习的过程中对大的权重进行惩罚，来抑制过拟合。很多过拟合原本就是因为权重参数取值过大才发生的。

##### Dropout

这是一种在学习的过程中随机删除神经元的方法。训练时，随机选出隐藏层的神经元，然后将其删除，被删除的神经元不再进行信号的传递。

\*\*集成学习：\*\*就是让多个模型单独进行学习，推理时再取多个模型的输出的平均值。

#### 超参数的验证

\*\*超参数(hyper-parameter)：\*\*除了权重和偏置等参数，比如：神经元数量、batch大小、参数更新时的学习率或权值衰减等。如果这些超参数没有设置合适的值，模型的性能就会很差。这部分尽可能高效的寻找超参数。

在对超参数进行验证的时候，不能使用测试数据评估超参数的性能，因为如果使用测试数据调整超参数，超参数的值会对测试数据发生过拟合。  
因此，调整超参数的时候，必须使用超参数专用的确认数据。  
用于调整超参数的数据，一般称为**验证数据（validation data）**

### 卷积神经网络(Convolutional Neural Network / CNN)

CNN经常被用于图像识别、语音识别等各种场合。

#### 整体结构

\*\*全连接（fully-connected）：\*\*神经网络中，相邻层的所有神经元之间都有连接。Affine层和全连接层（Fully Connected Layer）通常在深度学习中是等价的概念。  
![基于全连接层（Affine层）的网络的例子](https://i-blog.csdnimg.cn/blog_migrate/8ea1d9082fdbf4f2a1d4c450f7fbd61a.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/feb31e0242e033e1fb809d8916bbd1f0.png)

CNN中新增加了Convolution层和Pooling层。CNN的连接顺序是“Convolution-ReLU-(Pooling)”，Pooling层有时候会被省略。  
此外，靠近输出的层中使用了之前的“Affine-Softmax”组合，这些都是一般的CNN中比较常见的结构。

#### 卷积层

##### 全连接层存在的问题

全连接层存在的问题是数据的**形状被忽视了**。比如，输入的数据是图像时，图像通常是**高、长、通道**方向上的3维形状。但是向全连接层输入时，需要将3维数据拉平为1维数据。实际上，前面提到的使用了MNIST数据集的例子中，输入图像就是1通道、高28像素、长28像素的（1.28,28）形状，但是却被排成1列，以784个数据的形式输入到最开始的Affine层。图像是3维形状，这个形状中应该含有重要的空间信息。

而**卷积层可以保持形状不变**。

CNN中，有时将卷积层的输入输出数据称为**特征图 （feature map）**。卷积层的输入数据称为**输入特征图（input feature map）**，输出数据称为**输出特征图（output feature map）**。

##### 卷积运算

卷积层进行的运算就是卷积运算。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/16641c6cd70590f885e7e70f27153f71.png)  
卷积运算对输入数据应用滤波器。

对于输入数据，卷积运算以一定的间隔滑动滤波器的窗口并应用（即下图中的灰色部分）。将各个位置上的滤波器的元素和输入的对应元素相乘，然后再求和（有时将这个计算称为**乘积累加运算**）。然后，将这个结果保存到输出的对应位置。将这个过程在所有的位置都进行一遍，就可以得到卷积运算的输出。

滤波器的参数就对应之前的权重，此外，CNN中也存在偏置。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/565e1338b83ceb77234b8f0854621e29.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3c99ce01d9c18aaf4fbc384ca0ec840a.png)  
上图向应用了滤波器的数据加上了偏置。偏置通常只有1个，这个值会被加到应用了滤波器的所有元素上。

##### 填充（padding）

在进行卷积层的处理之前，有时要向输入数据的周围填入固定的数据（比如0），这称为填充。

下图对大小为4\*4的输入数据应用了**幅度为1的填充**（这是指用幅度为1像素的0填充周围）。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/04f8545b15b4a6ee990c1481bd38f95a.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a1ca12d82f64240b175f3f9685d5ff8b.png)  
使用填充**主要是为了调整输出的大小**，可实现在保持空间大小不变的情况下将数据传给下一层。

##### 步幅（stride）

应用滤波器的位置间隔被称为**步幅（stride）**。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/137dbfd0861981b139e3c101233df065.png)  
增大步幅后，输出会变小。增大填充后，输出会变大。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/70663841c62ef9e703595087587ab55c.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/24bc41b06dd1d4d008b1a5f3e644dd97.png)  
使用这个公式的时候，必须使他们同时分别可以除尽。

##### 3维数据的卷积运算

图像是3维数据，除了高和长方向外，还需要处理通道方向。

通道方向上有多个特征图时，会按通道进行输入数据和滤波器的卷积运算，并将结果相加，从而得到输出。输入数据和滤波器的通道要设置为相同的值。滤波器的大小要设置为任意值（不过，每个通道的滤波器大小要全部相同）。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c4c46ba51527e3ec2a82b5a5195728b0.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/db1a65a6ae8f9f28cce2d429ae5160a8.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/caa80e1f9cb5e04eb5385666fa962e53.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/651d19e19c7b193cef48e78665f45f73.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fbe24060efb216f88a20958beb3c4891.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4d3cc3834b4f004346b2388f60f4e42f.png)

##### 批处理

批处理可以能够实现处理的高效化和学习时对mini-batch的对应。在对卷积运算进行批处理时，需要将在各层间传递的数据保存为4维数据，就是按照（batch\_num, channel, height, width）的顺序保存数据。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a58fbf0237e6bde99418680606d09d26.png)

#### 池化层

下图是**Max池化**时的处理顺序，它是获取最大值的运算。一般来说，池化的窗口大小会和步幅设定成相同的值。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8995d8f7069a4eefa78efb05fa96531e.png)  
**池化层的特征：**  
\*\*没有要学习的参数：\*\*池化只是从目标区域取最大值（或平均值），所以不存在要学习的参数。  
\*\*通道数不发生变化：\*\*计算是按照通道独立进行的，所以输入和输出数据的通道数不会发生变化。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/56640e25fd6b64d50dac8a91056982b9.png)  
\*\*对微小的位置变化具有鲁棒性（健壮）：\*\*输入数据发生微小偏差时，池化仍会返回相同的结果。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ebab2e28b5bbccf0467e84972dbfa367.png)

#### 卷积层和池化层的实现

##### 卷积层的实现

**im2col**是一个函数，将输入数据展开以适合滤波器（权重）。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1897ee3d65841f3b23cc0b0a95df9b92.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6a7f554d03609766c789386b6a195287.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/244e93c36cb873e90f5cf7f985af0ba2.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6b1123bab6c3506f3d263d9db1f51d43.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/00cf7a9c9b3e192d0e7e13c42dac14c0.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9f50aed51a732bd30ff4fc9c892185ac.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ff3cfbbb973871d03c1edb30cb9ac677.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b34013a7ad7fcb6d4b7790ec29d8bdc9.png)  
这里用im2col展开输入数据，并用reshape将滤波器展开为2维数组。然后计算展开后的矩阵的乘积。  
通过在reshape时指定为-1，reshape函数会自动计算-1维度上的元素个数，以使多维数组的元素个数前后一致。比如，（10,3,5,5）形状的数组的元素共有750个，指定reshape(10,-1)后，就会转换成（10,75）形状的数组。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d650175db5df248f664771cfa0a5986e.png)  
**在进行卷积层的反向传播时，必须进行im2col的逆处理。**

##### 池化层的实现

池化层的实现和卷积层相同，都使用im2col展开输入数据。不过，池化的情况下，在通道方向上是独立的，这一点和卷积层不同。即：池化的应用区域按通道单独展开。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/41074cd8f27aebfec58f4e41282b108c.png)  
像这样展开后，只需要对展开的矩阵求各行的最大值，并转换为合适的形状即可。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e97bb5b01a7c7cfa726b3b8473a37bb5.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2dcc44b3dbefbb65b1750bb611f88586.png)  
np.max可以指定axis参数，并在这个参数指定的各个轴方向上求最大值。

#### CNN的实现

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2e0122c6ecae8f2d152b8262ee321276.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b45a796e1a5d34c1f7652a48f1e16496.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/98621baa0c6e06340105a3a44cd9e895.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e942da9286f9949e97ab657b8460c468.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ce877bb40043ee055c5e92495c867113.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/de7bb783e14936bf1071606717a212a9.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/89c73783f64a1d0eacce515e93aa5acc.png)

#### CNN的可视化

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/07e79828c8d076c630ed3a3b4d8a5c28.png)  
卷积层的滤波器会提取边缘或斑块等原始信息。

随着层次的加深，提取的信息（正确的讲，是反映强烈的神经元）也越来越抽象。

随着层次的加深，神经元从简单的形状向“高级”信息变化。换句话说，就像我们理解东西的含义一样，响应的对象在逐渐变化。

LeNet和AlexNet是CNN的代表性网络。

AlexNet网络结构堆叠了多层卷积层和池化层，最后经过全连接层输出结果。

### 深度学习

深度学习是加深了层的深度神经网络。基于之前介绍的网络，通过叠加层，就可以创建深度网络。

\*\*Data Augmentation（数据扩充）：\*\*该算法“人为的”扩充输入图像（训练图像）帮助提高精确度。即对于输入图像，通过施加旋转、垂直或水平方向上的移动等微小变化，增加图像的数量。这在数据集的图像数量有限时尤其有效。

加深层的好处是可以减少网络的参数数量。说的详细一点，就是与没有加深层的网络相比，加深了层的网络可以用更少的参数达到同等水平或更强的表现力。

不过，通过加深网络，就可以分层次地分解需要学习的问题。也可以分层次的传递信息。

对于大多数的问题，都可以期待通过加深网络来提高性能。

VGG、GoogleLeNet、ResNet等是几个著名的网络。