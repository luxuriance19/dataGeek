### 1、back propagation算法原理理解？
BP算法有两个计算过程，前向计算过程和后向传播的过程。  
算法流程：  
1、 初始化每一层网络的权重变量，将样本内误差Ein=0和每层的梯度置为0。  
2、for 每个样本（xn,yn),n = 1,...,N,做：  
     计算每层的X（前向过程）。
     计算每一层的delta（反向传播过程）。  
     将这个样本的误差加入样本误差Ein。  
     for 每层，计算：  
       梯度更新值。  
       更新梯度。  
       
BP算法的核心在与梯度的计算的过程，前向的过程当中，只需要按照初始化的权重来计算最后网络层的输出。然后再根据网络层输出的误差进行针对每一层的权重进行求导，根据
求得的导数更新权重的值。在这里，每一层的delta函数，例如第l层，也就是误差函数对l层的激活函数的导数（这里包含了多层的求导的递推过程）。
反向传播算法的一大优点是可以通过图来计算。详细参见Computing Gradient。


### sigmoid函数、tanh函数、ReLU函数的区别？各自的优缺点？对应的tf函数是？  
这三个函数都是激活函数，是为了引入非线性的因素，主要原因是因为线性模型的表达能力不够。

从PLA模型可知，在二维平面当中，只能够完全区分三个点的情况，但是二维平面取值可以有四组，即线性情况下无法可可分。  
实例： XOR模型 输入和输出的关系 [0,0],0; [0,1],1;[1,0],1;[1,1],0。根据这个问题，以最小化MSE为训练目标训练一个线性模型，出来的结果是y=0.5
所以我们需要一个非线性模型来转化这些特征 **这就是为什么引入被称为激活函数的固定非线性函数来实现这个目标**， 经过放射变换以后在进行非线性变换。
上面这个例子当中，可以先根据z = XW+C， W为2×2的矩阵，元素都为1， c=[0,-1]为2x1的列变量，然后经过max{0，z}的激活函数（非线性变化）最后得到数据为
[0,0],0; [0,1],1;[1,0],1;[2,1],0, 这个时候就可以根据异或进行运算了。  

通俗的理解：  
如果在神经网络当中没有非线性的激活函数，实际上神经网络每一层的变化都可以看成是一个放射变化，那么不管经过多少层神经网络的变化，输出都是输入的线性组合，
实际上这还是PLA实现的功能，也就是无法分开大于d+1维的数量的数据。 为了让数据可分，引入了非线性变化，然数据进行非线性的映射直至最终可以完全分开。  

激活函数的特点：  
* 非线性： 当激活函数是非线性的时候，两层的神经网络能够包含的目标集合就已经基本上是所有的函数了。但是如果不加激活函数，那么两层实际上还是线性变换，MLP实际上也就相当于单层的射精网络。
* 可微性： 根据神经网络的BP算法，可知神经网络需要通过梯度求导来求解最优的算法。所以只要是最终的优化方法是根据梯度求解的时候，这个性质是激活函数必须的。
* 单调性： 当激活函数是单调的时候，单层的神经网络可以保证是凸函数。(这个解释有点问题，需要进一步求证，这个条件不一定需要满足）
* f(x) 約等于 x： 当激活函数满足这个细腻改制的时候，如果参数的初始化是随机设置很小的值，神经网络的训练会比较高校，如果不满足这个性质，就需要用心设置。
* 输出值的方位： 当激活函数的输出值是**有限**的时候，基于梯度的优化方法会更加稳定，因为特征的表示受有限全值的影响;当激活函数输出是**无限**的时候，模型的训练会更加的高效，不会出现梯度消失的情况，不过此时对learning_rate的设计需要更加小心（需要更小的值).

激活函数：
**ReLU（Recitified linear unit）： max(0,z)**
（整流线性单元  max（0，z)）：
* 优点： * 易于优化，只要整流处于激活状态，导数都保持比较大并且一致，易于训练学习。相较于tanh和sigmoid对于随机梯度下降的收敛有巨大的加速作用。 因为是由于它的非线性， 非饱和的公式。
       * sigmoid和tanh需要计算指数等，计算复杂度高，ReLU只需要一个阈值就可以得到激活值。计算量比较小。
* 缺点:  不能通过基于梯度学习的方法学习那些使他们激活函数变为0的样本。  
        （详细版本）ReLU 在训练的时候很”脆弱”，容易导致神经元”坏死”。举个例子：由于 ReLU 在 x<0 时梯度为 0，这样就导致负的梯度在这个 ReLU 被置零，而且这个神经元有可能再也不会被任何数据激活。实际操作中，如果你的学习率 很大，那么很有可能你网络中的40%的神经元都坏死了。 当然，如果你设置了一个合适的较小的学习率，这个问题发生的情况其实也不会太频繁。
> tf.nn.relu(features, name=None)

基于这个原因下面有三个线性整流单元的扩展
max（0，z) + alpha\*min(0,z) 如果alpha=-1就是 **绝对值整流(absolute value rectification)** ， 对图像中的对象识别，也就是寻找在输入照明极性反转下不变的特征有意义。 alpha=0.01这类的比较小的值是 **渗漏整流线性单元(Leaky ReLU)** 。 将alpha作为一个学习参数是 **参数化整流线性单元(parametric ReLU)** 。 
整流线性耽于那的另外一个扩展
**maxout单元** 将z分为魅族具有k个值的组，输出每组中最大的元素。 
)
* 优点： 因为可以学习多达k段的分段线性的凸函数，所以可以视为学习激活函数本身而不仅仅是短圆之间的关系，使用足够大的k，可以通过任意的精确度来近似任何凸函数。具有两块地maxout层可以学习实现和传统层相同地输入x的函数。 
* 缺点： 因为需要k个权重向量参数化，所以需要比整流线性单元更多的正则化。如果训练集很大，每个单元的块数保持比较低，那么可以再没有正则的情况下工作不错。
> tf.contrib.layers.maxout(inputs, num_units,axis=-1,scope=None)

重点介绍： 
**渗漏整流线性单元(Leaky ReLU)**
Leaky ReLU 是为解决 “ReLU死亡” 问题的尝试。ReLU 中当x<0 时，函数值为0。而="" leaky="" relu="" 则是给出一个很小的负数梯度值，比如="" 0.01。这样，既修正了数据分布，又保留了一些负轴的值，使得负轴信息不会全部丢失。="" 公式：f(x)="α"x(x=""<=""0),="" f(x)="x(x">=0),α 是一个小常数。
扩展：* PReLU. 对于 Leaky ReLU 中的 α，通常都是通过先验知识人工赋值的。Parametric ReLU 是将 α 作为参数训练，取得的效果还不错。
      * Randomized Leaky ReLU. 其是 Leaky ReLU 的 random 版本,在训练过程中，α 是从一个高斯分布中随机出来的，然后再在测试过程中进行修正。
> tf.nn.leaky_relu(features,alpha=0.2, name=None)

** Maxout** （需加深理解） 
Maxout 是对 ReLU 和 Leaky ReLU 的一般化归纳，它的函数是：max(w1Tx+b1,w2Tx+b2)。Maxout 神经元就拥有 ReLU 单元的所有优点（线性操作和不饱和），而没有它的缺点（死亡的 ReLU 单元）。Maxout 的拟合能力是非常强的，它可以拟合任意的的凸函数。缺点是参数 Double 了,也就是参数加倍了。  

**sigmoid函数：**  
* 优点： 输入任意实属值，将其变换到(0,1)区间，大的负数映射为0，大的正数映射为1。
* 缺点： * sigmoid函数存在饱和区间，所以容易出现梯度消失的问题。 当输入非常大或者非常小的时候， 根据sigmoid函数的图片可以看出，神经元的梯度是接近0的，这          样权重基本上不会更新。 所以如果初始化权重过大， 那么大多数神经元将会饱和，导致神经网络几乎不学习。 **所以不鼓励sigmoid作为前馈神经网络中的隐藏单元, 在循环网络，许多概率模型以及一些自编码器当中，因为不能够使用分段线性激活函数，所以sigmoid的使用比较常见，尽管它存在饱和的问题**
       * sigmoid的输出不是0均值的， 这回导致后层的神经元的输入不是0均值的信号，这会对后面梯度的计算产生影响。 例如：假设后面的神经元的输入都是正值， 那么对权值w的局部求导就都是正值，那么反向传播的过程当中，w都会朝着一个方向更新，也就是说要么正向更新，要么负向更新，使得收敛比较缓慢。
       * 幂计算相对而言比较耗时。
> tf函数： y = 1/(1+exp(-x))
> tf.sigmoid(x,name=None) /tf.nn.sigmoid

**tanh函数（双曲正切函数 hyperbolic tangent)：**
* 优点： 解决了sigmoid输出不是零均值的问题。  
* 缺点： 仍然具有饱和性问题。
> tf函数： y = 1/(1+exp(-x))
> tf.sigmoid(x,name=None) /tf.nn.sigmoid

**SELU**  
将输入自动normalization到均值为零方差为1。
* 优点：不同的特征的维度的尺度都是一样的，这样在梯度计算的时候，不会出现不同维度特征差别大导致梯度忽大忽小出现zig-zag的情况（出现这种情况的化，收敛会比较慢），所以会比较容易训练。也改进了会出现梯度消失的问题。
* 缺点：暂时未知，一般在SELU的输入前面会归一化处理。
selu(x) = lambda\*x (x>0)  
selu(x) = lambda\*(alpha\*exp(x) - alpha) (x<=0)  
lambda = 1.050700； alpha = 1.67326
> tf.nn.selu(features, name=None)

### softmax和cross_entropy原理解释？
* softmax： 在LR二分类问题当中，可以利用sigmoid函数将输入仿射空间映射到(0,1)的范围区间，从而得到预测类别的概率。将这个问题推广到多分类的问题当中，就可以使用softmax函数。对输出的值进行归一化处理。   

> 假设模型在softamx前面已经已经输出一个C维的向量，也就是预测的类别为C，如果输出层是全连接层，定义输出为a1,a2,...,ac。  
针对每个样本，它属于类别i的概率为：  
yi = exp(ai)/\sum_{k = 1}^{C} exp(ak)  
这样的处理保证了所有输出的概率和为1，也就是每个类别的和为1。  
对输出yi对aj求导：i = j 为yi(1-yi), i != j 为 yi*yi

**这个里面解决了为什么自己写的sigmoid_cross_entropy_with_logit的问题**  
因为在math.exp(x)有计算范围，当x的值过大或者过小的时候，输出都会是nan。  
所以我们可以针对这个问题给ai加一个常数F。
exp(ai+F) = Eexp(ai); E = exp(F),针对每个ai都加上F，yi的输出值不会改变。  
这个想法是想让所有的输入exp的值都在x的附近，所以F的取值可以是-max(a1,a2,...,aC)。这个做法的前提是假设所有的输入ai的值都比较接近，否则差异过大也只是变了符号而已。  
为了避免numpy移除的问题，可以采用bigfloat； eg： bigfloat.exp(x), 避免了exp(x)计算的溢出。

* cross_entropy
> 相对熵，相对熵也就是KL散度，用来衡量两个分布之间的距离：
$D_{KL}(p\lVert q) = begin{a}

### tf.placeholder(),tf.constant(),tf.Variable()的区别？ 
* tf.placeholder()：
> 顾名思义，这个是占位符号，用来传递训练样本给模型。当数据的维度不确定的时候，可以用None来确定。tf.placeholder()不需要指定初始值，但是需要指定初始的类型。
* tf.constant(value, dtype=None, shape=None, name=‘Const’, verify_shape=False)：
> 定义常量，如果定义了传递value的形状，也就是shape参数，value的shape不能够超过shape的大小，如果说value的长度小于shape定义的参数，那么多余的由传递的value的最后一个值填充。
* tf.Variable():
> 通过构造一个Variable类的实例在图中添加一个变量，主要作用在一些可训练的变量的定义上面，例如权重或者是截距偏置的值。
Variable类的构造函数要求一个variable的实例必须要有一个初始值，这个初始值可以是任意的形状和类型。这个变量的形状和类型由这个初始值确定。
变量的值可以被改变。  

### tf.Graph()概念理解？
因为神经网络大部分都是靠梯度来训练的，而BP算法的梯度计算可以通过图形来计算。  
tf.Graph()实际上就是定义了一个计算的流程图。一个节点代表一个操作，一条边代表了一个张量。通过边的连接决定了这些操作的流程的走向。  
通过这个图形的构建，可以定义最后变量的计算的走向，来优化需要优化的变量。

### tf.name_scope和tf.variable_scope()的理解？
* tf.name_scope(name,default_name=None,values=None)
是用户用来定义一个Python操作的内容管理器。这个内容管理器是用在图形内部的。在图形可视化的时候，同一个name_scope的操作会建立一个群组（group)来减少graph显示的复杂度。如果说在一个name_scope中定义同一个名字，那么tensorflow会自动补后缀。
* tf.variable_scope()  
是为了定义多个操作创建变量层的一个内容管理器。  
variable scope允许我们创建一个新的变量并且提供检查的功能。变量的作用域让我们在创建和使用变量的时候控制变量的重用，并且还允许我们用分层的方式来给变量命名。  
比如我们如果想每层都创建权重值w和偏置b，这个时候，如果说我们写了一个函数，经过不同的输入我们想要调用两次，程序执行就会出错。  
因为这个函数在第一次执行的时候，就会创建这个函数里面的变量，在第二次调用函数的时候，程序会不知道我们是想要创建新的变量还是使用援用的变量。所以函数就会起冲突。
变量域就是在这个时候起作用：我们通过不同的变量域，然后调用这个函数。
'
     ## 1:write a function to create a convolutional/relu layer:'
     def conv_relu(input,kernel_shape,bias_shape):
         #Create variable named "weights"
         weights = tf.get_variable("weights", kernel_shape, 
                                   initializer=tf.random_normal_initializer())
         # Create variable named "biases"
         biases = tf.get_variable("biases", bias_shape,
                                 initializer=tf.constant_initializer(0.0))
         conv = tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='SAME')
         return tf.nn.relu(conv+biases)

     input1 = tf.random_normal([1,10,10,32])
     input2 = tf.random_normal([1,20,20,32])
     x = conv_relu(input1, kernel_shape=[5,5,32,32],bias_shape=[32])
     x = conv_relu(input2, kernel_shape=[5,5,32,32],bias_shape=[32])# This fails
     # 第二句fail的原因，因为weights，biases变量已经存在，程序不知道我们当前是想使用
     # 原有的变量还是重新建立一个新的变量，所有操作冲突，无法执行。

     # 但是通过不同的scopes来调用conv_relu,就会明确我们是想要创建新的变量，那么程序就会work
     # this is working when we call conv_relu in different scope
     def my_image_filter(input_images):
         with tf.variable_scope("conv1"):
             # Variables created here will be named "conv1/weights", "conv1/biases"
             relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
         with tf.variable_scope("conv2"):
             # Variables created here will be named "conv2/weights", "conv2/biases".
             return conv_relu(relu1, [5, 5, 32, 32], [32])
'

### tf.variable_scope() 和tf.get_variable()的理解？
tf.get_variable()是创建变量最好的方式，这个函数要求你对创建的变量进行定义。这个名称将被其他副本用来访问同一个变量，以及检验和导出模型时命名这个变量的值。  
> 利用tf.get_variable()的方式创建变量只需要提供变量的名称和形状就可以了。在定义变量的时候，可以利用initializer参数对其进行初始化。这也时变量初始化最好的方式。

>tf.Variable()和tf.get_variable()异同：
相同点:都是定义一个变量。
不同点：tf.get_variable()通过变量名字来定义一个变量，创建变量的时候会由initializer的定义，可以直接初始化。tf.Variable()在定义的时候变量的名字可以不选择，通过变量的初始值来定义一个变量，返回一个initializer的操作。

### tf.global_variables_initializer() 什么时候使用？
在变量使用之前，必须要被初始化。 所以在模型的训练之前，就需要通过tf.global_variable_initializer()来初始化变量，这个函数会返回一个操作来初始化所有的变量。但是这个操作会忽略变量之间的相关性。所以如果由变量的依赖关系的时候，最后利用变量的initial_value()来初始化所有变量，避免报错。

### 学习中知识点的收获：
softmax的求导第一次计算，感受到了其中的神奇。tensorflow的一些操作还是有一点疑问，需要从实践中获取。曾经翻译过的tensorflow的API也算是派上了一点用场，然而赶不上google的更新速度，年前还没有中文翻译，现在中文翻译只是tutorial没有了。自己要加油学习。

参考文献：  
https://blog.csdn.net/cyh_24/article/details/50593400   
https://www.datageekers.com/t/zxca368-tf--week2/871  
http://www.cnblogs.com/tornadomeet/p/3428843.html  
https://data-sci.info/2017/06/11/%E6%9C%80%E6%96%B0%E6%BF%80%E6%B4%BB%E7%A5%9E%E7%B6%93%E5%85%83self-normalization-neural-network-selu/
https://github.com/shaohua0116/Activation-Visualization-Histogram  
https://zhuanlan.zhihu.com/p/27223959
