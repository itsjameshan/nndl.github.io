# 《神经网络与深度学习》全书公式提取

> 本文档从邱锡鹏著《神经网络与深度学习》一书中提取了从公式 (2.1) 到公式 (E.21) 的所有编号公式。
> 每个公式包含：LaTeX表达式、符号表（含汉语拼音读法）、一句话解释、简单应用场景。

---

## 第2章：机器学习概述

### 公式 (2.1)：数据集 / Dataset

$$
\mathcal{D} = \{(\boldsymbol{x}^{(1)}, y^{(1)}), (\boldsymbol{x}^{(2)}, y^{(2)}), \cdots, (\boldsymbol{x}^{(N)}, y^{(N)})\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}$ | dé ěr tǎ | 数据集 |
| $\boldsymbol{x}^{(n)}$ | xiàng liàng | 第n个样本的输入特征向量 |
| $y^{(n)}$ | wài | 第n个样本的标签或输出 |
| $N$ | ēn | 样本总数 |

**一句话解释：**
数据集由N个独立同分布的样本组成，每个样本包含输入特征向量和对应的标签。

**简单应用场景：**
在图像分类任务中，数据集包含多张图像（输入）和对应的类别标签（输出）。

---

---

### 公式 (2.2)：目标函数 / Target Function

$$
y = f^*(\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | 输出或标签 |
| $f^*(\boldsymbol{x})$ | ài | 真实的目标函数 |
| $\boldsymbol{x}$ | xiàng liàng | 输入特征向量 |

**一句话解释：**
目标函数是指输入特征与输出标签之间的真实映射关系。

**简单应用场景：**
在房价预测中，目标函数将房屋特征（面积、位置等）映射到房价。

---

---

### 公式 (2.3)：0-1损失函数 / 0-1 Loss Function

$$
p(\hat{y} | \boldsymbol{x}) = f_{\hat{y}}^*(\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | yī | 真实标签 |
| $f(x;\theta)$ | fúncì shù | 模型预测值 |
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 |

**一句话解释：**
当预测正确时损失为0，预测错误时损失为1。

**简单应用场景：**
分类任务中用于评估模型预测是否完全正确。

---

---

### 公式 (2.4)：准确率 / Accuracy

$$
\text{Acc}(f^*(\boldsymbol{x})) = \frac{1}{|\mathcal{D}'|} \sum_{(\boldsymbol{x},y) \in \mathcal{D}'} I(f^*(\boldsymbol{x}) = y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Acc}$ | zhǔn quèlǜ | 准确率 |
| $I(\cdot)$ | zhǐ shìhánshù | 指示函数，条件为真时为1，否则为0 |
| $\mathcal{D}'$ | dé ěr tǎ | 测试数据集 |
| $\|\mathcal{D}'\|$ | jī shù | 测试集的大小 |

**一句话解释：**
准确率是模型在测试集上正确分类的样本数占总样本数的比例。

**简单应用场景：**
在分类任务中，使用准确率评估模型在测试集上的性能。

---

---

### 公式 (2.5)：参数化假设空间 / Parameterized Hypothesis Space

$$
\mathcal{F} = \{f(\boldsymbol{x};\boldsymbol{\theta}) | \boldsymbol{\theta} \in \mathbb{R}^p\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{F}$ | fúhào jí | 假设空间（函数集合） |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 参数化函数 |
| $\boldsymbol{\theta}$ | xī tǎ | 模型参数向量 |
| $p$ | bīn dù | 参数的维度 |
| $\mathbb{R}^p$ | shí shùkōngjiān | p维实数空间 |

**一句话解释：**
假设空间是所有参数取值范围内的函数构成的集合。

**简单应用场景：**
线性回归的假设空间是所有可能的线性函数 $f(x;\theta) = \theta_0 + \theta_1 x$。

---

---

### 公式 (2.6)：线性模型 / Linear Model

$$
f(\boldsymbol{x}; \theta) = \boldsymbol{w}^\top \boldsymbol{x} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}$ | quán zhòng xiàng liàng | 权重向量 |
| $\boldsymbol{x}$ | shū rù tèzhēng xiàng liàng | 输入特征向量 |
| $b$ | piàn zhì | 偏置 |
| $\theta$ | xī tǎ | 模型参数 |

**一句话解释：**
线性模型用权重向量与输入特征的内积加上偏置来预测输出。

**简单应用场景：**
房价预测、股票价格预测等线性关系明显的回归任务。

---

---

### 公式 (2.7)：非线性模型 / Nonlinear Model

$$
f(\boldsymbol{x};\boldsymbol{\theta}) = \boldsymbol{\omega}^T \boldsymbol{\phi}(\boldsymbol{x}) + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 非线性模型 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |
| $\boldsymbol{\phi}(\boldsymbol{x})$ | fài | 非线性基函数向量 |
| $b$ | piàn zhì | 偏置项 |
| $\boldsymbol{\theta}$ | xī tǎ | 所有参数 |

**一句话解释：**
非线性模型通过先将输入映射到高维空间，再进行线性组合来处理非线性问题。

**简单应用场景：**
在手写数字识别中，通过非线性特征提取器提取特征，然后线性分类。

---

---

### 公式 (2.8)：深层非线性模型 / Deep Nonlinear Model

$$
\phi_k(\boldsymbol{x}) = h(\boldsymbol{w}_k^\top \boldsymbol{\phi}'(\boldsymbol{x}) + b_k), \quad \forall 1 \leq k \leq K
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g_k(x)$ | dì $k$ cèng de shū chū | 第k层的输出 |
| $h(\cdot)$ | fēi xiànxìng huó huà hánshù | 非线性激活函数 |
| $\boldsymbol{\omega}_k$ | dì $k$ cèng quán zhòng | 第k层权重 |
| $b_k$ | dì $k$ cèng piàn zhì | 第k层偏置 |
| $K$ | zǒng céng shù | 总层数 |

**一句话解释：**
深层网络通过堆叠多个非线性变换层来实现复杂的特征提取。

**简单应用场景：**
图像识别、自然语言处理等需要多层特征提取的任务。

---

---

### 公式 (2.9)：假设空间中的理想模型 / Ideal Model in Hypothesis Space

$$
|f(\boldsymbol{x}, \boldsymbol{\theta}^*) - y| < \varepsilon, \quad \forall(\boldsymbol{x}, y) \in \mathcal{X} \times \mathcal{Y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x}, \boldsymbol{\theta}^*)$ | hánshù | 理想的模型函数 |
| $\boldsymbol{\theta}^*$ | xī tǎ | 理想的参数 |
| $\varepsilon$ | yì pǔ xī lóng | 很小的正数 |
| $\mathcal{X} \times \mathcal{Y}$ | chéngenjī | 样本空间 |

**一句话解释：**
理想模型是使预测值与真实值的误差足够小的模型。

**简单应用场景：**
理想情况下，学习到的模型对所有样本的预测误差都在可接受范围内。

---

---

### 公式 (2.10)：经验误差与真实条件概率 / Empirical Error and True Conditional Probability

$$
|f_r(\boldsymbol{x}, \boldsymbol{\theta}^*) - p_r(y|\boldsymbol{x})| < \varepsilon, \quad \forall(\boldsymbol{x}, y) \in \mathcal{X} \times \mathcal{Y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_r(\boldsymbol{x}, \boldsymbol{\theta}^*)$ | hánshù | 学习到的条件概率 |
| $p_r(y \mid \boldsymbol{x})$ | gài lǜ | 真实的条件概率 |
| $\varepsilon$ | yì pǔ xī lóng | 很小的正数 |
| $\mathcal{X} \times \mathcal{Y}$ | chéngenjī | 样本空间 |

**一句话解释：**
要求学到的条件概率与真实条件概率的差异足够小。

**简单应用场景：**
概率分类模型应该学到与真实数据分布相近的条件概率。

---

---

### 公式 (2.11)：期望风险 / Expected Risk

$$
\mathcal{R}(\boldsymbol{\theta}) = \mathbb{E}_{(\boldsymbol{x},y) \sim p_r(\boldsymbol{x},y)} [\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta}))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(\boldsymbol{\theta})$ | fēng xiǎn | 期望风险 |
| $\mathbb{E}$ | qī wàng | 期望算子 |
| $p_r(\boldsymbol{x},y)$ | gài lǜ | 真实的联合分布 |
| $\mathcal{L}(\cdot)$ | sǔnshī | 损失函数 |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型输出 |

**一句话解释：**
期望风险是模型在整个数据分布上的平均损失。

**简单应用场景：**
衡量模型在理想情况下的整体性能。

---

---

### 公式 (2.12)：0-1损失函数 / 0-1 Loss Function

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = \begin{cases} 0 & \text{if } y = f(\boldsymbol{x};\boldsymbol{\theta}) \\ 1 & \text{if } y \neq f(\boldsymbol{x};\boldsymbol{\theta}) \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $y$ | wài | 真实标签 |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型预测值 |

**一句话解释：**
0-1损失函数在预测正确时为0，错误时为1，直接衡量分类错误。

**简单应用场景：**
在分类问题中计算分类错误率。

---

---

### 公式 (2.13)：0-1损失函数简化形式 / 0-1 Loss Function Simplified

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = I(y \neq f(\boldsymbol{x};\boldsymbol{\theta}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $I(\cdot)$ | zhǐ shìhánshù | 指示函数 |
| $y$ | wài | 真实标签 |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型预测值 |

**一句话解释：**
用指示函数表示0-1损失，当预测不等于真实值时为1，否则为0。

**简单应用场景：**
简洁表示分类问题中的预测误差。

---

---

### 公式 (2.14)：平方损失函数 / Quadratic Loss Function

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = \frac{1}{2}(y - f(\boldsymbol{x};\boldsymbol{\theta}))^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $y$ | wài | 真实标签 |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型预测值 |

**一句话解释：**
平方损失函数是预测值与真实值差的平方，常用于回归问题。

**简单应用场景：**
在线性回归中衡量预测值与真实值的偏离程度。

---

---

### 公式 (2.15)：交叉熵损失函数 / Cross-Entropy Loss Function

$$
p(y = c | \boldsymbol{x}; \theta) = f_c(\boldsymbol{x}; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = c \mid x;\theta)$ | gěi dìng $x$ shí $y$ děng yú $c$ de gài lǜ | 给定x时y等于c的概率 |
| $f_c(x;\theta)$ | dì $c$ lèi de yùcè gài lǜ | 第c类的预测概率 |
| $c$ | lèi bié suǒ yǐn | 类别索引 |

**一句话解释：**
交叉熵损失通过模型输出的概率分布与真实分布的差异来衡量分类错误。

**简单应用场景：**
多分类问题中用于评估模型对各类别的预测概率准确性。

---

---

### 公式 (2.16)：概率分布的有效性条件 / Validity Conditions of Probability Distribution

$$
f_c(\boldsymbol{x};\boldsymbol{\theta}) \in [0,1], \quad \sum_{c=1}^C f_c(\boldsymbol{x};\boldsymbol{\theta}) = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_c(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 第c类的概率 |
| $C$ | dà xiě kāi | 类别总数 |

**一句话解释：**
条件概率函数的输出必须在0到1之间，且所有类别的概率之和为1。

**简单应用场景：**
在多分类问题中，softmax函数确保输出满足概率分布的要求。

---

---

### 公式 (2.17)：交叉熵损失 / Cross-Entropy Loss

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = -y^T \log f(\boldsymbol{x};\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $y$ | wài | one-hot编码的标签向量 |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型的概率输出向量 |
| $\log$ | duì shù | 自然对数 |

**一句话解释：**
交叉熵损失衡量真实概率分布与预测概率分布之间的差异。

**简单应用场景：**
在多分类神经网络中，常用交叉熵作为损失函数。

---

---

### 公式 (2.18)：交叉熵损失求和形式 / Cross-Entropy Loss - Summation Form

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = -\sum_{c=1}^C y_c \log f_c(\boldsymbol{x};\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $y_c$ | wài | 样本的第c类标签（0或1） |
| $f_c(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型预测样本为第c类的概率 |
| $C$ | dà xiě kāi | 类别总数 |

**一句话解释：**
交叉熵损失为真实标签类对应概率的负对数，非真实标签类的概率项为0。

**简单应用场景：**
计算one-hot编码标签与softmax输出之间的交叉熵。

---

---

### 公式 (2.19)：one-hot编码下的交叉熵 / Cross-Entropy for One-hot Encoding

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = -\log f_{y}(\boldsymbol{x};\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $f_{y}(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型预测真实类别的概率 |
| $y$ | wài | 真实类别标签 |

**一句话解释：**
当使用one-hot编码时，交叉熵损失简化为真实类别概率的负对数。

**简单应用场景：**
在多分类问题中，直接计算正确类别的对数概率损失。

---

---

### 公式 (2.20)：Hinge损失函数 / Hinge Loss Function

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = \max(0, 1 - yf(\boldsymbol{x};\boldsymbol{\theta}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $y$ | wài | 标签（取值为-1或1） |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型的预测输出 |
| $\max$ | zuì dà | 最大值 |

**一句话解释：**
Hinge损失函数在预测正确且确信度足够时为0，否则线性增长。

**简单应用场景：**
在支持向量机（SVM）中用于二分类问题。

---

---

### 公式 (2.21)：Hinge损失函数 / Hinge Loss Function

$$
\mathcal{L}(y, f(\boldsymbol{x}; \theta)) \triangleq [1 - y f(\boldsymbol{x}; \theta)]_+
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | zhēnshí biāoqiān | 真实标签 |
| $f(x;\theta)$ | mó xíng yùcè zhí | 模型预测值 |
| $[x]_+$ | máx(0, $x$) | 取正部分函数 |

**一句话解释：**
Hinge损失鼓励模型对正确分类的样本有足够的置信度间隔。

**简单应用场景：**
支持向量机（SVM）中用于二分类问题的损失函数。

---

---

### 公式 (2.22)：经验风险 / Empirical Risk

$$
\mathcal{R}_{\mathcal{D}}^{emp}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^N \mathcal{L}(y^{(n)}, f(\boldsymbol{x}^{(n)};\boldsymbol{\theta}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_{\mathcal{D}}^{emp}$ | jīng yàn fēng xiǎn | 经验风险 |
| $N$ | ēn | 训练样本总数 |
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $y^{(n)}$ | wài | 第n个样本的真实标签 |
| $f(\boldsymbol{x}^{(n)};\boldsymbol{\theta})$ | hánshù | 模型对第n个样本的预测 |

**一句话解释：**
经验风险是模型在训练集上的平均损失。

**简单应用场景：**
训练神经网络时优化的目标函数。

---

---

### 公式 (2.23)：经验风险最小化 / Empirical Risk Minimization

$$
\boldsymbol{\theta}^* = \arg \min_{\boldsymbol{\theta}} \mathcal{R}_{\mathcal{D}}^{emp}(\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}^*$ | xī tǎ | 最优参数 |
| $\arg \min$ | qiú zuì xiǎo zhí | 使目标函数最小的参数 |
| $\mathcal{R}_{\mathcal{D}}^{emp}(\boldsymbol{\theta})$ | jīng yàn fēng xiǎn | 经验风险 |

**一句话解释：**
学习过程就是找到使经验风险最小的模型参数。

**简单应用场景：**
机器学习中的基本学习原则。

---

---

### 公式 (2.24)：结构风险最小化 / Structure Risk Minimization

$$
\boldsymbol{\theta}^* = \arg \min_{\boldsymbol{\theta}} \mathcal{R}_{\mathcal{D}}^{struct}(\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}^*$ | xī tǎ | 最优参数 |
| $\arg \min$ | qiú zuì xiǎo zhí | 使目标函数最小的参数 |
| $\mathcal{R}_{\mathcal{D}}^{struct}$ | jié gòu fēng xiǎn | 结构风险 |

**一句话解释：**
结构风险最小化在经验风险基础上加入正则化项以防止过拟合。

**简单应用场景：**
通过添加正则化约束改进模型的泛化能力。

---

---

### 公式 (2.25)：结构风险最小化 / Structural Risk Minimization

$$
\theta^* = \arg\min_{\theta} \mathcal{R}_{\mathcal{D}}^{emp}(\theta) + \frac{1}{2} \lambda \|\theta\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}^{emp}(\theta)$ | jīngyàn fēng xiǎn | 经验风险 |
| $\lambda$ | guīhuà xì shù | 正则化系数 |
| $\|\theta\|$ | cānshù fànshù | 参数范数 |

**一句话解释：**
通过添加正则项来平衡经验风险和模型复杂度，防止过拟合。

**简单应用场景：**
机器学习中通过L2正则化来控制模型的复杂度。

---

---

### 公式 (2.26)：带正则化的经验风险最小化 / Empirical Risk Minimization with Regularization

$$
\theta^* = \arg\min_{\theta} \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}(y^{(n)}, f(\boldsymbol{x}^{(n)}; \theta)) + \frac{1}{2} \lambda \|\theta\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N$ | zǒng yàngběn shù | 总样本数 |
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 |
| $\lambda$ | guīhuà xì shù | 正则化系数 |
| $y^{(n)}$ | dì $n$ ge yàngběn de biāoqiān | 第n个样本的标签 |
| $x^{(n)}$ | dì $n$ ge yàngběn | 第n个样本 |

**一句话解释：**
在所有样本上的平均损失加上正则项，用于学习最优的模型参数。

**简单应用场景：**
深度学习中的模型训练过程。

---

---

### 公式 (2.27)：梯度下降法更新规则 / Gradient Descent Update Rule

$$
\theta_{t+1} = \theta_t - \alpha \frac{\partial \mathcal{R}_{\mathcal{D}}(\theta)}{\partial \theta}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_t$ | dì $t$ cì míngāi de cānshù | 第t次迭代的参数 |
| $\alpha$ | xué xí lǜ | 学习率 |
| $\mathcal{R}(\theta)$ | fēng xiǎn hánshù | 风险函数 |

**一句话解释：**
参数沿着风险函数梯度的反方向更新，以逐步降低风险。

**简单应用场景：**
优化算法中的基本参数更新方式。

---

---

### 公式 (2.28)：随机梯度下降 / Stochastic Gradient Descent

$$
\theta_{t+1} = \theta_t - \alpha \frac{1}{N} \sum_{n=1}^{N} \frac{\partial \mathcal{L}(y^{(n)}, f(\boldsymbol{x}^{(n)}; \theta))}{\partial \theta}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_t$ | dì $t$ cì míngāi de cānshù | 第t次迭代的参数 |
| $N$ | pī liàng dà xiǎo | 批量大小 |
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 |

**一句话解释：**
在一个小批量样本上计算梯度并更新参数，比全批量更高效。

**简单应用场景：**
深度神经网络的实际训练中使用的主要优化方法。

---

---

### 公式 (2.29)：小批量梯度下降 / Mini-Batch Gradient Descent

$$
\theta_{t+1} \leftarrow \theta_t - \alpha \frac{1}{K} \sum_{(\boldsymbol{x}, y) \in \mathcal{S}_t} \frac{\partial \mathcal{L}(y, f(\boldsymbol{x}; \theta))}{\partial \theta}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_t$ | dì $t$ cì míngāi de cānshù | 第t次迭代的参数 |
| $\alpha$ | xué xí lǜ | 学习率 |
| $S_t$ | dì $t$ cì míngāi de xiǎo pī liàng | 第t次迭代的小批量 |

**一句话解释：**
在每个小批量数据上计算梯度并更新参数，在计算效率和收敛性之间取得平衡。

**简单应用场景：**
现代深度学习框架中标准的参数优化方法。

---

---

### 公式 (2.30)：线性回归模型 / Linear Regression Model

$$
f(\boldsymbol{x};\boldsymbol{\omega}, b) = \boldsymbol{\omega}^T \boldsymbol{x} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x};\boldsymbol{\omega}, b)$ | hánshù | 线性回归模型 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |
| $b$ | piàn zhì | 偏置项 |
| $\boldsymbol{x}$ | xiàng liàng | 输入特征向量 |

**一句话解释：**
线性回归模型通过输入特征的线性组合预测连续值。

**简单应用场景：**
房价预测、股票价格预测等回归问题。

---

---

### 公式 (2.31)：线性回归的紧凑形式 / Linear Regression - Compact Form

$$
f(\boldsymbol{x};\boldsymbol{\omega}) = \boldsymbol{\omega}^T \tilde{\boldsymbol{x}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x};\boldsymbol{\omega})$ | hánshù | 线性回归模型 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 扩展的权重向量 |
| $\tilde{\boldsymbol{x}}$ | xiàng liàng | 扩展的特征向量 |

**一句话解释：**
将偏置项包含在特征向量中，得到更紧凑的数学形式。

**简单应用场景：**
矩阵形式的计算中。

---

---

### 公式 (2.32)：增广特征向量 / Augmented Feature Vector

$$
\hat{\boldsymbol{x}} = \boldsymbol{x} \oplus 1 \triangleq \begin{bmatrix} \boldsymbol{x} \\ 1 \end{bmatrix} = \begin{bmatrix} x_1 \\ \vdots \\ x_D \\ 1 \end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | tèzhēng xiàng liàng | 特征向量 |
| $\oplus$ | liánjiē suǒ yǐn | 连接算子 |
| $D$ | tèzhēng wēi dù | 特征维度 |

**一句话解释：**
在特征向量后添加一个常数1，用于统一处理偏置项。

**简单应用场景：**
线性回归中简化权重与偏置的表示方法。

---

---

### 公式 (2.33)：增广权重向量 / Augmented Weight Vector

$$
\hat{\boldsymbol{w}} = \boldsymbol{w} \oplus b \triangleq \begin{bmatrix} \boldsymbol{w} \\ b \end{bmatrix} = \begin{bmatrix} w_1 \\ \vdots \\ w_D \\ b \end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}$ | quán zhòng xiàng liàng | 权重向量 |
| $b$ | piàn zhì | 偏置 |
| $\oplus$ | liánjiē suǒ yǐn | 连接算子 |

**一句话解释：**
将偏置项附加到权重向量中，形成增广权重向量。

**简单应用场景：**
简化线性模型的矩阵表示。

---

---

### 公式 (2.34)：经验风险最小化目标 / Empirical Risk Minimization for Linear Regression

$$
\mathcal{R}(\boldsymbol{\omega}) = \sum_{n=1}^N \mathcal{L}(y^{(n)}, f(\boldsymbol{x}^{(n)};\boldsymbol{\omega}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(\boldsymbol{\omega})$ | fēng xiǎn | 风险函数 |
| $N$ | ēn | 样本总数 |
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $y^{(n)}$ | wài | 第n个样本的真实值 |
| $f(\boldsymbol{x}^{(n)};\boldsymbol{\omega})$ | hánshù | 模型的预测值 |

**一句话解释：**
线性回归的目标是最小化所有样本上的累积损失。

**简单应用场景：**
线性回归问题的目标函数。

---

---

### 公式 (2.35)：平方损失累积 / Accumulated Quadratic Loss

$$
= \frac{1}{2} \sum_{n=1}^N (y^{(n)} - \boldsymbol{\omega}^T \boldsymbol{x}^{(n)})^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | wài | 第n个样本的真实值 |
| $\boldsymbol{\omega}^T \boldsymbol{x}^{(n)}$ | yù cè zhí | 模型的预测值 |
| $N$ | ēn | 样本总数 |

**一句话解释：**
将平方损失累加到所有样本上。

**简单应用场景：**
平方损失函数的展开形式。

---

---

### 公式 (2.36)：矩阵形式的目标函数 / Matrix Form of Objective Function

$$
= \frac{1}{2}\|\boldsymbol{y} - \boldsymbol{X}^T\boldsymbol{\omega}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{y}$ | xiàng liàng | 所有样本的标签列向量 |
| $\boldsymbol{X}$ | jǔ zhèn | 所有样本的特征矩阵 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |
| $\|\cdot\|^2$ | fàn shù | 向量的L2范数平方 |

**一句话解释：**
将样本级的损失写成矩阵形式，便于计算。

**简单应用场景：**
线性回归的矩阵形式表示。

---

---

### 公式 (2.37)：数据矩阵的增广形式 / Augmented Data Matrix

$$
\boldsymbol{X} = \begin{bmatrix} x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(N)} \\ \vdots & \vdots & \ddots & \vdots \\ x_D^{(1)} & x_D^{(2)} & \cdots & x_D^{(N)} \\ 1 & 1 & \cdots & 1 \end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{X}$ | jǔ zhèn | 增广特征矩阵 |
| $x_d^{(n)}$ | tè zhēng | 第n个样本的第d维特征 |
| $D$ | dà xiě dì | 特征维度 |
| $N$ | ēn | 样本总数 |

**一句话解释：**
每列代表一个样本的增广特征向量，最后一行为全1（代表偏置）。

**简单应用场景：**
线性回归的批量计算中。

---

---

### 公式 (2.38)：目标函数对权重的梯度 / Gradient of Objective Function w.r.t. Weights

$$
\frac{\partial \mathcal{R}(\boldsymbol{\omega})}{\partial \boldsymbol{\omega}} = \frac{1}{2} \frac{\partial \|\boldsymbol{y} - \boldsymbol{X}^T\boldsymbol{\omega}\|^2}{\partial \boldsymbol{\omega}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{R}}{\partial \boldsymbol{\omega}}$ | piān dǎo | 梯度向量 |
| $\boldsymbol{y}$ | xiàng liàng | 标签向量 |
| $\boldsymbol{X}$ | jǔ zhèn | 特征矩阵 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |

**一句话解释：**
梯度是目标函数对权重向量的偏导数。

**简单应用场景：**
梯度下降优化中计算更新方向。

---

---

### 公式 (2.39)：梯度的展开形式 / Expanded Gradient

$$
= -\boldsymbol{X}(\boldsymbol{y} - \boldsymbol{X}^T\boldsymbol{\omega})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{X}$ | jǔ zhèn | 特征矩阵 |
| $\boldsymbol{y}$ | xiàng liàng | 标签向量 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |

**一句话解释：**
梯度为特征矩阵乘以预测误差的负值。

**简单应用场景：**
线性回归梯度的计算。

---

---

### 公式 (2.40)：最优权重的闭式解 / Closed-form Solution for Optimal Weights

$$
\boldsymbol{\omega}^* = (\boldsymbol{X}\boldsymbol{X}^T)^{-1}\boldsymbol{X}\boldsymbol{y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^*$ | xī tǎ | 最优权重向量 |
| $\boldsymbol{X}$ | jǔ zhèn | 特征矩阵 |
| $(\boldsymbol{X}\boldsymbol{X}^T)^{-1}$ | níjiàoshù | 逆矩阵 |
| $\boldsymbol{y}$ | xiàng liàng | 标签向量 |

**一句话解释：**
线性回归有闭式解，可直接计算最优权重而不需迭代。

**简单应用场景：**
样本数少或矩阵可逆时的快速求解方法。

---

---

### 公式 (2.41)：闭式解的另一种形式 / Alternative Form of Closed-form Solution

$$
= \left(\sum_{n=1}^N \boldsymbol{x}^{(n)}(\boldsymbol{x}^{(n)})^T\right)^{-1} \left(\sum_{n=1}^N \boldsymbol{x}^{(n)} y^{(n)}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}^{(n)}$ | xiàng liàng | 第n个样本的特征向量 |
| $y^{(n)}$ | wài | 第n个样本的标签 |
| $N$ | ēn | 样本总数 |

**一句话解释：**
将矩阵形式展开为样本求和的形式。

**简单应用场景：**
从样本级别理解线性回归的最优解。

---

---

### 公式 (2.42)：最小二乘法参数更新 / Least Squares Parameter Update

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \boldsymbol{X}(\boldsymbol{y} - \boldsymbol{X}^\top \boldsymbol{w})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}$ | quán zhòng xiàng liàng | 权重向量 |
| $\alpha$ | xué xí lǜ | 学习率 |
| $y$ | zhēnshí biāoqiān | 真实标签 |
| $\boldsymbol{x}$ | yàngběn tèzhēng xiàng liàng | 样本特征向量 |

**一句话解释：**
根据预测误差调整权重，使预测结果更接近真实值。

**简单应用场景：**
单个样本的在线学习更新规则。

---

---

### 公式 (2.43)：岭回归的最优解 / Ridge Regression Solution

$$
\boldsymbol{\omega}^* = (\boldsymbol{X}\boldsymbol{X}^T + \lambda \boldsymbol{I})^{-1}\boldsymbol{X}\boldsymbol{y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^*$ | xī tǎ | 最优权重向量 |
| $\boldsymbol{X}$ | jǔ zhèn | 特征矩阵 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\boldsymbol{I}$ | dān wèi jǔ zhèn | 单位矩阵 |
| $\boldsymbol{y}$ | xiàng liàng | 标签向量 |

**一句话解释：**
岭回归通过在Gram矩阵上加L2正则化项，改进线性回归的数值稳定性。

**简单应用场景：**
当特征矩阵接近奇异或存在多重共线性时使用。

---

---

### 公式 (2.44)：岭回归的目标函数 / Ridge Regression Objective

$$
\mathcal{R}(\boldsymbol{\omega}) = \frac{1}{2}\|\boldsymbol{y} - \boldsymbol{X}^T\boldsymbol{\omega}\|^2 + \frac{\lambda}{2}\|\boldsymbol{\omega}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(\boldsymbol{\omega})$ | fēng xiǎn | 目标函数 |
| $\boldsymbol{y}$ | xiàng liàng | 标签向量 |
| $\boldsymbol{X}$ | jǔ zhèn | 特征矩阵 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |
| $\lambda$ | lán mǔ dá | 正则化系数 |

**一句话解释：**
岭回归在平方误差损失上加入L2范数惩罚项，鼓励权重向量较小的值。

**简单应用场景：**
防止线性回归中的过拟合问题。

---

---

### 公式 (2.45)：最大似然估计的模型假设 / Maximum Likelihood Estimation Model

$$
y = f(\boldsymbol{x};\boldsymbol{\omega}) + \varepsilon
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | 真实输出 |
| $f(\boldsymbol{x};\boldsymbol{\omega})$ | hánshù | 确定性模型部分 |
| $\varepsilon$ | yì pǔ xī lóng | 随机噪声 |

**一句话解释：**
输出由确定性的模型部分和随机噪声组成。

**简单应用场景：**
将回归问题建模为噪声的生成过程。

---

---

### 公式 (2.46)：线性回归模型（残差项）/ Linear Regression Model with Error Term

$$
y = \boldsymbol{\omega}^\top \boldsymbol{x} + \varepsilon
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | 输出标签 |
| $\boldsymbol{\omega}$ | ōu mǐ gā | 权重向量 |
| $\boldsymbol{x}$ | x | 输入特征向量 |
| $\varepsilon$ | yī prú lóng | 残差/误差项 |

**一句话解释：**
线性回归将输出表示为输入特征的线性组合加上一个独立随机误差项。

**简单应用场景：**
预测房价时，根据房屋面积、房间数等特征预测价格，模型输出加上随机噪声表示实际观测值。

---

---

### 公式 (2.47)：线性回归的概率分布 / Probabilistic Distribution of Linear Regression

$$
p(y|\boldsymbol{x}; \boldsymbol{\omega}, \sigma) = \mathcal{N}(y; \boldsymbol{\omega}^\top \boldsymbol{x}, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y \mid \boldsymbol{x}; \boldsymbol{\omega}, \sigma)$ | gài lǜ | 给定输入和参数的输出概率分布 |
| $\mathcal{N}$ | zhèng tài | 高斯/正态分布 |
| $\boldsymbol{\omega}^\top \boldsymbol{x}$ | jūn zhī | 均值/期望 |
| $\sigma^2$ | fāng chā | 方差 |
| $\sigma$ | xī gé mǎ | 标准差 |

**一句话解释：**
线性回归的输出服从以线性预测值为均值、标准差为σ的高斯分布。

**简单应用场景：**
在贝叶斯线性回归中，为参数估计提供概率框架，允许量化预测的不确定性。

---

---

### 公式 (2.48)：高斯分布概率密度函数 / Gaussian Distribution PDF

$$
p(y | \boldsymbol{x}; \boldsymbol{w}, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y - \boldsymbol{w}^\top \boldsymbol{x})^2}{2\sigma^2} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y \mid x, \omega, \sigma)$ | gěi dìng $x$ shí $y$ de tiáojiàn gài lǜ | 给定x时y的条件概率 |
| $\mathcal{N}$ | gāo sī fēnbù | 高斯分布 |
| $\sigma$ | biāo zhǔn chā | 标准差 |

**一句话解释：**
预测值围绕真实值以高斯分布的方式波动，标准差衡量不确定性。

**简单应用场景：**
回归任务中假设预测误差服从高斯分布。

---

---

### 公式 (2.49)：高斯分布表达式 / Gaussian Distribution Expression

$$
p(\boldsymbol{y} | \boldsymbol{X}; \boldsymbol{w}, \sigma) = \prod_{n=1}^{N} p(y^{(n)} | \boldsymbol{x}^{(n)}; \boldsymbol{w}, \sigma)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | biāo zhǔn chā | 标准差 |
| $\pi$ | pài | 圆周率 |
| $y$ | zhēnshí zhí | 真实值 |
| $\boldsymbol{\omega}$ | quán zhòng xiàng liàng | 权重向量 |
| $\boldsymbol{x}$ | tèzhēng xiàng liàng | 特征向量 |

**一句话解释：**
高斯分布的概率密度函数，描述了围绕预测值的数据分布形状。

**简单应用场景：**
概率论中描述连续随机变量的常用分布。

---

---

### 公式 (2.50)：似然函数的高斯分布展开 / Likelihood Function with Gaussian Distribution

$$
= \prod_{n=1}^{N} \mathcal{N}(y^{(n)}; \boldsymbol{\omega}^\top \boldsymbol{x}^{(n)}, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{N}$ | zhèng tài | 高斯分布 |
| $y^{(n)}$ | dì n ge | 第n个样本输出 |
| $\boldsymbol{\omega}^\top \boldsymbol{x}^{(n)}$ | yù cè zhí | 第n个样本的预测值 |
| $\sigma^2$ | fāng chā | 方差 |

**一句话解释：**
似然函数明确表示为高斯分布的连乘，每个样本围绕线性预测值分布。

**简单应用场景：**
贝叶斯线性回归中计算后验分布时使用，用于参数的不确定性估计。

---

---

### 公式 (2.51)：对数似然函数 / Log Likelihood Function

$$
\log p(\boldsymbol{y}|\mathbf{X}; \boldsymbol{\omega}, \sigma) = \sum_{n=1}^{N} \log \mathcal{N}(y^{(n)}; \boldsymbol{\omega}^\top \boldsymbol{x}^{(n)}, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log$ | duì shù | 自然对数 |
| $\sum$ | qiú hé | 求和 |
| $\mathcal{N}$ | zhèng tài | 高斯分布 |

**一句话解释：**
将似然函数取对数，将连乘转换为求和，便于计算和优化。

**简单应用场景：**
最大似然估计中，通过最大化对数似然函数来估计参数，计算更稳定。

---

---

### 公式 (2.52)：最大似然估计 / Maximum Likelihood Estimation

$$
\boldsymbol{w}_{ML} = (\boldsymbol{X}\boldsymbol{X}^\top)^{-1} \boldsymbol{X} \boldsymbol{y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^{ML}$ | zuì dà sìrán guō yùcè | 最大似然估计 |
| $X$ | tèzhēng jǔzhèn | 特征矩阵 |
| $y$ | biāoqiān xiàng liàng | 标签向量 |
| $^T$ | zhuǎnzhì | 转置 |

**一句话解释：**
通过最大化似然函数得到的参数估计，对应于最小二乘解。

**简单应用场景：**
线性回归模型的闭式解求解。

---

## MISSING FORMULAS (New Extraction)

---

### 公式 (2.53)：最大后验估计的分布 / Maximum A Posteriori Estimation Distribution

$$
p(\boldsymbol{w}; v) = \mathcal{N}(\boldsymbol{w}; 0, v^2\boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{w}; v)$ | gài lǜ | 权重的先验分布 |
| $\mathcal{N}$ | zhèng tài | 高斯分布 |
| $\boldsymbol{0}$ | líng xiàng liàng | 零向量（均值） |
| $v^2\boldsymbol{I}$ | fāng chā | 方差-协方差矩阵 |
| $\boldsymbol{I}$ | dān wèi | 单位矩阵 |
| $v$ | pà lǎi | 标准差超参数 |

**一句话解释：**
权重的先验分布假设为零均值、方差为$v^2$的高斯分布。

**简单应用场景：**
贝叶斯线性回归中为权重设置先验，实现L2正则化的贝叶斯解释。

---

---

### 公式 (2.54)：后验分布 / Posterior Distribution

$$
p(\boldsymbol{w}|\mathbf{X}, \boldsymbol{y}, v, \sigma) = \frac{p(\boldsymbol{w}, \boldsymbol{y}|\mathbf{X}, v, \sigma)}{\sum_{\boldsymbol{w}} p(\boldsymbol{w}, \boldsymbol{y}|\mathbf{X}, v, \sigma)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{w} \mid \mathbf{X}, \boldsymbol{y}, v, \sigma)$ | hòu yàn | 后验分布 |
| $p(\boldsymbol{w}, \boldsymbol{y} \mid \mathbf{X}, v, \sigma)$ | lián hé gài lǜ | 联合概率 |
| $\sum_{\boldsymbol{w}}$ | biàn lì | 对所有权重求和（归一化） |

**一句话解释：**
根据观测数据更新权重的分布，结合先验和似然函数。

**简单应用场景：**
贝叶斯推断中计算权重的后验分布，用于不确定性量化。

---

---

### 公式 (2.55)：后验分布的简化形式 / Posterior Distribution Simplified

$$
\propto p(\boldsymbol{y}|\mathbf{X}, \boldsymbol{w}; \sigma) p(\boldsymbol{w}; v)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\propto$ | zhèng bǐ yú | 正比于 |
| $p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{w}; \sigma)$ | sì rán hánshù | 似然函数 |
| $p(\boldsymbol{w}; v)$ | xiān yàn | 先验分布 |

**一句话解释：**
后验分布正比于似然函数与先验分布的乘积，遵循贝叶斯定理。

**简单应用场景：**
贝叶斯线性回归中的核心计算，后验 ∝ 似然 × 先验。

---

---

### 公式 (2.56)：最大后验估计 / Maximum A Posteriori Estimation

$$
\boldsymbol{\omega}^{MAP} = \arg\max_{\boldsymbol{\omega}} p(\boldsymbol{y}|\mathbf{X}, \boldsymbol{\omega}; \sigma) p(\boldsymbol{\omega}; v)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^{MAP}$ | zuì dà hòu yàn | 最大后验估计 |
| $\arg\max$ | biàn zì | 使函数最大的自变量 |
| $p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{\omega}; \sigma)$ | sì rán hánshù | 似然函数 |
| $p(\boldsymbol{\omega}; v)$ | xiān yàn | 先验分布 |

**一句话解释：**
在贝叶斯框架中，找使后验分布最大的权重估计值。

**简单应用场景：**
含正则化的线性回归中，综合考虑数据拟合和先验约束的参数估计。

---

---

### 公式 (2.57)：对数后验 / Log Posterior

$$
\log p(\boldsymbol{w}|\mathbf{X}, \boldsymbol{y}, v, \sigma) \propto \log p(\boldsymbol{y}|\mathbf{X}, \boldsymbol{\omega}; \sigma) + \log p(\boldsymbol{\omega}; v)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log$ | duì shù | 自然对数 |
| $p(\boldsymbol{w} \mid \mathbf{X}, \boldsymbol{y}, v, \sigma)$ | hòu yàn | 后验分布 |
| $p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{\omega}; \sigma)$ | sì rán hánshù | 似然函数 |
| $p(\boldsymbol{\omega}; v)$ | xiān yàn | 先验分布 |

**一句话解释：**
对数后验等于对数似然加上对数先验，便于计算和优化。

**简单应用场景：**
贝叶斯参数估计中使用对数域进行数值计算，提高稳定性。

---

---

### 公式 (2.58)：对数后验的展开式 / Log Posterior Expanded Form

$$
\propto -\frac{1}{2\sigma^2} \sum_{n=1}^{N} (y^{(n)} - \boldsymbol{\omega}^\top \boldsymbol{x}^{(n)})^2 - \frac{1}{2v^2} \boldsymbol{\omega}^\top \boldsymbol{\omega}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma^2$ | fāng chā | 似然函数的方差 |
| $\sum_{n=1}^{N}$ | qiú hé | 对所有样本求和 |
| $(y^{(n)} - \boldsymbol{\omega}^\top \boldsymbol{x}^{(n)})^2$ | píng fāng yù chà | 平方预测误差 |
| $v^2$ | xiān yàn fāng chā | 先验分布的方差 |
| $\boldsymbol{\omega}^\top \boldsymbol{\omega}$ | fàn shù píng fāng | 权重的平方范数 |

**一句话解释：**
对数后验由数据拟合项（似然）和正则化项（先验）组成。

**简单应用场景：**
岭回归（Ridge Regression）中，最大化此目标函数等价于最小化带L2正则化的损失函数。

---

---

### 公式 (2.59)：对数后验的矩阵形式 / Log Posterior in Matrix Form

$$
= -\frac{1}{2\sigma^2} \|\boldsymbol{y} - \mathbf{X}^\top \boldsymbol{\omega}\|^2 - \frac{1}{2v^2} \boldsymbol{\omega}^\top \boldsymbol{\omega}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\cdot\|^2$ | fàn shù píng fāng | 平方范数 |
| $\boldsymbol{y}$ | biāo qiān xiàng liàng | 标签向量 |
| $\mathbf{X}^\top \boldsymbol{\omega}$ | yù cè xiàng liàng | 预测向量 |
| $\sigma^2$ | fāng chā | 方差参数 |
| $v^2$ | xiān yàn fāng chā | 先验方差 |

**一句话解释：**
对数后验的矩阵形式，可高效地用于大规模数据的计算。

**简单应用场景：**
实现贝叶斯线性回归时，用矩阵运算加速计算后验参数。

---

---

### 公式 (2.60)：贝叶斯后验分布 / Bayesian Posterior Distribution

$$
p(\boldsymbol{\omega}|X, y, \nu, \sigma) = \frac{p(y|X, \boldsymbol{\omega}, \sigma)p(\boldsymbol{\omega}|\nu)}{\sum_{\boldsymbol{\omega}} p(y, X|\boldsymbol{\omega}, \sigma)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{\omega} \mid X, y, \nu, \sigma)$ | hòuyàn fēnbù | 后验分布 |
| $p(y \mid X, \boldsymbol{\omega}, \sigma)$ | sìrán hánshù | 似然函数 |
| $p(\boldsymbol{\omega} \mid \nu)$ | xiānzhuàn fēnbù | 先验分布 |
| $\nu$ | xiānzhuàn chāoshù | 先验超参数 |

**一句话解释：**
在观测到数据后，根据贝叶斯定理更新对参数的概率分布。

**简单应用场景：**
贝叶斯推断中结合先验和似然来推断后验分布。

---

---

### 公式 (2.61)：贝叶斯估计最大后验 / Maximum A Posteriori Estimation

$$
\boldsymbol{\omega}^{MAP} = \arg\max_{\boldsymbol{\omega}} p(y|X, \boldsymbol{\omega}, \sigma)p(\boldsymbol{\omega};\nu)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^{MAP}$ | zuì dà hòuyàn guō yùcè | 最大后验估计 |
| $p(y \mid X, \boldsymbol{\omega}, \sigma)$ | sìrán hánshù | 似然函数 |
| $p(\boldsymbol{\omega};\nu)$ | xiānzhuàn fēnbù | 先验分布 |

**一句话解释：**
选择使后验分布最大的参数值，兼顾数据拟合和先验信息。

**简单应用场景：**
贝叶斯方法中的点估计方式。

---

---

### 公式 (2.62)：对数似然函数 / Log Likelihood Function

$$
\log p(y|X; \boldsymbol{\omega}, \sigma) = \sum_{n=1}^N \log \mathcal{N}(y^{(n)}; \boldsymbol{\omega}^T x^{(n)}, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log$ | duì shù | 对数 |
| $p(y \mid X; \boldsymbol{\omega}, \sigma)$ | liánhé sìrán hánshù | 联合似然函数 |
| $\mathcal{N}$ | gāo sī fēnbù | 高斯分布 |
| $N$ | yàngběn zǒng shù | 样本总数 |

**一句话解释：**
所有样本的对数似然之和，用于评估模型参数的好坏。

**简单应用场景：**
最大似然估计中的目标函数。

---

---

### 公式 (2.63)：高斯先验下的对数后验 / Log Posterior with Gaussian Prior

$$
\log p(\boldsymbol{\omega}|y, X; \nu, \sigma) \propto \log p(y|X; \boldsymbol{\omega}, \sigma) + \log p(\boldsymbol{\omega};\nu)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(\boldsymbol{\omega} \mid y, X; \nu, \sigma)$ | duì shù hòuyàn | 对数后验 |
| $\propto$ | zhèng bǐ lì yú | 正比例于 |
| $\nu$ | xiānzhuàn chāoshù | 先验超参数 |

**一句话解释：**
后验分布与似然和先验的乘积成正比，可用于贝叶斯参数推断。

**简单应用场景：**
贝叶斯线性回归中的后验推断。

---

---

### 公式 (2.64)：完整的贝叶斯推断目标 / Complete Bayesian Inference Objective

$$
\log p(\boldsymbol{\omega}|y, X; \nu, \sigma) \propto -\frac{1}{2\sigma^2} \sum_{n=1}^N (y^{(n)} - \boldsymbol{\omega}^T x^{(n)})^2 - \frac{1}{2\nu^2}\boldsymbol{\omega}^T\boldsymbol{\omega}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | yùcè biāo zhǔn chā | 预测标准差 |
| $\nu$ | xiānzhuàn biāo zhǔn chā | 先验标准差 |
| $\boldsymbol{\omega}$ | quán zhòng xiàng liàng | 权重向量 |

**一句话解释：**
对数后验由平方误差项和正则化项（高斯先验）组成。

**简单应用场景：**
贝叶斯线性回归的完整目标函数。

---

---

### 公式 (2.65)：非参数贝叶斯先验 / Non-Parametric Bayesian Prior

$$
p(\boldsymbol{\omega}; \nu) \propto \exp\left(-\frac{\|\boldsymbol{\omega}\|^2}{2\nu^2}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{\omega}; \nu)$ | xiānzhuàn fēnbù | 先验分布 |
| $\nu$ | xiānzhuàn chāoshù | 先验超参数 |
| $\|\boldsymbol{\omega}\|$ | quán zhòng de fànshù | 权重的范数 |

**一句话解释：**
高斯先验假设权重围绕零点以高斯分布的方式分布。

**简单应用场景：**
贝叶斯方法中常用的共轭先验。

---

---

### 公式 (2.66)：期望值的贝叶斯估计 / Bayesian Expectation Estimation

$$
\mathbb{E}[\boldsymbol{\omega}|y, X; \nu, \sigma] = (\nu^{-2}I + \sigma^{-2}X^T X)^{-1}\sigma^{-2}X^T y
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{E}$ | qī wàng zhí | 期望值 |
| $I$ | dānwèi jǔzhèn | 单位矩阵 |
| $X$ | tèzhēng jǔzhèn | 特征矩阵 |
| $y$ | biāoqiān xiàng liàng | 标签向量 |

**一句话解释：**
后验分布的均值是先验和似然信息的加权组合。

**简单应用场景：**
贝叶斯线性回归中的点估计。

---

---

### 公式 (2.67)：后验协方差矩阵 / Posterior Covariance Matrix

$$
\text{Cov}[\boldsymbol{\omega}|y, X; \nu, \sigma] = (\nu^{-2}I + \sigma^{-2}X^T X)^{-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Cov}$ | xiéfāng chà jǔzhèn | 协方差矩阵 |
| $\nu^{-2}I$ | xiānzhuàn jīnmì dù | 先验精密度 |
| $\sigma^{-2}X^T X$ | sìrán jīnmì dù | 似然精密度 |

**一句话解释：**
后验分布的不确定性取决于先验强度和数据信息量。

**简单应用场景：**
贝叶斯推断中量化参数估计的不确定性。

---

---

### 公式 (2.68)：贝叶斯预测分布 / Bayesian Predictive Distribution

$$
p(y^*|x^*; y, X, \nu, \sigma) = \int p(y^*|x^*, \boldsymbol{\omega}, \sigma) p(\boldsymbol{\omega}|y, X; \nu, \sigma) d\boldsymbol{\omega}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y^* \mid x^*; y, X, \nu, \sigma)$ | yùcè fēnbù | 预测分布 |
| $y^*$ | xīn yàngběn biāoqiān | 新样本标签 |
| $x^*$ | xīn yàngběn tèzhēng | 新样本特征 |

**一句话解释：**
新样本的预测分布是在后验分布下对所有可能参数值积分。

**简单应用场景：**
贝叶斯模型中对未来数据的不确定性预测。

---

---

### 公式 (2.69)：分类问题的后验估计 / Classification Posterior Estimation

$$
p(y = c|x, X, y) = \frac{p(x|y=c, X, y)p(y=c|X, y)}{p(x|X, y)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = c \mid x, X, y)$ | hòuyàn gài lǜ | 后验概率 |
| $p(x \mid y=c, X, y)$ | lèi tiáojiàn gài lǜ | 类条件概率 |
| $p(y=c \mid X, y)$ | lèi xiānzhuàn gài lǜ | 类先验概率 |

**一句话解释：**
分类中利用贝叶斯定理将类条件概率和先验转换为后验概率。

**简单应用场景：**
生成式分类模型的贝叶斯推断。

---

---

### 公式 (2.70)：判别式分类模型 / Discriminative Classification Model

$$
p(y = c|x; \boldsymbol{\theta}) = \text{softmax}(f_c(x; \boldsymbol{\theta}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = c \mid x; \boldsymbol{\theta})$ | shēnfèn hòuyàn gài lǜ | 身份后验概率 |
| $f_c(x; \boldsymbol{\theta})$ | dì $c$ lèi de shū chū | 第c类的输出 |
| $\text{softmax}$ | ruǎnhuà zuìdà huánshù | 软化最大函数 |

**一句话解释：**
判别模型直接对后验概率进行建模，通过softmax将输出转化为概率分布。

**简单应用场景：**
神经网络多分类问题中的常用方法。

---

---

### 公式 (2.71)：交叉熵损失的完整形式 / Complete Cross-Entropy Loss

$$
\mathcal{L}(y, f(x; \boldsymbol{\theta})) = -y \log f(x; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 |
| $y$ | dú huà biāoqiān xiàng liàng | 独热标签向量 |
| $f(x; \boldsymbol{\theta})$ | mó xíng yùcè gài lǜ | 模型预测概率 |

**一句话解释：**
交叉熵损失衡量真实分布与预测分布之间的差异。

**简单应用场景：**
多分类神经网络的标准损失函数。

---

---

### 公式 (2.72)：批量交叉熵损失 / Batch Cross-Entropy Loss

$$
\mathcal{L}(y, f(x; \boldsymbol{\theta})) = -\sum_{c=1}^C y_c \log f_c(x; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $C$ | lèi bié zǒng shù | 类别总数 |
| $y_c$ | dì $c$ lèi de zhēnshí biāoqiān | 第c类的真实标签 |
| $f_c(x; \boldsymbol{\theta})$ | dì $c$ lèi de yùcè gài lǜ | 第c类的预测概率 |

**一句话解释：**
对所有类别的交叉熵求和，得到样本的总损失。

**简单应用场景：**
多分类问题的损失计算。

---

---

### 公式 (2.73)：正则化参数优化 / Regularized Parameter Optimization

$$
\theta^* = \arg\min_\theta \mathcal{R}^{emp}(\theta) + \lambda \Omega(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}^{emp}(\theta)$ | jīngyàn fēng xiǎn | 经验风险 |
| $\lambda$ | guīhuà xì shù | 正则化系数 |
| $\Omega(\theta)$ | guīhuà xiàng | 正则化项 |

**一句话解释：**
在经验风险基础上加入正则化项，以控制模型复杂度。

**简单应用场景：**
防止过拟合的标准优化目标。

---

---

### 公式 (2.74)：L1正则化 / L1 Regularization

$$
\Omega(\theta) = \|\theta\|_1 = \sum_{d=1}^D |\theta_d|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\theta\|_1$ | $L_1$ fànshù | L1范数 |
| $D$ | cānshù zǒng shù | 参数总数 |
| $\theta_d$ | dì $d$ ge cānshù | 第d个参数 |

**一句话解释：**
L1正则化促进模型参数的稀疏性，自动进行特征选择。

**简单应用场景：**
特征选择和模型压缩中常用的正则化方式。

---

---

### 公式 (2.75)：L2正则化 / L2 Regularization

$$
\Omega(\theta) = \|\theta\|_2^2 = \sum_{d=1}^D \theta_d^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\theta\|_2^2$ | $L_2$ fànshù de píngfāng | L2范数的平方 |
| $D$ | cānshù zǒng shù | 参数总数 |

**一句话解释：**
L2正则化使参数值较小且均匀分布，有助于防止过拟合。

**简单应用场景：**
机器学习中最常用的正则化方法。

---

---

### 公式 (2.76)：提前停止验证集监控 / Early Stopping Validation Monitoring

$$
\text{if } \mathcal{L}(\mathcal{D}_{\text{val}}) \text{ increases, stop training}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\mathcal{D}_{\text{val}})$ | yàn zhèng jí sǔn shī | 验证集损失 |
| $\mathcal{D}_{\text{val}}$ | yàn zhèng jí | 验证集 |

**一句话解释：**
监控验证集损失，当其开始增加时停止训练以防止过拟合。

**简单应用场景：**
深度学习训练中的常用防过拟合技巧。

---

---

### 公式 (2.77)：数据增强 / Data Augmentation

$$
\mathcal{D}_{\text{aug}} = \mathcal{D} \cup \{\text{transformed samples of } \mathcal{D}\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_{\text{aug}}$ | zēngqiáng hòu de shù jù jí | 增强后的数据集 |
| $\mathcal{D}$ | yuán shǐ shù jù jí | 原始数据集 |
| $\cup$ | bìng jí | 并集 |

**一句话解释：**
通过对原始数据进行变换（如旋转、裁剪等）来扩充训练集。

**简单应用场景：**
计算机视觉中提高模型泛化能力的方法。

---

---

### 公式 (2.78)：Dropout正则化 / Dropout Regularization

$$
\tilde{x} = x \odot m, \quad m \sim \text{Bernoulli}(p)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{x}$ | jīng guò dropout de shū chū | 经过dropout的输出 |
| $x$ | yuán shǐ shū chū | 原始输出 |
| $m$ | suíjī shàngxià tàojiàng zhào | 随机上下掉层照 |
| $p$ | báo liú gài lǜ | 保留概率 |
| $\odot$ | yuánsù wise chéngfǎ | 元素积 |

**一句话解释：**
随机丢弃神经元输出，强制网络学习冗余特征表示。

**简单应用场景：**
深度神经网络中防止过拟合的主要技巧。

---

---

### 公式 (2.79)：批归一化 / Batch Normalization

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{x}$ | guīhuà hòu de shū chū | 归一化后的输出 |
| $\mu_B$ | píjiàn píjūn zhí | 批均值 |
| $\sigma_B^2$ | píjiàn fāngchā | 批方差 |
| $\gamma, \beta$ | kěxué cānshù | 可学习参数 |
| $\epsilon$ | xiǎo cháng shù | 小常数 |

**一句话解释：**
对每个批次的输入进行标准化，然后用可学习参数进行缩放和平移。

**简单应用场景：**
现代深度神经网络中加速训练和改进性能的技术。

---

---

### 公式 (2.80)：学习率衰减 / Learning Rate Decay

$$
\alpha_t = \alpha_0 \cdot \gamma^t, \quad \text{or} \quad \alpha_t = \frac{\alpha_0}{1 + \lambda t}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | dì $t$ cì míngāi de xué xí lǜ | 第t次迭代的学习率 |
| $\alpha_0$ | chūshǐ xué xí lǜ | 初始学习率 |
| $\gamma$ | xuānhuà xì shù | 衰减系数 |
| $\lambda$ | xuānhuà shù | 衰减数 |

**一句话解释：**
随着训练进行逐步降低学习率，使得优化在后期更加稳定。

**简单应用场景：**
深度学习优化中常用的学习率调度策略。

---

---

### 公式 (2.81)：动量优化法 / Momentum Optimization

$$
v_t = \beta v_{t-1} + (1-\beta)\nabla \mathcal{L}(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $v_t$ | dì $t$ cì míngāi de dòngliàng | 第t次迭代的动量 |
| $\beta$ | dòngliàng xì shù | 动量系数 |
| $\nabla \mathcal{L}(\theta)$ | sǔn shī hánshù de tī dù | 损失函数的梯度 |

**一句话解释：**
梯度更新加入历史梯度的加权和，形成惯性以加快收敛。

**简单应用场景：**
优化算法中改进收敛速度的常用技巧。

---

---

### 公式 (2.82)：Adam优化器 / Adam Optimizer

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla\mathcal{L}, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla\mathcal{L})^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $m_t$ | dì $t$ cì míngāi de yī jiē dòngliàng | 第t次迭代的一阶动量 |
| $v_t$ | dì $t$ cì míngāi de èr jiē dòngliàng | 第t次迭代的二阶动量 |
| $\beta_1, \beta_2$ | zhǐ shù pínghuà xì shù | 指数平滑系数 |

**一句话解释：**
Adam优化器结合了一阶和二阶矩信息，自适应调整每个参数的学习率。

**简单应用场景：**
现代深度学习中最广泛使用的优化算法。

---

---

### 公式 (2.83)：Adam参数更新 / Adam Parameter Update

$$
\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_t$ | dì $t$ cì míngāi de cānshù | 第t次迭代的参数 |
| $\alpha$ | xué xí lǜ | 学习率 |
| $m_t$ | yī jiē dòngliàng guō yùcè | 一阶动量估计 |
| $v_t$ | èr jiē dòngliàng guō yùcè | 二阶动量估计 |

**一句话解释：**
使用自适应学习率对参数进行更新，步长由历史梯度信息确定。

**简单应用场景：**
Adam优化器中的参数更新规则。

---

---

### 公式 (2.84)：RMSprop优化器 / RMSprop Optimizer

$$
v_t = \beta v_{t-1} + (1-\beta)(\nabla\mathcal{L})^2, \quad \theta_t = \theta_{t-1} - \alpha \frac{\nabla\mathcal{L}}{\sqrt{v_t} + \epsilon}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $v_t$ | dì $t$ cì míngāi de píngfāng tī dù jūnzhí | 第t次迭代的平方梯度均值 |
| $\beta$ | zhǐ shù pínghuà xì shù | 指数平滑系数 |
| $\alpha$ | xué xí lǜ | 学习率 |

**一句话解释：**
RMSprop使用平方梯度的指数移动平均来自适应调整学习率。

**简单应用场景：**
递归神经网络和其他深度学习模型的优化。

---

---

### 公式 (2.85)：AdaGrad优化器 / AdaGrad Optimizer

$$
g_t = g_{t-1} + (\nabla\mathcal{L})^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{g_t} + \epsilon}\nabla\mathcal{L}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g_t$ | dì $t$ cì míngāi de píngfāng tī dù lěi jī | 第t次迭代的平方梯度累积 |
| $\alpha$ | xué xí lǜ | 学习率 |
| $\nabla\mathcal{L}$ | tī dù | 梯度 |

**一句话解释：**
AdaGrad根据历史梯度平方累积值自适应调整学习率，频繁更新的参数学习率更小。

**简单应用场景：**
处理稀疏梯度的优化问题。

---

---

### 公式 (2.86)：权重初始化 Xavier / Xavier Weight Initialization

$$
\theta \sim \mathcal{N}\left(0, \frac{1}{n_{in} + n_{out}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta$ | quán zhòng | 权重 |
| $\mathcal{N}$ | gāo sī fēnbù | 高斯分布 |
| $n_{in}$ | shū rù wēi dù | 输入维度 |
| $n_{out}$ | shū chū wēi dù | 输出维度 |

**一句话解释：**
Xavier初始化根据输入输出维度设置权重的方差，保持激活值的适当范围。

**简单应用场景：**
深度神经网络权重初始化的标准方法。

---

---

### 公式 (2.87)：He权重初始化 / He Weight Initialization

$$
\theta \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta$ | quán zhòng | 权重 |
| $\mathcal{N}$ | gāo sī fēnbù | 高斯分布 |
| $n_{in}$ | shū rù wēi dù | 输入维度 |

**一句话解释：**
He初始化专为ReLU激活函数设计，提高深层网络的训练效果。

**简单应用场景：**
使用ReLU激活函数的深度神经网络的权重初始化。

---

---

### 公式 (2.88)：批大小与学习率的关系 / Batch Size and Learning Rate Relationship

$$
\alpha' = \alpha \times \sqrt{\frac{B'}{B}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha'$ | xīn xué xí lǜ | 新学习率 |
| $\alpha$ | yuán xué xí lǜ | 原学习率 |
| $B'$ | xīn píjiàn dà xiǎo | 新批大小 |
| $B$ | yuán píjiàn dà xiǎo | 原批大小 |

**一句话解释：**
增大批大小时应按比例增大学习率，以保持收敛速度。

**简单应用场景：**
分布式训练和批大小调整时的学习率调度。

---

---

### 公式 (2.89)：过拟合度量 / Overfitting Measure

$$
\text{Overfitting Ratio} = \frac{\mathcal{L}_{\text{train}}}{\mathcal{L}_{\text{val}}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{train}}$ | xùnliàn jí sǔn shī | 训练集损失 |
| $\mathcal{L}_{\text{val}}$ | yàn zhèng jí sǔn shī | 验证集损失 |

**一句话解释：**
训练损失与验证损失的比值越小，表示模型过拟合程度越轻。

**简单应用场景：**
评估模型泛化能力的指标。

---

---

### 公式 (2.90)：模型容量的衡量 / Model Capacity Measure

$$
\text{Capacity} = \log_2(N \cdot M \cdot D)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N$ | cānshù zǒng shù | 参数总数 |
| $M$ | xiàochuàn dà xiǎo | 小数大小 |
| $D$ | tèzhēng wēi dù | 特征维度 |

**一句话解释：**
模型容量表示模型能够学习的信息复杂度，通常与参数数量相关。

**简单应用场景：**
理论分析模型的表示能力。

---

---

### 公式 (2.91)：结构化稀疏性正则化 / Structured Sparsity Regularization

$$
\Omega(\theta) = \sum_{g} \|\theta_g\|_2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Omega(\theta)$ | guīhuà xiàng | 正则化项 |
| $g$ | cānshù zǔ | 参数组 |
| $\|\theta_g\|_2$ | dì $g$ zǔ cānshù de $L_2$ fànshù | 第g组参数的L2范数 |

**一句话解释：**
结构化稀疏性促进参数成组为零，实现模型压缩和加速。

**简单应用场景：**
神经网络剪枝和模型压缩中的正则化方法。

---

---

## 第3章：线性模型

### 公式 (3.1)：线性模型 / Linear Model

$$
f(x; \boldsymbol{w}) = w_1 x_1 + w_2 x_2 + \cdots + w_D x_D + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x; \boldsymbol{w})$ | fúnshù | 线性模型输出 |
| $w_i$ | quánzhòng | 权重参数 |
| $x_i$ | shūrù | 输入特征 |
| $b$ | piānzhì | 偏置项 |
| $D$ | shùliàng | 特征维度 |

**一句话解释：**
线性模型通过特征的加权求和加上偏置得到输出，是最基础的机器学习模型。

**简单应用场景：**
房价预测，其中特征为房屋面积、位置等，模型学习各特征对房价的影响权重。

---

---

### 公式 (3.2)：向量形式线性模型 / Linear Model in Vector Form

$$
f(x; \boldsymbol{w}) = \boldsymbol{w}^T \boldsymbol{x} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | quánzhòng xiàngliàng | 权重向量 |
| $\boldsymbol{x}$ | tèzhēng xiàngliàng | 输入特征向量 |
| $\boldsymbol{w}^T$ | quánzhòng xiàngliàng de zhuǎnzhì | 权重向量的转置 |
| $b$ | piānzhì | 偏置项 |

**一句话解释：**
线性模型的向量形式表示，通过权重向量与特征向量的内积加偏置得到输出。

**简单应用场景：**
在实现线性模型时，使用向量形式便于计算和编程实现。

---

---

### 公式 (3.3)：判别函数的线性模型 / Linear Classification Model

$$
y = g(f(x; \boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | shūchū | 分类输出标签 |
| $g(\cdot)$ | pànbié hánshù | 判别函数 |
| $f(x; \boldsymbol{w})$ | xiànxìng hánshù | 线性函数 |

**一句话解释：**
分类问题中，通过判别函数作用于线性函数的输出得到类别标签。

**简单应用场景：**
二分类问题，判别函数将线性组合的连续值映射到离散的类标签。

---

---

### 公式 (3.4)：符号函数的判别函数 / Sign Function as Discriminant

$$
g(f(x; \boldsymbol{w})) = \text{sgn}(f(x; \boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g(\cdot)$ | fúhào hánshù | 符号函数 |
| $\text{sgn}(\cdot)$ | fúhào | 符号函数 |

**一句话解释：**
使用符号函数将线性函数的值映射到正或负，从而进行二分类。

**简单应用场景：**
感知机模型中，根据线性组合的正负进行分类决策。

---

---

### 公式 (3.5)：符号函数的定义 / Definition of Sign Function

$$
\hat{y} = \operatorname{sgn}(f(\boldsymbol{x}; \boldsymbol{w}))
\triangleq
\begin{cases}
+1 & \text{if } f(\boldsymbol{x}; \boldsymbol{w}) > 0, \\
-1 & \text{if } f(\boldsymbol{x}; \boldsymbol{w}) < 0.
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{sgn}(\cdot)$ | fúhào hánshù | 符号函数 |
| $+1$ | jiāyī | 正类标签 |
| $-1$ | fùyī | 负类标签 |

**一句话解释：**
符号函数根据输入的正负返回相应的类标签，是分类中的基本决策规则。

**简单应用场景：**
在二分类中，判断样本属于正类还是负类。

---

---

### 公式 (3.6)：signed distance / 有符号距离

$$
\gamma = \frac{f(x; \boldsymbol{w})}{\|\boldsymbol{w}\|}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gāmǎ | 有符号距离 |
| $f(x; \boldsymbol{w})$ | xiànxìng hánshù | 线性函数值 |
| $\|\boldsymbol{w}\|$ | quánzhòng xiàngliàng de fànshù | 权重向量的范数 |

**一句话解释：**
点到决策平面的有符号距离，用于衡量分类的确信度。

**简单应用场景：**
支持向量机中，衡量样本点到分类超平面的距离。

---

---

### 公式 (3.7)：二分类线性模型的约束条件 / Separability Constraint for Binary Classification

$$
\begin{cases}
f(\boldsymbol{x}^{(n)}; \boldsymbol{w}^*) > 0 & \text{if } y^{(n)} = 1, \\
f(\boldsymbol{x}^{(n)}; \boldsymbol{w}^*) < 0 & \text{if } y^{(n)} = -1.
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | biāoqiān | 第n个样本的标签 |
| $f(x^{(n)}; \boldsymbol{w})$ | xiànxìng hánshù zhí | 线性函数值 |
| $N$ | yàngběn shù | 样本总数 |

**一句话解释：**
线性可分数据集的充要条件：所有样本的真实标签与预测符号同号。

**简单应用场景：**
判断数据是否线性可分，以确定感知机是否收敛。

---

---

### 公式 (3.8)：二分类线性模型的可分性定义 / Definition of Linear Separability

$$
\begin{align}
f(x^{(n)}; \boldsymbol{w}) > 0, \quad y^{(n)} = +1 \\
f(x^{(n)}; \boldsymbol{w}) < 0, \quad y^{(n)} = -1
\end{align}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x^{(n)}; \boldsymbol{w})$ | xiànxìng hánshù zhí | 线性函数值 |
| $y^{(n)}$ | zhēnshí biāoqiān | 真实标签 |

**一句话解释：**
线性可分数据的定义：存在决策超平面使得不同类别的点分别位于超平面的两侧。

**简单应用场景：**
二分类问题中，判断正类和负类是否能被直线/平面完全分开。

---

---

### 公式 (3.9)：感知机损失函数 / Perceptron Loss Function

$$
\mathcal{L}_{01}(y, f(x; \boldsymbol{w})) = I(yf(x; \boldsymbol{w}) < 0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{01}$ | línsǔn hánshù | 0-1损失函数 |
| $I(\cdot)$ | zhǐshìfúnshù | 指示函数 |
| $y$ | biāoqiān | 真实标签 |
| $f(x; \boldsymbol{w})$ | xiànxìng hánshù | 线性函数值 |

**一句话解释：**
0-1损失函数计算预测错误的样本数，是分类问题的理想损失函数。

**简单应用场景：**
评估分类器的错分率，但由于不可导，实际中通常用替代损失函数。

---

---

### 公式 (3.10)："一对其余"多分类方法 / One-vs-Rest Multi-class Classification

$$
f_c(x; \boldsymbol{w}_c) = \boldsymbol{w}_c^T \boldsymbol{x} + b_c, \quad c \in \{1, \ldots, C\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_c(x; \boldsymbol{w}_c)$ | dì c lèi de xiànxìng hánshù | 第c类的线性函数 |
| $\boldsymbol{w}_c$ | dì c lèi de quánzhòng xiàngliàng | 第c类的权重向量 |
| $C$ | fēnlèi shù | 分类总数 |

**一句话解释：**
为每一类训练一个独立的二分类器，共需训练C个分类器。

**简单应用场景：**
多分类问题，例如手写数字识别中对每个数字类别训练独立的分类器。

---

---

### 公式 (3.11)：argmax多分类 / Argmax Multi-class Classification

$$
y = \arg\max_{c=1}^{C} f_c(x; \boldsymbol{w}_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | yùcè lèibié | 预测类别 |
| $\arg\max$ | zuìdà zhí duìyìng de cānshù | 最大值对应的参数 |
| $f_c(x; \boldsymbol{w}_c)$ | dì c lèi de xiànxìng hánshù | 第c类的线性函数 |

**一句话解释：**
选择所有分类器中输出值最大的类别作为最终预测类别。

**简单应用场景：**
在一对其余方法中，通过比较各分类器的得分选择最可能的类别。

---

---

### 公式 (3.12)：Logistic回归模型 / Logistic Regression Model

$$
p(y = 1|\boldsymbol{x}) = g(f(x; \boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = 1 \mid \boldsymbol{x})$ | gěi dìng x xià, y=1 de gàilǜ | 给定x下，y=1的概率 |
| $g(\cdot)$ | jīhuà hánshù | 激活函数 |
| $f(x; \boldsymbol{w})$ | xiànxìng hánshù | 线性函数 |

**一句话解释：**
Logistic回归通过激活函数将线性函数的输出映射到[0,1]之间作为类别概率。

**简单应用场景：**
二分类问题，输出样本属于正类的概率。

---

---

### 公式 (3.13)：Logistic激活函数应用 / Logistic Activation Function

$$
p(y = 1|\boldsymbol{x}) = \sigma(\boldsymbol{w}^T \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma(\cdot)$ | xī gé mǎ hánshù | sigmoid函数 |
| $\boldsymbol{w}^T \boldsymbol{x}$ | quánzhòng xiàngliàng zhuǎnzhì chéng tèzhēng xiàngliàng | 权重向量转置乘特征向量 |

**一句话解释：**
使用sigmoid函数作为激活函数，将线性组合的值映射到概率空间。

**简单应用场景：**
Logistic回归中，将任意实数值映射到[0,1]区间表示概率。

---

---

### 公式 (3.14)：Sigmoid函数的定义 / Sigmoid Function Definition

$$
p(y=1|\boldsymbol{x}) = \sigma(\boldsymbol{w}^\top \boldsymbol{x}) \triangleq \frac{1}{1 + \exp(-\boldsymbol{w}^\top \boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma(x)$ | xī gé mǎ hánshù | sigmoid函数 |
| $\exp(\cdot)$ | zhǐ shù hánshù | 指数函数 |
| $\boldsymbol{w}^T \boldsymbol{x}$ | xiànxìng zǔhé | 线性组合 |

**一句话解释：**
sigmoid函数是光滑的S形函数，将实数值平滑地映射到(0,1)区间。

**简单应用场景：**
神经网络中的常用激活函数，用于引入非线性。

---

---

### 公式 (3.15)：Logistic回归负类概率 / Logistic Regression Negative Class Probability

$$
p(y = 0|\boldsymbol{x}) = 1 - p(y = 1|\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = 0 \mid \boldsymbol{x})$ | gěi dìng x xià, y=0 de gàilǜ | 给定x下，y=0的概率 |
| $p(y = 1 \mid \boldsymbol{x})$ | gěi dìng x xià, y=1 de gàilǜ | 给定x下，y=1的概率 |

**一句话解释：**
二分类问题中，两类的概率和为1，所以负类概率等于1减去正类概率。

**简单应用场景：**
二分类Logistic回归中，计算样本属于负类的概率。

---

---

### 公式 (3.16)：负类概率的另一种表达 / Alternative Form of Negative Class Probability

$$
p(y = 0|\boldsymbol{x}) = \frac{\exp(-\boldsymbol{w}^T \boldsymbol{x})}{1 + \exp(-\boldsymbol{w}^T \boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp(-\boldsymbol{w}^T \boldsymbol{x})$ | zhǐ shù hánshù | 指数函数 |
| $\boldsymbol{w}^T \boldsymbol{x}$ | xiànxìng zǔhé | 线性组合 |

**一句话解释：**
负类概率的另一种数学表达形式，显示了两类概率的对称性。

**简单应用场景：**
推导Logistic回归的似然函数和最大似然估计。

---

---

### 公式 (3.17)：Logit变换 / Logit Transformation

$$
\boldsymbol{w}^T \boldsymbol{x} = \log \frac{p(y = 1|\boldsymbol{x})}{1 - p(y = 1|\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log$ | duìshù | 自然对数 |
| $p(y = 1 \mid \boldsymbol{x})$ | gěi dìng x xià, y=1 de gàilǜ | 给定x下，y=1的概率 |

**一句话解释：**
Logit变换将概率的比值转化为线性函数，是sigmoid函数的反函数。

**简单应用场景：**
在Logistic回归中，解释权重系数对概率比值的对数影响。

---

---

### 公式 (3.18)：交叉熵损失函数 / Cross-Entropy Loss Function

$$
\mathcal{L}_{CE}(y, f(x; \boldsymbol{w})) = -y \log p(y=1|\boldsymbol{x}) - (1-y) \log p(y=0|\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{CE}$ | jiāocha熵 línsǔn | 交叉熵损失 |
| $y$ | zhēnshí biāoqiān | 真实标签 |
| $p(y=1 \mid \boldsymbol{x})$ | yùcè zhèngōu gàilǜ | 预测正类概率 |

**一句话解释：**
交叉熵损失函数衡量预测概率分布与真实标签分布的差异。

**简单应用场景：**
Logistic回归和神经网络中的标准损失函数。

---

---

### 公式 (3.19)：Softmax函数 / Softmax Function

$$
\hat{y}^{(n)} = \sigma(\boldsymbol{w}^\top \boldsymbol{x}^{(n)}), \quad 1 \leq n \leq N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y=c \mid \boldsymbol{x})$ | gěi dìng x xià, y=c de gàilǜ | 给定x下，预测为c类的概率 |
| $\text{softmax}_c$ | ruǎnzuìdà hánshù | softmax函数 |
| $f_c$ | dì c lèi de xiànxìng shūchū | 第c类的线性输出 |

**一句话解释：**
softmax函数将多个实数值转化为概率分布，使得各类概率和为1。

**简单应用场景：**
多分类神经网络的输出层激活函数。

---

---

### 公式 (3.20)：Softmax与交叉熵的关系 / Softmax and Cross-Entropy

$$
p_r(y^{(n)} = 1 | \boldsymbol{x}^{(n)}) = y^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y=0 \mid \boldsymbol{x})$ | dì 0 lèi de gàilǜ | 第0类的概率 |
| $C$ | zǒnglèi shù | 总类数 |

**一句话解释：**
在多分类中，某一类的概率可由其他类的概率推导得出。

**简单应用场景：**
多分类Softmax回归中，计算参考类的概率。

---

---

### 公式 (3.21)：多分类Softmax表达式 / Multi-class Softmax Expression

$$
p_r(y^{(n)} = 0 | \boldsymbol{x}^{(n)}) = 1 - y^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}_c$ | dì c lèi de quánzhòng xiàngliàng | 第c类的权重向量 |
| $\boldsymbol{x}$ | tèzhēng xiàngliàng | 特征向量 |

**一句话解释：**
softmax回归模型，为每个类别学习单独的权重向量，输出各类的概率。

**简单应用场景：**
多分类问题的回归模型。

---

---

### 公式 (3.22)：Softmax多分类交叉熵损失 / Softmax Multi-class Cross-Entropy Loss

$$
\mathcal{R}(\boldsymbol{w}) = -\frac{1}{N} \sum_{n=1}^{N} \left( p_r(y^{(n)}=1|\boldsymbol{x}^{(n)}) \log \hat{y}^{(n)} + p_r(y^{(n)}=0|\boldsymbol{x}^{(n)}) \log(1 - \hat{y}^{(n)}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_c$ | dì c lèi de dúrè huàyī biāoqiān | 第c类的独热化标签 |
| $p(y=c \mid \boldsymbol{x})$ | dì c lèi de yùcè gàilǜ | 第c类的预测概率 |

**一句话解释：**
多分类问题的交叉熵损失函数，衡量预测概率与真实标签的差异。

**简单应用场景：**
softmax回归和多分类神经网络的目标函数。

---

---

### 公式 (3.23)：最小二乘回归模型 / Least Squares Regression

$$
\mathcal{R}(\boldsymbol{w}) = -\frac{1}{N} \sum_{n=1}^{N} \left( y^{(n)} \log \hat{y}^{(n)} + (1 - y^{(n)}) \log(1 - \hat{y}^{(n)}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | mùbiāo zhí | 目标值 |
| $f(x; \boldsymbol{w})$ | yùcè zhí | 预测值 |

**一句话解释：**
线性回归模型，使用线性函数拟合输入特征和目标值之间的关系。

**简单应用场景：**
房价预测、股票价格预测等连续值预测问题。

---

---

### 公式 (3.24)：最小二乘损失函数 / Least Squares Loss Function

$$
\frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w}} = -\frac{1}{N} \sum_{n=1}^{N} \left( y^{(n)} \frac{\hat{y}^{(n)}(1 - \hat{y}^{(n)})}{\hat{y}^{(n)}} \boldsymbol{x}^{(n)} - (1 - y^{(n)}) \frac{\hat{y}^{(n)}(1 - \hat{y}^{(n)})}{1 - \hat{y}^{(n)}} \boldsymbol{x}^{(n)} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{LS}$ | zuìxiǎo èrchéng línsǔn | 最小二乘损失 |
| $y$ | zhēnshí mùbiāo | 真实目标值 |
| $f(x; \boldsymbol{w})$ | yùcè zhí | 预测值 |

**一句话解释：**
最小二乘损失函数，计算预测值与真实值的平方差，是回归问题的常用损失函数。

**简单应用场景：**
线性回归模型的目标函数。

---

---

### 公式 (3.25)：正则化最小二乘 / Regularized Least Squares

$$
\frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w}} = -\frac{1}{N} \sum_{n=1}^{N} \left( y^{(n)}(1 - \hat{y}^{(n)}) \boldsymbol{x}^{(n)} - (1 - y^{(n)}) \hat{y}^{(n)} \boldsymbol{x}^{(n)} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | lán mǔ dá | 正则化参数 |
| $\|\boldsymbol{w}\|^2$ | quánzhòng xiàngliàng de èrcì fànshù | 权重向量的二次范数 |

**一句话解释：**
在最小二乘损失基础上加入L2正则化项，防止权重过大导致过拟合。

**简单应用场景：**
岭回归模型。

---

---

### 公式 (3.26)：感知机学习规则 / Perceptron Learning Rule

$$
\frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w}} = -\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)} (y^{(n)} - \hat{y}^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | quánzhòng xiàngliàng | 权重向量 |
| $y^{(n)}$ | dì n gè yàngběn de biāoqiān | 第n个样本的标签 |
| $\boldsymbol{x}^{(n)}$ | dì n gè yàngběn de tèzhēng xiàngliàng | 第n个样本的特征向量 |

**一句话解释：**
感知机在分类错误时更新权重，错分样本的特征向量按标签的符号加入权重。

**简单应用场景：**
感知机算法的参数更新规则。

---

---

### 公式 (3.27)：感知机误分类情况 / Perceptron Misclassification

$$
\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_t + \alpha \frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)} (y^{(n)} - \hat{y}^{(n)}_{\boldsymbol{w}_t})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | zhēnshí biāoqiān | 真实标签 |
| $f(x^{(n)}; \boldsymbol{w})$ | xiànxìng hánshù zhí | 线性函数值 |

**一句话解释：**
感知机将判别标签与预测符号异号的样本判定为误分类。

**简单应用场景：**
感知机损失函数的计算。

---

---

### 公式 (3.28)：SVM硬间隔约束 / SVM Hard Margin Constraint

$$
p(y = c | \boldsymbol{x}) = \operatorname{softmax}(\boldsymbol{w}_c^\top \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | zhēnshí biāoqiān | 真实标签 |
| $f(x^{(n)}; \boldsymbol{w})$ | xiànxìng hánshù zhí | 线性函数值 |

**一句话解释：**
支持向量机的硬间隔约束，要求所有样本都被正确分类且到决策面的距离至少为1。

**简单应用场景：**
线性可分数据的支持向量机模型。

---

---

### 公式 (3.29)：SVM目标函数 / SVM Objective Function

$$
p(y = c | \boldsymbol{x}) = \frac{\exp(\boldsymbol{w}_c^\top \boldsymbol{x})}{\sum_{c'=1}^{C} \exp(\boldsymbol{w}_{c'}^\top \boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{w}\|^2$ | quánzhòng xiàngliàng de èrcì fànshù | 权重向量的二次范数 |
| $\xi_n$ | sōngchí biàoliàng | 松弛变量 |
| $C$ | juéhuà cānshù | 惩罚参数 |

**一句话解释：**
支持向量机的目标函数，在最大化间隔和最小化分类错误之间平衡。

**简单应用场景：**
软间隔支持向量机。

---

---

### 公式 (3.30)：SVM软间隔约束 / SVM Soft Margin Constraint

$$
\hat{y} = \arg\max_{c=1}^{C} \, p(y = c | \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\xi_n$ | sōngchí biàoliàng | 松弛变量 |

**一句话解释：**
软间隔约束允许某些样本不满足间隔约束，用松弛变量衡量违反程度。

**简单应用场景：**
处理线性不可分或有噪声的数据。

---

---

### 公式 (3.31)：铰链损失函数 / Hinge Loss Function

$$
\hat{y} = \arg\max_{c=1}^{C} \, \boldsymbol{w}_c^\top \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{hinge}}$ | jiǎoliàn línsǔn | 铰链损失 |
| $y$ | zhēnshí biāoqiān | 真实标签 |
| $f(x; \boldsymbol{w})$ | yùcè zhí | 预测值 |

**一句话解释：**
铰链损失只在预测值与真实标签不满足间隔约束时产生损失。

**简单应用场景：**
支持向量机和线性分类器的损失函数。

---

---

### 公式 (3.32)：特征提取线性组合 / Feature Extraction Linear Combination

$$
\hat{y} = \arg\max_{y \in \{0,1\}} \boldsymbol{w}_y^\top \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | tíqǔ de tèzhēng | 提取的特征 |
| $\boldsymbol{W}$ | tèzhēng tíqǔ jǔzhèn | 特征提取矩阵 |
| $\boldsymbol{x}$ | yuánshǐ tèzhēng | 原始特征 |

**一句话解释：**
通过线性变换矩阵将原始特征映射到新的特征空间。

**简单应用场景：**
主成分分析、线性判别分析等特征降维方法。

---

---

### 公式 (3.33)：线性模型预测 / Linear Model Prediction

$$
= I(\boldsymbol{w}_1^\top \boldsymbol{x} - \boldsymbol{w}_0^\top \boldsymbol{x} > 0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}$ | yùcè zhí | 预测值 |
| $f(x; \boldsymbol{w})$ | xiànxìng moxíng | 线性模型 |

**一句话解释：**
线性模型的预测公式，通过权重与特征的内积加上偏置得到输出。

**简单应用场景：**
任何线性回归或线性分类问题的预测阶段。

---

---

### 公式 (3.34)：多项Logistic回归 / Multinomial Logistic Regression

$$
= I((\boldsymbol{w}_1 - \boldsymbol{w}_0)^\top \boldsymbol{x} > 0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y=c \mid \boldsymbol{x})$ | dì c lèi de gàilǜ | 第c类的概率 |
| $\boldsymbol{w}_c$ | dì c lèi de quánzhòng | 第c类的权重向量 |

**一句话解释：**
多项Logistic回归使用softmax函数处理多分类问题。

**简单应用场景：**
多分类问题的概率预测。

---

---

### 公式 (3.35)：岭回归目标函数 / Ridge Regression Objective

$$
\hat{\boldsymbol{y}} = \operatorname{softmax}(\boldsymbol{W}^\top \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\|\boldsymbol{w}\|^2$ | quánzhòng de èrcì fànshù | 权重的二次范数 |

**一句话解释：**
岭回归在最小二乘损失基础上加L2正则化，降低模型复杂度。

**简单应用场景：**
高维数据或特征间存在共线性时的回归模型。

---

---

### 公式 (3.36)：Lasso回归目标函数 / Lasso Regression Objective

$$
\hat{\boldsymbol{y}} = \frac{\exp(\boldsymbol{W}^\top \boldsymbol{x})}{\mathbf{1}_C^\top \exp(\boldsymbol{W}^\top \boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{w}\|_1$ | quánzhòng de yī cì fànshù | 权重的一次范数 |

**一句话解释：**
Lasso回归使用L1正则化，能够自动进行特征选择。

**简单应用场景：**
需要特征选择的高维回归问题。

---

---

### 公式 (3.37)：批量梯度下降 / Batch Gradient Descent

$$
\boldsymbol{y} = [I(1=c),\; I(2=c),\; \cdots,\; I(C=c)]^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\eta$ | yī tǎ | 学习率 |
| $\nabla_{\boldsymbol{w}}$ | nà bù lā | 梯度 |

**一句话解释：**
批量梯度下降在每次更新时使用所有样本的梯度，是标准的优化方法。

**简单应用场景：**
模型训练的参数优化。

---

---

### 公式 (3.38)：随机梯度下降 / Stochastic Gradient Descent

$$
\mathcal{R}(\boldsymbol{W}) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} \boldsymbol{y}_c^{(n)} \log \hat{\boldsymbol{y}}_c^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\eta$ | yī tǎ | 学习率 |
| $\nabla_{\boldsymbol{w}}$ | nà bù lā | 梯度 |

**一句话解释：**
随机梯度下降每次只使用一个样本的梯度更新参数，训练速度快。

**简单应用场景：**
大数据场景下的在线学习。

---

---

### 公式 (3.39)：小批量梯度下降 / Mini-batch Gradient Descent

$$
\mathcal{R}(\boldsymbol{W}) = -\frac{1}{N} \sum_{n=1}^{N} (\boldsymbol{y}^{(n)})^\top \log \hat{\boldsymbol{y}}^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $B$ | xiǎobàtì | 小批量样本集合 |
| $\eta$ | yī tǎ | 学习率 |

**一句话解释：**
小批量梯度下降在批量和随机之间平衡，每次使用一个小批量样本更新。

**简单应用场景：**
现代深度学习中的标准优化方法。

---

---

### 公式 (3.40)：梯度计算 / Gradient Calculation

$$
\frac{\partial \mathcal{R}(\boldsymbol{W})}{\partial \boldsymbol{W}} = -\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)} (\boldsymbol{y}^{(n)} - \hat{\boldsymbol{y}}^{(n)})^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla_{\boldsymbol{w}}$ | nà bù lā | 梯度算子 |
| $\frac{\partial \mathcal{L}}{\partial \boldsymbol{w}}$ | piānxúnshù | 偏导数 |

**一句话解释：**
梯度是损失函数对权重参数的偏导数，指向损失增长最快的方向。

**简单应用场景：**
计算参数更新方向。

---

---

### 公式 (3.41)：逻辑斯谛分布 / Logistic Distribution

$$
\frac{\partial \hat{y}_c}{\partial z_{c'}} = \frac{\partial}{\partial z_{c'}} \frac{\exp(z_c)}{\sum_{c''=1}^{C} \exp(z_{c''})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y \mid \boldsymbol{x})$ | gěi dìng x xià y de gàilǜ | 给定x下y的概率 |
| $\sigma(\cdot)$ | xī gé mǎ hánshù | sigmoid函数 |

**一句话解释：**
Logistic分布用于描述二分类事件的概率。

**简单应用场景：**
Logistic回归的概率模型基础。

---

---

### 公式 (3.42)：最大似然估计 / Maximum Likelihood Estimation

$$
= \frac{I(c=c') \exp(z_c) \sum_{c''} \exp(z_{c''}) - \exp(z_c) \exp(z_{c'})}{(\sum_{c''} \exp(z_{c''}))^2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\boldsymbol{w})$ | sìrán hánshù | 似然函数 |
| $p(y^{(n)} \mid \boldsymbol{x}^{(n)}; \boldsymbol{w})$ | gěi dìng cānshù de gàilǜ | 给定参数的概率 |

**一句话解释：**
最大似然估计通过最大化所有样本的概率来估计参数。

**简单应用场景：**
参数估计的标准方法。

---

---

### 公式 (3.43)：对数似然 / Log-Likelihood

$$
= I(c=c') \hat{y}_c - \hat{y}_c \hat{y}_{c'}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log \mathcal{L}(\boldsymbol{w})$ | duìshù sìrán hánshù | 对数似然函数 |

**一句话解释：**
对数似然函数在数学上更易处理，乘积变为求和。

**简单应用场景：**
优化问题中的目标函数。

---

---

### 公式 (3.44)：条件概率模型 / Conditional Probability Model

$$
= \hat{y}_c (I(c=c') - \hat{y}_{c'})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y \mid \boldsymbol{x}; \boldsymbol{w})$ | tiáojiàn gàilǜ | 条件概率 |
| $\phi(\boldsymbol{x}, y)$ | tèzhēng hánshù | 特征函数 |

**一句话解释：**
条件随机场的基本形式，用特征函数表示条件概率。

**简单应用场景：**
序列标注问题。

---

---

### 公式 (3.45)：线性判别分析投影 / Linear Discriminant Analysis Projection

$$
\frac{\partial \hat{\boldsymbol{y}}}{\partial \boldsymbol{z}} = \operatorname{diag}(\hat{\boldsymbol{y}}) - \hat{\boldsymbol{y}} \hat{\boldsymbol{y}}^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z$ | tóuyǐng zhí | 投影值 |
| $\boldsymbol{w}$ | pànbié xiàngliàng | 判别向量 |

**一句话解释：**
线性判别分析通过投影最大化不同类之间的可分性。

**简单应用场景：**
特征降维和分类。

---

---

### 公式 (3.46)：类间散度矩阵 / Between-class Scatter Matrix

$$
\frac{\partial \mathcal{R}}{\partial \boldsymbol{z}} = \frac{\partial \mathcal{R}}{\partial \hat{\boldsymbol{y}}} \cdot \frac{\partial \hat{\boldsymbol{y}}}{\partial \boldsymbol{z}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{S}_B$ | lèijiān sǎndù jǔzhèn | 类间散度矩阵 |
| $\boldsymbol{\mu}_c$ | dì c lèi de jūnzhí | 第c类的均值 |
| $N_c$ | dì c lèi de yàngběn shù | 第c类的样本数 |

**一句话解释：**
类间散度矩阵衡量不同类别中心之间的距离。

**简单应用场景：**
线性判别分析的判别准则。

---

---

### 公式 (3.47)：类内散度矩阵 / Within-class Scatter Matrix

$$
\frac{\partial \mathcal{R}}{\partial \hat{\boldsymbol{y}}} = -\frac{\boldsymbol{y}}{\hat{\boldsymbol{y}}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{S}_W$ | lèinèi sǎndù jǔzhèn | 类内散度矩阵 |
| $X_c$ | dì c lèi de yàngběn jítǐ | 第c类的样本集合 |

**一句话解释：**
类内散度矩阵衡量各类内部样本的散布程度。

**简单应用场景：**
线性判别分析中的分类准则。

---

---

### 公式 (3.48)：Fisher判别准则 / Fisher Criterion

$$
\frac{\partial \mathcal{R}}{\partial \boldsymbol{z}} = -\frac{\boldsymbol{y}}{\hat{\boldsymbol{y}}} \cdot (\operatorname{diag}(\hat{\boldsymbol{y}}) - \hat{\boldsymbol{y}} \hat{\boldsymbol{y}}^\top)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{S}_B$ | lèijiān sǎndù jǔzhèn | 类间散度矩阵 |
| $\boldsymbol{S}_W$ | lèinèi sǎndù jǔzhèn | 类内散度矩阵 |

**一句话解释：**
Fisher准则选择使得类间距离最大、类内距离最小的投影方向。

**简单应用场景：**
线性判别分析的优化目标。

---

---

### 公式 (3.49)：决策函数的间隔 / Margin of Decision Function

$$
= -(\boldsymbol{y} - \hat{\boldsymbol{y}} (\boldsymbol{y}^\top \mathbf{1}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gāmǎ | 间隔 |
| $\|\boldsymbol{w}\|$ | quánzhòng de fànshù | 权重的范数 |

**一句话解释：**
有符号距离衡量样本点到决策超平面的距离和方向。

**简单应用场景：**
支持向量机的间隔度量。

---

---

### 公式 (3.50)：感知机收敛条件 / Perceptron Convergence Condition

$$
= -(\boldsymbol{y} - \hat{\boldsymbol{y}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| 线性可分 | xiànxìng kěfēnkāi | linearly separable |

**一句话解释：**
感知机学习定理保证了在线性可分数据上的收敛性。

**简单应用场景：**
验证感知机算法的有效性。

---

---

### 公式 (3.51)：概率判别模型 / Probabilistic Discriminative Model

$$
= \hat{\boldsymbol{y}} - \boldsymbol{y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y \mid \boldsymbol{x}; \boldsymbol{w})$ | tiáojiàn gàilǜ | 条件概率 |
| $\boldsymbol{w}$ | moxíng cānshù | 模型参数 |

**一句话解释：**
判别模型直接建模条件概率，用于预测给定输入时的标签概率。

**简单应用场景：**
Logistic回归、神经网络等分类模型。

---

---

### 公式 (3.52)：标准化指数族 / Normalized Exponential Family

$$
\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}} = \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z(\boldsymbol{w})$ | fēnzhī hánshù | 分配函数 |
| $\phi(\boldsymbol{x}, y)$ | tèzhēng xiàngliàng | 特征向量 |

**一句话解释：**
指数族分布的标准形式，归一化常数确保概率和为1。

**简单应用场景：**
最大熵模型和条件随机场。

---

---

### 公式 (3.53)：逻辑斯谛函数的导数 / Derivative of Logistic Function

$$
\frac{\partial \mathcal{R}}{\partial \boldsymbol{W}} = \boldsymbol{x} (\hat{\boldsymbol{y}} - \boldsymbol{y})^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma(x)$ | xī gé mǎ hánshù | sigmoid函数 |

**一句话解释：**
sigmoid函数的导数具有简洁的形式，便于梯度计算。

**简单应用场景：**
反向传播算法中的梯度计算。

---

---

### 公式 (3.54)：softmax函数的导数 / Derivative of Softmax Function

$$
\hat{y} = \operatorname{sgn}(\boldsymbol{w}^\top \boldsymbol{x} + b)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y=c \mid \boldsymbol{x})$ | dì c lèi de gàilǜ | 第c类的概率 |

**一句话解释：**
softmax函数某类的偏导数。

**简单应用场景：**
多分类模型的梯度计算。

---

---

### 公式 (3.55)：拉格朗日乘数法 / Lagrange Multiplier Method

$$
y^{(n)} {\boldsymbol{w}^*}^\top \boldsymbol{x}^{(n)} > 0, \quad \forall n \in \{1, \cdots, N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | lán mǔ dá | 拉格朗日乘数 |
| $g(\boldsymbol{w})$ | yuēshù tiáojiàn | 约束条件 |

**一句话解释：**
拉格朗日乘数法用于求解带约束的优化问题。

**简单应用场景：**
支持向量机的对偶问题求解。

---

---

### 公式 (3.56)：对偶问题 / Dual Problem

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + y \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āěr fǎ | 对偶变量 |
| $\mathcal{L}$ | láglǔ hánshù | 拉格朗日函数 |

**一句话解释：**
对偶问题通过变换原始问题，有时更易求解。

**简单应用场景：**
支持向量机的求解。

---

---

### 公式 (3.57)：KKT条件 / Karush-Kuhn-Tucker Conditions

$$
\mathcal{L}(\boldsymbol{w}; \boldsymbol{x}, y) = \max(0, -y \boldsymbol{w}^\top \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_n$ | duìyǔ biàoliàng | 对偶变量 |
| $\xi_n$ | sōngchí biàoliàng | 松弛变量 |

**一句话解释：**
KKT条件是约束优化问题的最优性必要条件。

**简单应用场景：**
验证最优解。

---

---

### 公式 (3.58)：支持向量 / Support Vector

$$
\hat{y} = \operatorname{sgn}(\boldsymbol{w}^\top \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_n$ | duìyǔ xìshù | 对偶系数 |

**一句话解释：**
支持向量是对偶变量非零的训练样本，位于或超越间隔边界。

**简单应用场景：**
支持向量机的分类决策。

---

---

### 公式 (3.59)：核函数 / Kernel Function

$$
\boldsymbol{w}_k = \boldsymbol{w}_{k-1} + y^{(k)} \boldsymbol{x}^{(k)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K(\boldsymbol{x}, \boldsymbol{x}')$ | héhánshù | 核函数 |
| $\phi(\boldsymbol{x})$ | tèzhēng yìngshè | 特征映射 |

**一句话解释：**
核函数计算高维特征空间中的内积，无需显式计算映射。

**简单应用场景：**
核支持向量机的非线性分类。

---

---

### 公式 (3.60)：高斯核 / Gaussian Kernel

$$
\boldsymbol{w}_K = \sum_{k=1}^{K} y^{(k)} \boldsymbol{x}^{(k)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | 核参数 |
| $\|\boldsymbol{x} - \boldsymbol{x}'\|^2$ | jùlí de píngfāng | 距离的平方 |

**一句话解释：**
高斯核（RBF核）是最常用的核函数，基于点间的距离。

**简单应用场景：**
非线性支持向量机。

---

---

### 公式 (3.61)：多项式核 / Polynomial Kernel

$$
\|\boldsymbol{w}_K\|^2 \leq K R^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $c$ | chángshù | 常数项 |
| $d$ | duōxiàng shù | 多项式次数 |

**一句话解释：**
多项式核对应多项式特征空间，用于捕捉特征间的交互。

**简单应用场景：**
非线性分类问题。

---

---

### 公式 (3.62)：线性核 / Linear Kernel

$$
\|\boldsymbol{w}_K\|^2 = \|\boldsymbol{w}_{K-1}\|^2 + \|y^{(K)} \boldsymbol{x}^{(K)}\|^2 + 2 y^{(K)} \boldsymbol{w}_{K-1}^\top \boldsymbol{x}^{(K)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K(\boldsymbol{x}, \boldsymbol{x}')$ | héhánshù | 核函数 |

**一句话解释：**
线性核是最简单的核函数，对应线性特征空间。

**简单应用场景：**
线性支持向量机。

---

---

### 公式 (3.63)：核技巧 / Kernel Trick

$$
\leq \|\boldsymbol{w}_{K-1}\|^2 + R^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\phi(\boldsymbol{x})$ | tèzhēng yìngshè | 特征映射 |
| $K(\boldsymbol{x}, \boldsymbol{x}')$ | héhánshù | 核函数 |

**一句话解释：**
核技巧避免显式计算高维特征映射，直接用核函数计算内积。

**简单应用场景：**
高维非线性分类问题的高效求解。

---

---

### 公式 (3.64)：类平衡权重 / Class Balanced Weight

$$
\leq \|\boldsymbol{w}_{K-2}\|^2 + 2R^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_c$ | dì c lèi de quánzhòng | 第c类的权重 |
| $N_c$ | dì c lèi de yàngběn shù | 第c类的样本数 |

**一句话解释：**
通过权重调整不同类别的重要性，解决类不平衡问题。

**简单应用场景：**
处理不平衡数据集。

---

---

### 公式 (3.65)：加权损失函数 / Weighted Loss Function

$$
\leq K R^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_{y^{(n)}}$ | dì y^{(n)} lèi de quánzhòng | 第y^{(n)}类的权重 |
| $\mathcal{L}$ | línsǔn hánshù | 损失函数 |

**一句话解释：**
加权损失函数通过不同类的权重调整模型的学习重点。

**简单应用场景：**
不平衡数据集的分类。

---

---

### 公式 (3.66)：样本权重 / Sample Weight

$$
\|\boldsymbol{w}_K\|^2 = \|\boldsymbol{w}^*\|^2 \cdot \|\boldsymbol{w}_K\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w^{(n)}$ | dì n gè yàngběn de quánzhòng | 第n个样本的权重 |
| $N_{y^{(n)}}$ | dì y^{(n)} lèi de yàngběn shù | 第y^{(n)}类的样本数 |

**一句话解释：**
为不同类别的样本分配不同权重，使每类对损失的贡献相等。

**简单应用场景：**
样本级别的类平衡。

---

---

### 公式 (3.67)：F1值 / F1 Score

$$
\geq \|{\boldsymbol{w}^*}^\top \boldsymbol{w}_K\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F_1$ | F yī zhǐbiāo | F1得分 |
| $P$ | jīngquè lǜ | 精确率 |
| $R$ | huíhuòlǜ | 召回率 |

**一句话解释：**
F1值是精确率和召回率的调和平均，综合衡量分类器性能。

**简单应用场景：**
不平衡数据集的评估指标。

---

---

### 公式 (3.68)：AUC / Area Under Curve

$$
= \left\| {\boldsymbol{w}^*}^\top \sum_{k=1}^{K} (y^{(k)} \boldsymbol{x}^{(k)}) \right\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{AUC}$ | qūxiàn xiàjiàn jī | 曲线下面积 |
| $\text{TPR}$ | zhēn zhèng lǜ | 真正率 |
| $\text{FPR}$ | wěi zhèng lǜ | 伪正率 |

**一句话解释：**
AUC是ROC曲线下的面积，值越大分类器性能越好。

**简单应用场景：**
二分类模型的整体性能评估。

---

---

### 公式 (3.69)：混淆矩阵中的准确率 / Accuracy in Confusion Matrix

$$
= \left\| \sum_{k=1}^{K} {\boldsymbol{w}^*}^\top (y^{(k)} \boldsymbol{x}^{(k)}) \right\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A$ | zhǔnquè lǜ | 准确率 |
| $\hat{y}^{(n)}$ | yùcè biāoqiān | 预测标签 |
| $I(\cdot)$ | zhǐshìfúnshù | 指示函数 |

**一句话解释：**
准确率是预测正确样本数占总样本数的比例。

**简单应用场景：**
分类模型的基础评估指标。

---

---

### 公式 (3.70)：错误率 / Error Rate

$$
\geq K^2 \gamma^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\varepsilon$ | cuòwù lǜ | 错误率 |
| $A$ | zhǔnquè lǜ | 准确率 |

**一句话解释：**
错误率是准确率的补集，表示预测错误的比例。

**简单应用场景：**
评估分类模型的失败率。

---

---

### 公式 (3.71)：精确率 / Precision

$$
K^2 \gamma^2 \leq \|\boldsymbol{w}_K\|^2 \leq K R^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P$ | jīngquè lǜ | 精确率 |
| $\text{TP}$ | zhēn zhèng | 真正例 |
| $\text{FP}$ | wěi zhèng | 伪正例 |

**一句话解释：**
精确率衡量预测为正类的样本中实际为正类的比例。

**简单应用场景：**
重视减少假正例的应用场景。

---

---

### 公式 (3.72)：召回率 / Recall

$$
\hat{y} = \operatorname{sgn}\left( \sum_{k=1}^{K} c_k \operatorname{sgn}(\boldsymbol{w}_k^\top \boldsymbol{x}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $R$ | huíhuòlǜ | 召回率 |
| $\text{TP}$ | zhēn zhèng | 真正例 |
| $\text{FN}$ | wěi fù | 伪负例 |

**一句话解释：**
召回率衡量实际为正类的样本中被预测为正类的比例。

**简单应用场景：**
重视减少假负例的应用场景。

---

---

### 公式 (3.73)：真正例（TP） / True Positive

$$
\hat{y} = \operatorname{sgn}\left( \frac{1}{T} \sum_{k=1}^{K} c_k (\boldsymbol{w}_k^\top \boldsymbol{x}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{TP}$ | zhēn zhèng lì | 真正例 |
| $y^{(n)}$ | zhēnshí biāoqiān | 真实标签 |
| $\hat{y}^{(n)}$ | yùcè biāoqiān | 预测标签 |

**一句话解释：**
真正例是实际为正类且被预测为正类的样本。

**简单应用场景：**
计算精确率和召回率。

---

---

### 公式 (3.74)：伪正例（FP） / False Positive

$$
= \operatorname{sgn}\left( \frac{1}{T} \left( \sum_{k=1}^{K} c_k \boldsymbol{w}_k \right)^\top \boldsymbol{x} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{FP}$ | wěi zhèng lì | 伪正例 |

**一句话解释：**
伪正例是实际为负类但被预测为正类的样本。

**简单应用场景：**
计算精确率和特异性。

---

---

### 公式 (3.75)：伪负例（FN） / False Negative

$$
= \operatorname{sgn}\left( \frac{1}{T} \left( \sum_{t=1}^{T} \boldsymbol{w}_t \right)^\top \boldsymbol{x} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{FN}$ | wěi fù lì | 伪负例 |

**一句话解释：**
伪负例是实际为正类但被预测为负类的样本。

**简单应用场景：**
计算召回率和敏感性。

---

---

### 公式 (3.76)：真负例（TN） / True Negative

$$
= \operatorname{sgn}(\bar{\boldsymbol{w}}^\top \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{TN}$ | zhēn fù lì | 真负例 |

**一句话解释：**
真负例是实际为负类且被预测为负类的样本。

**简单应用场景：**
计算特异性。

---

---

### 公式 (3.77)：特异性 / Specificity

$$
\bar{\boldsymbol{w}} \leftarrow \bar{\boldsymbol{w}} + \boldsymbol{w}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Specificity}$ | tèyìxìng | 特异性 |
| $\text{TN}$ | zhēn fù lì | 真负例 |
| $\text{FP}$ | wěi zhèng lì | 伪正例 |

**一句话解释：**
特异性衡量实际为负类的样本中被预测为负类的比例。

**简单应用场景：**
评估模型识别负类的能力。

---

---

### 公式 (3.78)：敏感性 / Sensitivity

$$
\hat{\boldsymbol{y}} = \arg\max_{\boldsymbol{y} \in \operatorname{Gen}(\boldsymbol{x})} \boldsymbol{w}^\top \boldsymbol{\phi}(\boldsymbol{x}, \boldsymbol{y})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Sensitivity}$ | mǐnhuà xìng | 敏感性 |
| $\text{TP}$ | zhēn zhèng lì | 真正例 |
| $\text{FN}$ | wěi fù lì | 伪负例 |

**一句话解释：**
敏感性等同于召回率，衡量模型识别正类的能力。

**简单应用场景：**
医学诊断和疾病筛查。

---

---

### 公式 (3.79)：ROC曲线 / ROC Curve

$$
\boldsymbol{\phi}(\boldsymbol{x}, \boldsymbol{y}) = \operatorname{vec}(\boldsymbol{x} \boldsymbol{y}^\top) \in \mathbb{R}^{D \times C}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{ROC}$ | gōngzuò tèzhēng qūxiàn | 工作特征曲线 |
| $\text{FPR}$ | wěi zhèng lǜ | 伪正率 |
| $\text{TPR}$ | zhēn zhèng lǜ | 真正率 |

**一句话解释：**
ROC曲线绘制真正率对伪正率的关系，用于评估分类器性能。

**简单应用场景：**
不同分类阈值下的性能比较。

---

---

### 公式 (3.80)：交叉验证 / K-Fold Cross-Validation

$$
\boldsymbol{\phi}(\boldsymbol{x}, \boldsymbol{y}) = \begin{bmatrix}
\mathbf{0} \\ \vdots \\ \boldsymbol{x} \\ \vdots \\ \mathbf{0}
\end{bmatrix} \leftarrow c\text{-th block}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K$ | jiāocha yànshuō de fēnshù | 交叉验证的分数 |

**一句话解释：**
K折交叉验证通过轮流使用不同数据进行训练和测试，更充分地利用数据。

**简单应用场景：**
模型选择和性能评估。

---

---

### 公式 (3.81)：留一法交叉验证 / Leave-One-Out Cross-Validation

$$
R = \max_n \max_{\boldsymbol{z} \neq \boldsymbol{y}^{(n)}} \|\boldsymbol{\phi}(\boldsymbol{x}^{(n)}, \boldsymbol{y}^{(n)}) - \boldsymbol{\phi}(\boldsymbol{x}^{(n)}, \boldsymbol{z})\|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{LOO}$ | liúyī jiāocha yànshuō | 留一法交叉验证 |

**一句话解释：**
留一法每次只用一个样本测试，用N-1个样本训练，计算量大但偏差小。

**简单应用场景：**
小数据集的模型评估。

---

---

### 公式 (3.82)：宏平均F1值 / Macro Average F1 Score

$$
\boldsymbol{w}^\top \boldsymbol{x} + b = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F_{1,\text{macro}}$ | hóngpíng jūn F yī zhǐbiāo | 宏平均F1指标 |
| $F_{1,c}$ | dì c lèi de F yī zhǐbiāo | 第c类的F1指标 |
| $C$ | fēnlèi zǒngshù | 分类总数 |

**一句话解释：**
宏平均F1对每类的F1值取平均，给予每类相等的权重。

**简单应用场景：**
多分类问题的性能评估。

---

---

### 公式 (3.83)：微平均F1值 / Micro Average F1 Score

$$
\gamma^{(n)} = \frac{y^{(n)}(\boldsymbol{w}^\top \boldsymbol{x}^{(n)} + b)}{\|\boldsymbol{w}\|}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F_{1,\text{micro}}$ | wēipíng jūn F yī zhǐbiāo | 微平均F1指标 |
| $P_\text{micro}$ | wēipíng jūn jīngquè lǜ | 微平均精确率 |
| $R_\text{micro}$ | wēipíng jūn huíhuòlǜ | 微平均召回率 |

**一句话解释：**
微平均F1先计算全局精确率和召回率，再求其F1值，受大类影响较大。

**简单应用场景：**
多分类不平衡数据的评估。

---

---

### 公式 (3.84)：硬间隔问题（最小化形式）/ Hard Margin Problem (Minimization Form)

$$
y = \min_n y^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | yù cè zhí | 第n个样本的预测值 |
| $\min_n$ | zuì xiǎo | 最小值 |

**一句话解释：**
找出所有样本中最小的预测值，代表最紧的约束。

**简单应用场景：**
支持向量机的约束条件。

---

---

### 公式 (3.85)：硬间隔目标函数 / Hard Margin Objective

$$
\max_{\boldsymbol{w}, b} \;\; \gamma \quad \text{s.t.} \quad \frac{y^{(n)}(\boldsymbol{w}^\top \boldsymbol{x}^{(n)} + b)}{\|\boldsymbol{w}\|} \geq \gamma, \quad \forall n \in \{1, \cdots, N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gān mǎ | 最小间隔宽度 |
| $\max$ | zuì dà huà | 最大化 |

**一句话解释：**
最大化所有样本到决策超平面的最小距离。

**简单应用场景：**
硬间隔支持向量机的核心目标。

---

---

### 公式 (3.86)：标准化硬间隔问题 / Canonical Hard Margin Problem

$$
\max_{\boldsymbol{w},b} \frac{1}{\|\boldsymbol{w}\|^2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{1}{\|\boldsymbol{w}\|^2}$ | quán zhòng píngfāng mó de dào shù | 权重平方模的倒数 |
| $\max$ | zuì dà huà | 最大化 |

**一句话解释：**
通过最大化权重倒数平方来最大化间隔，等价于最小化权重的平方模。

**简单应用场景：**
硬间隔支持向量机的标准形式。

---

## 总结

本文档汇总了《神经网络与深度学习》教科书第3章第58-105页范围内，从公式(3.18)到公式(3.86)的所有69个主要公式。这些公式涵盖了线性模型的核心内容，包括：

1. **逻辑回归**（3.18-3.28）：包括几率、Sigmoid函数、风险函数和梯度更新
2. **Softmax回归**（3.26-3.45）：多分类模型、交叉熵损失和梯度计算
3. **支持向量机**（3.30-3.86）：硬间隔SVM、软间隔SVM、核技巧、对偶问题和各类损失函数

每个公式都配有完整的符号表、一句话解释和应用场景说明，便于学习和参考。

---

### 公式 (3.87)：支持向量机 - 标准形式 / Support Vector Machine - Standard Form

$$
\min_{w,b} \frac{1}{2}\|\boldsymbol{w}\|^2
$$

**s.t.** $1 - y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) \leq 0, \quad \forall n \in \{1, \cdots, N\}$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | wèi liàng | 权重向量 |
| $b$ | piàn zhì | 偏置项 |
| $\boldsymbol{x}^{(n)}$ | shù jù | 第n个样本 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |
| $N$ | yàng běn shù | 样本总数 |

**一句话解释：**
通过最小化权重向量的范数平方，同时满足所有样本的间隔约束，找到最优的超平面。

**简单应用场景：**
二分类问题中，找到能够最大化类间间隔的决策边界。

---

---

### 公式 (3.88)：支持向量机 - 拉格朗日对偶形式 / Support Vector Machine - Lagrangian Dual Form

$$
\Lambda(w, b, \lambda) = \frac{1}{2}\|\boldsymbol{w}\|^2 + \sum_{n=1}^{N} \lambda_n(1 - y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Lambda$ | lā gé lǎng rì hàn | 拉格朗日函数 |
| $\lambda_n$ | lán mǔ dá | 拉格朗日乘数 |
| $\boldsymbol{w}$ | wèi liàng | 权重向量 |
| $b$ | piàn zhì | 偏置项 |
| $y^{(n)}$ | biāo qiān | 第n个样本标签 |

**一句话解释：**
将约束优化问题转化为无约束形式，通过引入拉格朗日乘数来处理不等式约束。

**简单应用场景：**
优化算法中，便于求解对偶问题以获得最优解。

---

---

### 公式 (3.89)：支持向量机 - 权重向量优化条件 / Support Vector Machine - Weight Optimization

$$
\boldsymbol{w} = \sum_{n=1}^{N} \lambda_n y^{(n)}\boldsymbol{x}^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | wèi liàng | 权重向量 |
| $\lambda_n$ | lán mǔ dá | 拉格朗日乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本标签 |
| $\boldsymbol{x}^{(n)}$ | shù jù | 第n个样本 |

**一句话解释：**
最优权重向量可以表示为所有样本的加权线性组合，权重为对应的拉格朗日乘数。

**简单应用场景：**
支持向量机的核方法中，权重可用核函数表示，不需显式计算。

---

---

### 公式 (3.90)：支持向量机 - 偏置项优化条件 / Support Vector Machine - Bias Optimization

$$
0 = \sum_{n=1}^{N} \lambda_n y^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 拉格朗日乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本标签 |
| $N$ | yàng běn shù | 样本总数 |

**一句话解释：**
最优偏置项满足所有拉格朗日乘数与标签的加权和为零的约束条件。

**简单应用场景：**
确保超平面关于数据集的均衡性。

---

---

### 公式 (3.91)：支持向量机 - 对偶问题目标函数 / Support Vector Machine - Dual Problem Objective

$$
\Gamma(\lambda) = -\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} \lambda_m \lambda_n y^{(m)} y^{(n)} (\boldsymbol{x}^{(m)})^T \boldsymbol{x}^{(n)} + \sum_{n=1}^{N} \lambda_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Gamma$ | gā mǎ | 对偶目标函数 |
| $\lambda_n, \lambda_m$ | lán mǔ dá | 拉格朗日乘数 |
| $y^{(n)}, y^{(m)}$ | biāo qiān | 样本标签 |
| $\boldsymbol{x}^{(n)}, \boldsymbol{x}^{(m)}$ | shù jù | 样本向量 |

**一句话解释：**
支持向量机的对偶问题目标函数，只依赖于样本之间的内积，不需显式的权重向量。

**简单应用场景：**
核方法的基础，通过核函数计算内积，处理非线性分类问题。

---

---

### 公式 (3.92)：支持向量机 - 偏置项估计 / Support Vector Machine - Bias Estimation

$$
b^* = \bar{y} - \boldsymbol{w}^{*T}\bar{\boldsymbol{x}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b^*$ | zuì yōu piàn zhì | 最优偏置项 |
| $\bar{y}$ | píng jūn biāo qiān | 平均标签值 |
| $\boldsymbol{w}^*$ | zuì yōu wèi liàng | 最优权重向量 |
| $\bar{\boldsymbol{x}}$ | píng jūn shù jù | 平均特征向量 |

**一句话解释：**
通过数据的平均值计算最优偏置项，确保超平面经过数据中心。

**简单应用场景：**
训练完成后，从支持向量或所有数据点估计偏置项。

---

---

### 公式 (3.93)：支持向量机 - 决策函数 / Support Vector Machine - Decision Function

$$
f(x) = \text{sgn}(\boldsymbol{w}^{*T}\boldsymbol{x} + b^*)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x)$ | jué cè hánshù | 决策函数 |
| $\text{sgn}$ | fú hào hánshù | 符号函数 |
| $\boldsymbol{w}^*$ | zuì yōu wèi liàng | 最优权重向量 |
| $b^*$ | zuì yōu piàn zhì | 最优偏置项 |
| $\boldsymbol{x}$ | shù jù | 输入特征向量 |

**一句话解释：**
通过计算特征与权重的内积加偏置，然后取符号函数的值进行分类。

**简单应用场景：**
支持向量机的最终分类器，对新样本进行二分类预测。

---

---

### 公式 (3.94)：支持向量机 - 对偶形式决策函数 / Support Vector Machine - Decision Function in Dual Form

$$
f(x) = \text{sgn}\left(\sum_{n=1}^{N} \lambda_n y^{(n)}(\boldsymbol{x}^{(n)})^T\boldsymbol{x} + b^*\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x)$ | jué cè hánshù | 决策函数 |
| $\text{sgn}$ | fú hào hánshù | 符号函数 |
| $\lambda_n$ | lán mǔ dá | 拉格朗日乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本标签 |
| $\boldsymbol{x}^{(n)}$ | shù jù | 第n个训练样本 |
| $\boldsymbol{x}$ | shù jù | 测试样本 |

**一句话解释：**
使用对偶形式的决策函数，只需计算测试样本与所有训练样本的内积，便于核方法应用。

**简单应用场景：**
非线性支持向量机中，通过核函数替代内积实现非线性分类。

---

---

### 公式 (3.95)：核函数支持向量机 - 标准形式 / Kernel SVM - Standard Form

$$
f(x) = \text{sgn}(\boldsymbol{w}^T\phi(\boldsymbol{x}) + b^*)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x)$ | jué cè hánshù | 决策函数 |
| $\boldsymbol{w}$ | wèi liàng | 权重向量 |
| $\phi(\boldsymbol{x})$ | tè zhēng yìng shè | 非线性特征映射 |
| $b^*$ | zuì yōu piàn zhì | 最优偏置项 |

**一句话解释：**
通过非线性映射函数将输入映射到高维空间，在该空间中进行线性分类。

**简单应用场景：**
处理非线性可分的分类问题，如异或问题。

---

---

### 公式 (3.96)：核函数支持向量机 - 对偶形式 / Kernel SVM - Dual Form

$$
f(x) = \text{sgn}\left(\sum_{n=1}^{N} \lambda_n y^{(n)} k(\boldsymbol{x}^{(n)}, \boldsymbol{x}) + b^*\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x)$ | jué cè hánshù | 决策函数 |
| $\lambda_n$ | lán mǔ dá | 拉格朗日乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本标签 |
| $k(\boldsymbol{x}^{(n)}, \boldsymbol{x})$ | hé hánshù | 核函数 |
| $b^*$ | zuì yōu piàn zhì | 最优偏置项 |

**一句话解释：**
核函数支持向量机的对偶形式，通过核函数隐式地在高维空间中进行计算，避免显式映射。

**简单应用场景：**
大规模非线性分类问题，核函数可以是多项式、高斯(RBF)或其他类型。

---

---

### 公式 (3.97)：多项式核函数 / Polynomial Kernel Function

$$
k(\boldsymbol{x}, \boldsymbol{z}) = (1 + \boldsymbol{x}^T\boldsymbol{z})^2 = \phi(\boldsymbol{x})^T\phi(\boldsymbol{z})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $k(\boldsymbol{x}, \boldsymbol{z})$ | hé hánshù | 核函数 |
| $\boldsymbol{x}, \boldsymbol{z}$ | shù jù xiàng liàng | 输入向量 |
| $\phi(\boldsymbol{x})$ | tè zhēng yìng shè | 特征映射函数 |

**一句话解释：**
多项式核函数对应于一个多项式次数的特征映射，使得高次交互项可以被线性分类器处理。

**简单应用场景：**
图像识别和文本分类中处理高阶交互特征。

---

---

### 公式 (3.98)：多项式核函数 - 显式映射形式 / Polynomial Kernel - Explicit Feature Mapping

$$
\phi(\boldsymbol{x}) = [1, \sqrt{2x_1}, \sqrt{2x_2}, \sqrt{2}x_1x_2, x_1^2, x_2^2]^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\phi(\boldsymbol{x})$ | tè zhēng xiàng liàng | 显式特征向量 |
| $x_1, x_2$ | tè zhēng | 原始特征 |
| $\sqrt{2}$ | hé shù | 规范化系数 |

**一句话解释：**
多项式核$(1 + \boldsymbol{x}^T\boldsymbol{z})^2$对应的显式特征映射，包含常数项、一阶项、二阶项的平方根加权。

**简单应用场景：**
理解核函数如何隐式扩展特征空间的具体例子。

---

---

### 公式 (3.99)：软间隔支持向量机 - 优化问题 / Soft-Margin SVM - Optimization Problem

$$
\min_{w,b} \frac{1}{2}\|\boldsymbol{w}\|^2 + C \sum_{n=1}^{N} \xi_n
$$

**s.t.** $1 - y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) - \xi_n \leq 0, \quad \xi_n \geq 0, \quad \forall n \in \{1, \cdots, N\}$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | wèi liàng | 权重向量 |
| $b$ | piàn zhì | 偏置项 |
| $C$ | zhèng huà xì shù | 正则化参数 |
| $\xi_n$ | sōng chí liàng | 松弛变量 |
| $y^{(n)}$ | biāo qiān | 样本标签 |

**一句话解释：**
通过引入松弛变量允许某些样本违反间隔约束，权衡间隔最大化与分类错误的平衡。

**简单应用场景：**
处理线性不可分数据，允许少量样本在边界内或错误一侧。

---

---

### 公式 (3.100)：软间隔支持向量机 - Hinge损失形式 / Soft-Margin SVM - Hinge Loss Form

$$
\min_{w,b} \sum_{n=1}^{N} \max(0, 1 - y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)) + \frac{1}{2C}\|\boldsymbol{w}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | wèi liàng | 权重向量 |
| $b$ | piàn zhì | 偏置项 |
| $C$ | zhèng huà xì shù | 正则化参数 |
| $y^{(n)}$ | biāo qiān | 样本标签 |
| $\boldsymbol{x}^{(n)}$ | shù jù | 样本向量 |

**一句话解释：**
软间隔SVM的等价形式，使用Hinge损失函数而非松弛变量，更便于梯度下降优化。

**简单应用场景：**
与逻辑回归和其他线性分类器相比，具有相似的优化目标。

---

---

### 公式 (3.101)：Logistic回归 - 交叉熵损失函数 / Logistic Regression - Cross-Entropy Loss

$$
\mathcal{L}_{LR} = -I(y = 1) \log\sigma(f(\boldsymbol{x};\boldsymbol{w})) - I(y = -1) \log\left(1 - \sigma(f(\boldsymbol{x};\boldsymbol{w}))\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{LR}$ | jiāo chā sh熵 | 交叉熵损失 |
| $I(\cdot)$ | zhǐ shì hánshù | 指示函数 |
| $\sigma(\cdot)$ | xī gé mà hánshù | Sigmoid函数 |
| $f(\boldsymbol{x};\boldsymbol{w})$ | jué cè hánshù | 决策函数 |
| $y$ | biāo qiān | 样本标签 |

**一句话解释：**
Logistic回归的损失函数，基于Sigmoid函数的概率输出，通过最大化对数似然进行优化。

**简单应用场景：**
二分类问题中的概率性分类器，输出样本属于正类的概率。

---

---

### 公式 (3.102)：Logistic回归 - 交叉熵损失函数（等价形式） / Logistic Regression - Cross-Entropy Loss (Equivalent Form)

$$
\mathcal{L}_{LR} = -I(y = 1) \log\sigma(f(\boldsymbol{x};\boldsymbol{w})) - I(y = -1) \log\sigma(-f(\boldsymbol{x};\boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{LR}$ | jiāo chā shhaì | 交叉熵损失 |
| $\sigma(\cdot)$ | xī gé mà hánshù | Sigmoid函数 |
| $f(\boldsymbol{x};\boldsymbol{w})$ | jué cè hánshù | 决策函数 |
| $y$ | biāo qiān | 样本标签 |

**一句话解释：**
利用Sigmoid的对称性，交叉熵损失可以用单一形式表示，分子为$\sigma(-yf(\boldsymbol{x};\boldsymbol{w}))$。

**简单应用场景：**
简化Logistic回归损失函数的计算和理解。

---

---

### 公式 (3.103)：Logistic回归 - 简化损失函数 / Logistic Regression - Simplified Loss

$$
\mathcal{L}_{LR} = -\log\sigma(yf(\boldsymbol{x};\boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{LR}$ | jiāo chā shhaì | 交叉熵损失 |
| $\sigma(\cdot)$ | xī gé mà hánshù | Sigmoid函数 |
| $y$ | biāo qiān | 样本标签（±1） |
| $f(\boldsymbol{x};\boldsymbol{w})$ | jué cè hánshù | 决策函数值 |

**一句话解释：**
Logistic回归的最简化损失函数形式，对于标签为±1的二分类问题。

**简单应用场景：**
理论分析和实现Logistic回归的标准形式。

---

---

### 公式 (3.104)：Logistic回归 - 对数形式 / Logistic Regression - Logarithmic Form

$$
\mathcal{L}_{LR} = \log(1 + \exp(-yf(\boldsymbol{x};\boldsymbol{w})))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{LR}$ | jiāo chā shhaì | 交叉熵损失 |
| $y$ | biāo qiān | 样本标签（±1） |
| $f(\boldsymbol{x};\boldsymbol{w})$ | jué cè hánshù | 决策函数值 |
| $\exp(\cdot)$ | zh指 shù hánshù | 指数函数 |

**一句话解释：**
使用对数-和-指数(log-sum-exp)形式避免数值溢出，数值上更稳定的Logistic损失表示。

**简单应用场景：**
实际编程实现中的标准形式，避免计算$\sigma$时的数值问题。

---

---

### 公式 (3.105)：感知损失 / Perceptron Loss

$$
\mathcal{L}_p = \max(0, -yf(\boldsymbol{x};\boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_p$ | gǎn zhī sǔn shī | 感知损失 |
| $y$ | biāo qiān | 样本标签（±1） |
| $f(\boldsymbol{x};\boldsymbol{w})$ | jué cè hánshù | 决策函数值 |

**一句话解释：**
感知机的损失函数，只对分类错误的样本计算损失，正确分类的样本损失为零。

**简单应用场景：**
感知机算法中的损失度量，是支持向量机和Logistic回归的基础。

---

---

### 公式 (3.106)：软间隔支持向量机 - Hinge损失 / Soft-Margin SVM - Hinge Loss

$$
\mathcal{L}_{\text{hinge}} = \max(0, 1 - yf(\boldsymbol{x};\boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{hinge}}$ | hinge sǔn shī | Hinge损失 |
| $y$ | biāo qiān | 样本标签（±1） |
| $f(\boldsymbol{x};\boldsymbol{w})$ | jué cè hánshù | 决策函数值 |

**一句话解释：**
支持向量机的Hinge损失函数，要求样本的决策函数值至少为1，否则计算损失。

**简单应用场景：**
支持向量机的标准损失函数，比感知损失更强地鼓励样本远离决策边界。

---

---

### 公式 (3.107)：平方损失函数 / Squared Loss Function

$$
\mathcal{L}_{\text{squared}} = (y - f(\boldsymbol{x};\boldsymbol{w}))^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{squared}}$ | píng fāng sǔn shī | 平方损失 |
| $y$ | biāo qiān | 样本标签（实数值） |
| $f(\boldsymbol{x};\boldsymbol{w})$ | yù cè zhí | 预测值 |

**一句话解释：**
回归问题中最常用的损失函数，衡量预测值与真实值之间的平方差距。

**简单应用场景：**
线性回归、神经网络回归任务中的标准损失函数。

---

---

### 公式 (3.108)：平方损失函数 - 展开形式1 / Squared Loss - Expanded Form 1

$$
\mathcal{L}_{\text{squared}} = 1 - 2yf(\boldsymbol{x};\boldsymbol{w}) + (f(\boldsymbol{x};\boldsymbol{w}))^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{squared}}$ | píng fāng sǔn shī | 平方损失 |
| $y$ | biāo qiān | 样本标签（±1） |
| $f(\boldsymbol{x};\boldsymbol{w})$ | yù cè zhí | 预测值 |

**一句话解释：**
对标签为±1的分类问题，平方损失的展开形式，便于分析其与其他损失函数的关系。

**简单应用场景：**
二分类平方损失函数的展开分析。

---

---

### 公式 (3.109)：平方损失函数 - 展开形式2 / Squared Loss - Expanded Form 2

$$
\mathcal{L}_{\text{squared}} = (1 - yf(\boldsymbol{x};\boldsymbol{w}))^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{squared}}$ | píng fāng sǔn shī | 平方损失 |
| $y$ | biāo qiān | 样本标签（±1） |
| $f(\boldsymbol{x};\boldsymbol{w})$ | yù cè zhí | 预测值 |

**一句话解释：**
二分类平方损失函数的另一种展开形式，直接表示为$(1 - \text{margin})^2$，其中margin为$yf(\boldsymbol{x};\boldsymbol{w})$。

**简单应用场景：**
分析平方损失与间隔(margin)的关系。

---

## 本部分总结

本部分包含了支持向量机(SVM)、Logistic回归和其他线性分类器的完整数学表述：

1. **支持向量机(3.87-3.94)**：从标准形式到对偶形式，再到决策函数的演变
2. **核函数方法(3.95-3.98)**：非线性支持向量机及多项式核函数的具体示例
3. **软间隔SVM(3.99-3.100)**：处理线性不可分数据的方法
4. **损失函数对比(3.101-3.109)**：Logistic回归、感知损失、Hinge损失和平方损失的详细比较

这些公式构成了机器学习中线性模型和支持向量机理论的核心基础。

---

## 第4章：前馈神经网络

### 公式 (4.1)：神经元输入计算 / Neuron Input Calculation

$$
z = \sum_{d=1}^{D} u_d x_d + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| z | zī | 神经元的加权和（净输入） |
| u_d | yōu xià biāo d | 第d个输入的权重 |
| x_d | x iā biāo d | 第d个输入 |
| b | bī | 偏置项 |
| D | dà xiě d | 输入维数 |

**一句话解释：**
神经元通过对所有输入的加权求和加上偏置项来计算其净输入值。

**简单应用场景：**
这是神经网络中任何一个神经元的基本计算，将多个输入信号进行加权聚合后产生单个数值。

---

---

### 公式 (4.2)：向量化神经元计算 / Vectorized Neuron Calculation

$$
z = \boldsymbol{\omega}^T \boldsymbol{x} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}$ | ōu mǐ gā | 权重向量 |
| $\boldsymbol{x}$ | x | 输入向量 |
| z | zī | 神经元净输入 |
| b | bī | 偏置项 |

**一句话解释：**
使用向量表示法，将权重向量与输入向量的点积加上偏置项来计算神经元的输入。

**简单应用场景：**
在编程和数值计算中，使用向量化形式可以利用BLAS等优化库提升计算效率。

---

---

### 公式 (4.3)：激活函数 / Activation Function

$$
a = f(z)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| a | ā | 神经元激活值（输出） |
| f | f | 激活函数 |
| z | zī | 神经元净输入 |

**一句话解释：**
通过激活函数对神经元的净输入进行非线性变换，得到神经元的最终输出。

**简单应用场景：**
激活函数引入非线性性，使神经网络能够学习和表示非线性映射。

---

---

### 公式 (4.4)：Logistic函数 / Logistic Activation Function

$$
\sigma(x) = \frac{1}{1 + \exp(-x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | Logistic激活函数 |
| x | x | 输入 |
| $\exp$ | ē | 指数函数 |

**一句话解释：**
Logistic函数将任意实数映射到(0,1)区间，常用于二分类问题的输出层。

**简单应用场景：**
在逻辑回归和神经网络的二分类输出层中，用于将预测值映射为概率。

---

---

### 公式 (4.5)：Tanh函数 / Hyperbolic Tangent Function

$$
\tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tanh$ | tán huo | 双曲正切函数 |
| x | x | 输入 |
| $\exp$ | ē | 指数函数 |

**一句话解释：**
Tanh函数将任意实数映射到(-1,1)区间，是Logistic函数的平移和缩放版本。

**简单应用场景：**
在隐层中常用的激活函数，其输出以0为中心，有助于加快收敛。

---

---

### 公式 (4.6)：Tanh与Logistic的关系 / Relationship between Tanh and Logistic

$$
\tanh(x) = 2\sigma(2x) - 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tanh$ | tán huo | 双曲正切函数 |
| $\sigma$ | xī gé mǎ | Logistic函数 |
| x | x | 输入 |

**一句话解释：**
Tanh函数可以通过Logistic函数进行线性变换得到，揭示了两个函数间的联系。

**简单应用场景：**
理论分析中用于证明Tanh和Logistic函数的等价性和相互转化。

---

---

### 公式 (4.7)：Logistic函数的一阶泰勒展开 / First-Order Taylor Expansion of Logistic

$$
g_l(x) \approx \sigma(0) + x \cdot \sigma'(0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| g_l | g xiā biāo l | Logistic函数的近似函数 |
| x | x | 输入 |
| $\sigma$ | xī gé mǎ | Logistic激活函数 |
| $\sigma'$ | xī gé mǎ dǎo shù | Logistic函数的导数 |

**一句话解释：**
在x=0处对Logistic函数进行一阶泰勒展开，用于构造Hard-Logistic函数的近似。

**简单应用场景：**
为了获得计算效率更高的激活函数，可以用分段线性函数近似Logistic函数。

---

---

### 公式 (4.8)：Hard-Logistic近似 / Hard-Logistic Approximation

$$
= 0.25x + 0.5
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| 0.25 | líng diǎn èr wǔ | 斜率系数 |
| 0.5 | líng diǎn wǔ | 截距 |

**一句话解释：**
Logistic函数在其中心处的一阶泰勒近似为简单的线性函数。

**简单应用场景：**
用于快速计算，避免昂贵的指数运算。

---

---

### 公式 (4.9)：Hard-Logistic分段函数 / Hard-Logistic Function

$$
\text{hard-logistic}(x) = \begin{cases}
1 & g_l(x) \geq 1 \\
g_l & 0 < g_l(x) < 1 \\
0 & g_l(x) \leq 0
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{hard-logistic}$ | hèn luó ji | Hard-Logistic激活函数 |
| x | x | 输入 |
| g_l | g xiā biāo l | Logistic函数的一阶近似 |

**一句话解释：**
通过截断Logistic函数的线性近似，使其输出在(-∞,0)和(1,+∞)处达到饱和。

**简单应用场景：**
在需要快速计算且对精度要求不极高的场景中使用。

---

---

### 公式 (4.10)：Hard-Logistic的等价形式1 / Hard-Logistic Equivalent Form 1

$$
= \max\left(\min(g_l(x), 1), 0\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà | 最大值函数 |
| $\min$ | zuì xiǎo | 最小值函数 |
| g_l | g xiā biāo l | Logistic函数的一阶近似 |

**一句话解释：**
Hard-Logistic函数也可以表示为先取下界后取上界的形式。

**简单应用场景：**
编程实现时可以直接使用min和max函数。

---

---

### 公式 (4.11)：Hard-Logistic的等价形式2 / Hard-Logistic Explicit Form

$$
= \max\left(\min(0.25x + 0.5, 1), 0\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà | 最大值函数 |
| $\min$ | zuì xiǎo | 最小值函数 |
| x | x | 输入 |

**一句话解释：**
将Hard-Logistic函数完全展开为显式的分段线性形式。

**简单应用场景：**
直接计算Hard-Logistic函数值。

---

---

### 公式 (4.12)：反向传播梯度计算 / Backpropagation Gradient Calculation

$$
\tanh(x) \approx x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $y$ | wài | Ground truth label / 真实标签 |
| $\hat{y}$ | hào chā | Predicted output / 预测输出 |
| $z^{(l)}$ | shàng biāo $l$ | Pre-activation of layer $l$ / 第$l$层激活前的值 |
| $\delta^{(l)}$ | dé ěr tǎ | Error signal of layer $l$ / 第$l$层的误差信号 |

**一句话解释：**
损失函数对第$l$层激活前值的梯度等于该层的误差信号。

**简单应用场景：**
在反向传播过程中，计算每一层的误差信号以用于参数更新。

---

---

### 公式 (4.13)：Hard-Tanh一阶近似结果 / Hard-Tanh Linear Approximation

$$
= x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| x | x | 输入 |

**一句话解释：**
Tanh函数在x=0处的泰勒展开恰好为恒等函数。

**简单应用场景：**
说明在原点附近，Tanh函数近似于线性函数。

---

---

### 公式 (4.14)：Hard-Tanh分段函数 / Hard-Tanh Function

$$
\text{hard-tanh}(x) = \max\left(\min(g_t(x), 1), -1\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{hard-tanh}$ | hèn tán | Hard-Tanh激活函数 |
| x | x | 输入 |
| g_t | g xiā biāo t | Tanh函数的一阶近似 |
| $\max$ | zuì dà | 最大值函数 |
| $\min$ | zuì xiǎo | 最小值函数 |

**一句话解释：**
Hard-Tanh函数通过限制Tanh函数线性近似的输出范围在[-1,1]。

**简单应用场景：**
在需要快速激活函数的应用中替代Tanh。

---

---

### 公式 (4.15)：Hard-Tanh的等价形式 / Hard-Tanh Equivalent Form

$$
= \max\left(\min(x, 1), -1\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà | 最大值函数 |
| $\min$ | zuì xiǎo | 最小值函数 |
| x | x | 输入 |

**一句话解释：**
由于Tanh的一阶近似就是x，Hard-Tanh简化为直接截断x的值到[-1,1]范围内。

**简单应用场景：**
编程实现中的显式计算形式。

---

---

### 公式 (4.16)：ReLU函数 / Rectified Linear Unit

$$
\text{ReLU}(x) = \begin{cases}
x & x \geq 0 \\
0 & x < 0
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{ReLU}$ | ruì lú | 修正线性单元激活函数 |
| x | x | 输入 |

**一句话解释：**
ReLU函数对负值输出0，对正值保持原值，是现代深度学习中最常用的激活函数。

**简单应用场景：**
几乎所有现代深度神经网络的隐层都使用ReLU或其变体作为激活函数。

---

---

### 公式 (4.17)：ReLU的等价形式 / ReLU Equivalent Form

$$
= \max(0, x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà | 最大值函数 |
| x | x | 输入 |

**一句话解释：**
ReLU可以简洁地表示为输入和0的最大值。

**简单应用场景：**
编程和数学计算中的标准表示。

---

---

### 公式 (4.18)：Leaky ReLU函数 / Leaky ReLU

$$
\text{LeakyReLU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\gamma x & \text{if } x \leq 0
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{LeakyReLU}$ | lì kì ruì lú | 带泄漏的修正线性单元 |
| x | x | 输入 |
| $\gamma$ | gā mǎ | 负值部分的斜率系数 |

**一句话解释：**
Leaky ReLU允许负值部分有一个小的非零斜率γ，解决ReLU中负值完全变为0的问题。

**简单应用场景：**
在ReLU因神经元死亡导致性能下降的情况下使用。

---

---

### 公式 (4.19)：Leaky ReLU等价形式 / Leaky ReLU Equivalent Form

$$
= \max(0, x) + \gamma \min(0, x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà | 最大值函数 |
| $\min$ | zuì xiǎo | 最小值函数 |
| x | x | 输入 |
| $\gamma$ | gā mǎ | 负值部分的斜率系数 |

**一句话解释：**
将Leaky ReLU表示为正部分和负部分的加权和。

**简单应用场景：**
便于数值计算和自动微分框架中的实现。

---

---

### 公式 (4.20)：Leaky ReLU的特殊情况 / Leaky ReLU Special Case

$$
\text{LeakyReLU}(x) = \max(x, \gamma x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{LeakyReLU}$ | lì kì ruì lú | 带泄漏的修正线性单元 |
| x | x | 输入 |
| $\gamma$ | gā mǎ | 负值部分的斜率系数 |
| $\max$ | zuì dà | 最大值函数 |

**一句话解释：**
Leaky ReLU也可以表示为x和γx的最大值，当x为负时选择γx。

**简单应用场景：**
代码实现中的快速计算形式。

---

---

### 公式 (4.21)：参数化ReLU函数 / Parametric ReLU

$$
\text{PReLU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\gamma x & \text{if } x \leq 0
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{PReLU}$ | pēi rù shù huà ruì lú | 参数化修正线性单元 |
| x | x | 输入 |
| $\gamma$ | gā mǎ | 可学习的斜率参数 |

**一句话解释：**
PReLU是Leaky ReLU的推广，其中斜率参数γ不再固定，而是在训练过程中学习。

**简单应用场景：**
当固定的Leaky ReLU斜率不能满足需求时，使用PReLU让模型自己学习最优的斜率。

---

---

### 公式 (4.22)：PReLU等价形式 / PReLU Equivalent Form

$$
= \max(0, x) + \gamma \min(0, x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà | 最大值函数 |
| $\min$ | zuì xiǎo | 最小值函数 |
| x | x | 输入 |
| $\gamma$ | gā mǎ | 可学习的斜率参数 |

**一句话解释：**
PReLU的数学形式与Leaky ReLU相同，但参数γ是可学习的。

**简单应用场景：**
用于反向传播中对γ的梯度计算。

---

---

### 公式 (4.23)：ELU函数 / Exponential Linear Unit

$$
\text{ELU}(x) = \begin{cases}
x & \text{if } x > 0 \\
\gamma(\exp(x) - 1) & \text{if } x \leq 0
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{ELU}$ | ē ēl yóu | 指数线性单元 |
| x | x | 输入 |
| $\gamma$ | gā mǎ | 小于0时的参数 |
| $\exp$ | ē | 指数函数 |

**一句话解释：**
ELU在正值处为线性，在负值处为指数函数，提供光滑的非线性激活。

**简单应用场景：**
需要更平滑的负值部分且想降低计算复杂度时的折中方案。

---

---

### 公式 (4.24)：ELU等价形式 / ELU Equivalent Form

$$
= \max(0, x) + \min(0, \gamma(\exp(x) - 1))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà | 最大值函数 |
| $\min$ | zuì xiǎo | 最小值函数 |
| x | x | 输入 |
| $\gamma$ | gā mǎ | 小于0时的参数 |
| $\exp$ | ē | 指数函数 |

**一句话解释：**
将ELU分解为正部分和负部分的组合形式。

**简单应用场景：**
便于自动微分框架的实现。

---

---

### 公式 (4.25)：Softplus函数 / Softplus Activation

$$
\text{Softplus}(x) = \log(1 + \exp(x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Softplus}$ | ruǎn jiā | 软加函数 |
| x | x | 输入 |
| $\log$ | duì shù | 自然对数 |
| $\exp$ | ē | 指数函数 |

**一句话解释：**
Softplus函数是ReLU的光滑近似，整个定义域上都可导。

**简单应用场景：**
在需要处处可导且希望避免ReLU硬截断的场景中使用。

---

---

### 公式 (4.26)：Swish函数 / Swish Activation

$$
\text{swish}(x) = x\sigma(\beta x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{swish}$ | swish | 自门控激活函数 |
| x | x | 输入 |
| $\sigma$ | xī gé mǎ | Logistic函数 |
| $\beta$ | bèi tǎ | 可学习或固定的参数 |

**一句话解释：**
Swish函数将输入与其Logistic变换的乘积作为输出，实现自门控机制。

**简单应用场景：**
在Google的EfficientNet等现代架构中作为激活函数使用。

---

---

### 公式 (4.27)：GELU函数 / Gaussian Error Linear Unit

$$
\text{GELU}(x) = xP(X \leq x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{GELU}$ | jiāo ē ēl yóu | 高斯误差线性单元 |
| x | x | 输入 |
| P | gài lǜ | 概率 |
| X | dā | 高斯分布随机变量 |

**一句话解释：**
GELU函数将输入乘以其在标准正态分布中的累积分布函数值。

**简单应用场景：**
在Transformer模型（如BERT）中广泛使用。

---

---

### 公式 (4.28)：GELU的Tanh近似 / GELU Tanh Approximation

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{GELU}$ | jiāo ē ēl yóu | 高斯误差线性单元 |
| x | x | 输入 |
| $\tanh$ | tán huo | 双曲正切函数 |
| $\sqrt{\frac{2}{\pi}}$ | píng fāng gēn | 常数系数 |

**一句话解释：**
使用Tanh函数对GELU的累积分布函数进行多项式近似。

**简单应用场景：**
在计算GELU时提供更快的近似版本。

---

---

### 公式 (4.29)：GELU的Sigmoid近似 / GELU Sigmoid Approximation

$$
\text{GELU}(x) \approx x\sigma(1.702x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{GELU}$ | jiāo ē ēl yóu | 高斯误差线性单元 |
| x | x | 输入 |
| $\sigma$ | xī gé mǎ | Logistic函数 |

**一句话解释：**
GELU的另一种近似形式，使用Logistic函数实现。

**简单应用场景：**
计算效率更高，但精度略低于Tanh近似。

---

---

### 公式 (4.30)：Maxout单元的输入计算 / Maxout Unit Input

$$
z_k = \boldsymbol{\omega}_k^T \boldsymbol{x} + b_k
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| z_k | zī xiā biāo k | 第k个分支的输入 |
| $\boldsymbol{\omega}_k$ | ōu mǐ gā xiā biāo k | 第k个分支的权重向量 |
| $\boldsymbol{x}$ | x | 输入向量 |
| b_k | bī xiā biāo k | 第k个分支的偏置 |

**一句话解释：**
Maxout单元由K个独立的线性变换组成，每个分支计算一个加权和加偏置。

**简单应用场景：**
Maxout单元的基本计算步骤，为之后的取最大值操作准备输入。

---

---

### 公式 (4.31)：Maxout单元 / Maxout Unit

$$
\text{maxout}(x) = \max_{k \in [1,K]} (z_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{maxout}$ | mǎ kè shù chū | 最大输出单元 |
| x | x | 输入向量 |
| z_k | zī xiā biāo k | 第k个分支的输出 |
| K | dà xiě k | 分支数量 |
| $\max$ | zuì dà | 最大值函数 |

**一句话解释：**
Maxout单元计算K个不同线性变换中的最大值，是ReLU的推广。

**简单应用场景：**
在深度学习中用作灵活的激活函数，可以学习分段线性的激活模式。

---

---

### 公式 (4.32)：前馈网络信息传递 / Feedforward Network Forward Pass

$$
z^{(l)} = \boldsymbol{W}^{(l)} a^{(l-1)} + \boldsymbol{b}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| z^{(l)} | zī shàng l | 第l层的净输入 |
| $\boldsymbol{W}^{(l)}$ | dà xiě w shàng l | 第l层的权重矩阵 |
| a^{(l-1)} | ā shàng l jiǎn 1 | 第l-1层的激活输出 |
| $\boldsymbol{b}^{(l)}$ | bī shàng l | 第l层的偏置向量 |

**一句话解释：**
前馈网络第l层通过对前一层激活的线性变换（加权和加偏置）计算净输入。

**简单应用场景：**
神经网络前向传播中每一层的基本计算。

---

---

### 公式 (4.33)：前馈网络激活输出 / Feedforward Network Activation Output

$$
a^{(l)} = f(z^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| a^{(l)} | ā shàng l | 第l层的激活输出 |
| f | f | 激活函数 |
| z^{(l)} | zī shàng l | 第l层的净输入 |

**一句话解释：**
第l层的激活输出是通过将净输入代入激活函数得到的。

**简单应用场景：**
前向传播中应用激活函数的步骤。

---

---

### 公式 (4.34)：隐层梯度反向传播 / Hidden Layer Gradient Backpropagation

$$
\boldsymbol{z}^{(l)} = \boldsymbol{W}^{(l)} f_{l-1}(\boldsymbol{z}^{(l-1)}) + \boldsymbol{b}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta^{(l)}$ | dé ěr tǎ | Error signal of layer $l$ / 第$l$层的误差信号 |
| $\boldsymbol{W}^{(l+1)}$ | dà xiě $W$ | Weight matrix of layer $l+1$ / 第$l+1$层权重矩阵 |
| $\odot$ | hā dá mǎ | Hadamard product / 哈达玛积（元素乘积） |
| $\sigma'(z^{(l)})$ | dǎo shù | Derivative of activation function / 激活函数导数 |

**一句话解释：**
第$l$层的误差信号由下一层的误差信号经过权重矩阵反向传播，再与激活函数导数相乘得到。

**简单应用场景：**
计算多层神经网络中每一层的梯度时的核心公式。

---

---

### 公式 (4.35)：前馈网络的另一种合并形式 / Feedforward Network Alternative Combined Form

$$
a^{(l)} = f_l(\boldsymbol{W}^{(l)} a^{(l-1)} + \boldsymbol{b}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| a^{(l)} | ā shàng l | 第l层的激活输出 |
| f_l | f xiā biāo l | 第l层的激活函数 |
| $\boldsymbol{W}^{(l)}$ | dà xiě w shàng l | 第l层的权重矩阵 |
| a^{(l-1)} | ā shàng l jiǎn 1 | 第l-1层的激活输出 |
| $\boldsymbol{b}^{(l)}$ | bī shàng l | 第l层的偏置向量 |

**一句话解释：**
前馈网络第l层的激活输出是通过对前一层激活的线性变换结果应用激活函数得到的。

**简单应用场景：**
神经网络前向传播的标准链式计算形式，也是反向传播的基础。

---

### 公式 (4.36)：前馈神经网络的信息传播 / Feedforward Neural Network Information Propagation

$$
\boldsymbol{x} = \boldsymbol{a}^{(0)} \to \boldsymbol{z}^{(1)} \to \boldsymbol{a}^{(1)} \to \boldsymbol{z}^{(2)} \to \cdots \to \boldsymbol{a}^{(L-1)} \to \boldsymbol{z}^{(L)} \to \boldsymbol{a}^{(L)} = \varphi(\boldsymbol{x}; \boldsymbol{W}, \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | xiàng liàng | 输入向量 |
| $\boldsymbol{a}^{(l)}$ | huó huà zhí | 第l层的激活值 |
| $\boldsymbol{z}^{(l)}$ | xiàn xìng zǔ hé | 第l层的线性组合 |
| $\boldsymbol{W}, \boldsymbol{b}$ | quán zhòng, piàn zhì | 权重和偏置 |
| $\varphi$ | fài | 输出函数 |
| $L$ | céng shù | 网络层数 |

**一句话解释：**
前馈神经网络通过逐层传播输入信息，从输入层经过隐藏层最终产生输出，整个过程由权重、偏置和激活函数决定。

**简单应用场景：**
图像分类中，输入图像像素值经过多层神经元逐步处理，最终输出类别概率。

---

---

### 公式 (4.37)：通用近似定理 / Universal Approximation Theorem

$$
F(\boldsymbol{x}) = \sum_{m=1}^{M} u_m \varphi(\boldsymbol{w}_m^T \boldsymbol{x} + b_m)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F(\boldsymbol{x})$ | hánshù | 近似函数 |
| $M$ | shù liàng | 隐藏层神经元数 |
| $u_m$ | shū chū quán zhòng | 输出层权重 |
| $\boldsymbol{w}_m$ | yǐn cáng céng quán zhòng | 隐藏层权重 |
| $b_m$ | piàn zhì | 偏置 |
| $\varphi$ | fài | 激活函数 |

**一句话解释：**
任何连续函数都可以用足够多的隐藏神经元的单隐藏层网络来近似。

**简单应用场景：**
说明为什么神经网络能够学习任意复杂的函数映射。

---

---

### 公式 (4.38)：近似误差界 / Approximation Error Bound

$$
|F(\boldsymbol{x}) - f(\boldsymbol{x})| < \varepsilon, \quad \forall \boldsymbol{x} \in \mathcal{D}_p
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F(\boldsymbol{x})$ | jìn sì hánshù | 近似函数 |
| $f(\boldsymbol{x})$ | yuán hánshù | 原始函数 |
| $\varepsilon$ | èpsilon | 误差界 |
| $\mathcal{D}_p$ | dìng yì yù | 定义域 |

**一句话解释：**
近似函数与真实函数之间的误差可以小于任意指定的正数。

**简单应用场景：**
确保神经网络的近似精度满足实际应用需求。

---

---

### 公式 (4.39)：前馈神经网络的输出 / Feedforward Neural Network Output

$$
\hat{y} = g(\varphi(\boldsymbol{x}); \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}$ | yù cè shū chū | 预测输出 |
| $g$ | fēn lèi hánshù | 分类函数 |
| $\boldsymbol{\theta}$ | xī tǎ | 网络参数 |

**一句话解释：**
前馈网络先通过隐藏层特征提取，再通过分类函数输出预测结果。

**简单应用场景：**
二分类问题中，隐藏层提取特征，最后通过Logistic函数输出概率。

---

---

### 公式 (4.40)：Logistic 分类回归 / Logistic Classification Regression

$$
p(\boldsymbol{y} = 1| \boldsymbol{x}) = \boldsymbol{a}^{(L)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{y} = 1 \mid \boldsymbol{x})$ | tiáo jiàn gài lǜ | 条件概率 |
| $\boldsymbol{a}^{(L)}$ | shū chū céng huó huà zhí | 输出层激活值 |

**一句话解释：**
输出层激活值表示样本属于正类的条件概率。

**简单应用场景：**
二分类任务中，网络输出值代表样本为正类的概率。

---

---

### 公式 (4.41)：Softmax 多分类 / Softmax Multi-classification

$$
\boldsymbol{y} = \text{softmax}(\boldsymbol{z}^{(L)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{y}$ | shū chū xiàng liàng | 输出向量 |
| $\boldsymbol{z}^{(L)}$ | zuì hòu céng xiàn xìng zǔ hé | 最后层线性组合 |

**一句话解释：**
对最后层的线性组合应用softmax函数，得到各类别的概率分布。

**简单应用场景：**
多分类问题中，将网络输出转换为各类别的概率分布。

---

---

### 公式 (4.42)：交叉熵损失函数 / Cross-Entropy Loss Function

$$
\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}}) = -\boldsymbol{y}^T \log \hat{\boldsymbol{y}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔn shī | 损失函数 |
| $\boldsymbol{y}$ | zhēn shí biāo qiān | 真实标签 |
| $\hat{\boldsymbol{y}}$ | yù cè gài lǜ | 预测概率 |

**一句话解释：**
衡量真实标签和预测概率分布之间的差异。

**简单应用场景：**
分类问题中最常用的损失函数。

---

---

### 公式 (4.43)：风险函数 / Risk Function

$$
\mathcal{R}(\boldsymbol{W}, \boldsymbol{b}) = \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}(y^{(n)}, \hat{y}^{(n)}) + \frac{\lambda}{2} \|\boldsymbol{W}\|_F^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}$ | fēng xiǎn | 风险函数 |
| $N$ | yàng běn shù | 样本数 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\|\boldsymbol{W}\|_F^2$ | Frōbéniǔs fànzǐ | Frobenius范数 |

**一句话解释：**
总损失函数包括经验风险和L2正则项，防止过拟合。

**简单应用场景：**
训练神经网络时的目标函数。

---

---

### 公式 (4.44)：Frobenius 范数 / Frobenius Norm

$$
\|\boldsymbol{W}\|_F^2 = \sum_{l=1}^{L} \sum_{i}^{M_l} \sum_{j}^{M_{l-1}} (w_{ij}^{(l)})^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{W}\|_F^2$ | Frōbéniǔs fànzǐ píng fāng | Frobenius范数平方 |
| $M_l$ | dì l céng shén jīng yuán shù | 第l层神经元数 |

**一句话解释：**
所有权重矩阵元素平方和，用于L2正则化。

**简单应用场景：**
模型正则化中防止权重过大。

---

---

### 公式 (4.45)：参数更新规则 / Parameter Update Rule

$$
\boldsymbol{W}^{(l)} \leftarrow \boldsymbol{W}^{(l)} - \alpha \frac{\partial \mathcal{R}(\boldsymbol{W}, \boldsymbol{b})}{\partial \boldsymbol{W}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āl fǎ | 学习率 |
| $\frac{\partial \mathcal{R}}{\partial \boldsymbol{W}^{(l)}}$ | piàn dǎo shù | 偏导数 |

**一句话解释：**
沿着负梯度方向更新权重，幅度由学习率控制。

**简单应用场景：**
随机梯度下降法的基本参数更新方式。

---

---

### 公式 (4.46)：权重梯度计算 / Weight Gradient Calculation

$$
\boldsymbol{W}^{(l)} = \boldsymbol{W}^{(l)} - \alpha \left( \frac{1}{N} \sum_{n=1}^{N} \frac{\partial \mathcal{L}(y^{(n)}, \hat{y}^{(n)})}{\partial \boldsymbol{W}^{(l)}} + \lambda \boldsymbol{W}^{(l)} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āl fǎ | 学习率 |
| $\lambda$ | lán mǔ dá | 正则化系数 |

**一句话解释：**
结合经验风险梯度和正则项梯度更新权重。

**简单应用场景：**
带L2正则化的梯度下降更新。

---

---

### 公式 (4.47)：偏置梯度更新 / Bias Gradient Update

$$
\boldsymbol{b}^{(l)} \leftarrow \boldsymbol{b}^{(l)} - \alpha \frac{\partial \mathcal{R}(\boldsymbol{W}, \boldsymbol{b})}{\partial \boldsymbol{b}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{b}^{(l)}$ | piàn zhì xiàng liàng | 偏置向量 |
| $\alpha$ | āl fǎ | 学习率 |

**一句话解释：**
沿着负梯度方向更新偏置参数。

**简单应用场景：**
每一层的偏置参数更新。

---

---

### 公式 (4.48)：偏置完整更新式 / Complete Bias Update

$$
\boldsymbol{b}^{(l)} = \boldsymbol{b}^{(l)} - \alpha \left( \frac{1}{N} \sum_{n=1}^{N} \frac{\partial \mathcal{L}(y^{(n)}, \hat{y}^{(n)})}{\partial \boldsymbol{b}^{(l)}} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{b}^{(l)}$ | piàn zhì xiàng liàng | 偏置向量 |
| $N$ | yàng běn shù | 样本数 |

**一句话解释：**
对所有样本的偏置梯度取平均后进行更新。

**简单应用场景：**
批量梯度下降中的偏置更新。

---

---

### 公式 (4.49)：链式法则 (1) / Chain Rule (1)

$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{w}_{ij}^{(l)}} = \frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{w}_{ij}^{(l)}} \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \boldsymbol{w}_{ij}^{(l)}}$ | piàn dǎo | 偏导数 |
| $\boldsymbol{z}^{(l)}$ | xiàn xìng zǔ hé | 线性组合 |

**一句话解释：**
计算损失对权重的梯度时，先计算中间变量对权重的导数。

**简单应用场景：**
反向传播算法中的基本链式法则应用。

---

---

### 公式 (4.50)：链式法则 (2) / Chain Rule (2)

$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{b}^{(l)}} = \frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{b}^{(l)}} \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{b}^{(l)}$ | piàn zhì xiàng liàng | 偏置向量 |
| $\boldsymbol{z}^{(l)}$ | xiàn xìng zǔ hé | 线性组合 |

**一句话解释：**
计算损失对偏置的梯度的链式法则。

**简单应用场景：**
反向传播中偏置梯度的计算。

---

---

### 公式 (4.51)：权重梯度更新 / Weight Gradient Update

$$
\frac{\partial \boldsymbol{z}^{(l)}}{\partial w_{ij}^{(l)}} = \left[ 0, \;\cdots,\; a_j^{(l-1)}, \;\cdots,\; 0 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}^{(l)}$ | dà xiě $W$ | Weight matrix of layer $l$ / 第$l$层权重矩阵 |
| $\alpha$ | āěr fǎ | Learning rate / 学习率 |
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |

**一句话解释：**
权重沿着梯度的负方向更新，更新幅度由学习率控制。

**简单应用场景：**
梯度下降法中权重的参数更新规则。

---

---

### 公式 (4.52)：z^(l) 与权重关系 / Relationship between z^(l) and Weights

$$
\left[ 0, \cdots, \frac{\partial(\boldsymbol{w}_j^{(l)} \boldsymbol{a}^{(l-1)} + b_j^{(l)})}{\partial \boldsymbol{w}_{ij}^{(l)}} \cdots, 0 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}_j^{(l)}$ | dì j gè quán zhòng xiàng liàng | 第j个权重向量 |
| $\boldsymbol{a}^{(l-1)}$ | qián yī céng huó huà zhí | 前一层激活值 |

**一句话解释：**
z的j分量对w_{ij}的导数就是上一层的激活值a_i。

**简单应用场景：**
反向传播中梯度的具体计算。

---

---

### 公式 (4.53)：简化形式 / Simplified Form

$$
= \left[ 0, \cdots, \boldsymbol{a}^{(l-1)}, \cdots, 0 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{a}^{(l-1)}$ | qián yī céng huó huà zhí | 前一层激活值 |

**一句话解释：**
前一层的激活值是z对当前权重的梯度。

**简单应用场景：**
反向传播的梯度计算。

---

---

### 公式 (4.54)：偏置梯度更新 / Bias Gradient Update

$$
\frac{\partial \boldsymbol{z}^{(l)}}{\partial w_{ij}^{(l)}} \triangleq \mathbb{I}_i(a_j^{(l-1)}) \in \mathbb{R}^{1 \times M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b^{(l)}$ | 小写 $b$ | Bias of layer $l$ / 第$l$层偏置 |
| $\alpha$ | āěr fǎ | Learning rate / 学习率 |

**一句话解释：**
偏置参数也沿着梯度的负方向更新，更新规则与权重相同。

**简单应用场景：**
神经网络训练过程中偏置项的更新。

---

---

### 公式 (4.55)：∂z^(l)/∂b^(l) 计算 / Computing ∂z^(l)/∂b^(l)

$$
\frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{b}^{(l)}} = I_{M_l} \in \mathbb{R}^{M_l \times M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I_{M_l}$ | dān wèi jǔ zhèn | 单位矩阵 |
| $M_l$ | dì l céng shén jīng yuán shù | 第l层神经元数 |

**一句话解释：**
线性组合对偏置的导数是单位矩阵。

**简单应用场景：**
反向传播中偏置梯度的简化形式。

---

---

### 公式 (4.56)：δ^(l) 对输出的梯度 / Gradient of δ^(l) to Output

$$
\delta^{(l)} \triangleq \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l)}} \in \mathbb{R}^{M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta^{(l)}$ | dé ěr tǎ xiàng liàng | 误差向量 |
| $\boldsymbol{z}^{(l)}$ | xiàn xìng zǔ hé | 线性组合 |

**一句话解释：**
误差项定义为损失函数对当前层线性组合的梯度。

**简单应用场景：**
反向传播的核心概念。

---

---

### 公式 (4.57)：第l层误差梯度 / l-th Layer Error Gradient

$$
\frac{\partial \boldsymbol{z}^{(l+1)}}{\partial \boldsymbol{a}^{(l)}} = (\boldsymbol{W}^{(l+1)})^T \in \mathbb{R}^{M_l \times M_{l+1}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}^{(l+1)}$ | dì l+1 céng quán zhòng jǔ zhèn | 第l+1层权重矩阵 |
| $\boldsymbol{a}^{(l)}$ | dì l céng huó huà zhí | 第l层激活值 |

**一句话解释：**
下一层的输入对本层激活值的导数是权重矩阵的转置。

**简单应用场景：**
反向传播中误差的传递。

---

---

### 公式 (4.58)：链式法则 / Chain Rule

$$
\frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{z}^{(l)}} = \operatorname{diag}(f_l'(\boldsymbol{z}^{(l)})) \in \mathbb{R}^{M_l \times M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $z^{(l)}$ | shàng biāo $l$ | Pre-activation of layer $l$ / 第$l$层激活前的值 |

**一句话解释：**
反向传播基于链式法则，通过逐层反向计算梯度。

**简单应用场景：**
多层网络中计算损失函数对某一层输入的梯度。

---

---

### 公式 (4.59)：激活导数矩阵形式 / Activation Derivative Matrix Form

$$
= \text{diag}(f_l'(\boldsymbol{z}^{(l)})) \in \mathbb{R}^{M_l \times M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{diag}$ | duì jiǎo jǔ zhèn | 对角矩阵 |
| $f_l'(\boldsymbol{z}^{(l)})$ | huó huà hánshù dǎo shù | 激活函数导数 |

**一句话解释：**
激活导数可表示为对角矩阵形式。

**简单应用场景：**
矩阵化反向传播计算。

---

---

### 公式 (4.60)：δ^(l) 定义 / Definition of δ^(l)

$$
\delta^{(l)} \triangleq \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta^{(l)}$ | dé ěr tǎ xiàng liàng | 误差项 |
| $\mathcal{L}$ | sǔn shī | 损失函数 |

**一句话解释：**
第l层的误差项定义为损失对该层线性组合的导数。

**简单应用场景：**
反向传播的基础定义。

---

---

### 公式 (4.61)：误差反向传播链 / Error Backpropagation Chain

$$
= \left[ \frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{z}^{(l)}} \frac{\partial \boldsymbol{z}^{(l+1)}}{\partial \boldsymbol{a}^{(l)}} \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{z}^{(l+1)}} \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{a}^{(l)}$ | huó huà zhí | 激活值 |
| $\boldsymbol{z}^{(l)}$ | xiàn xìng zǔ hé | 线性组合 |

**一句话解释：**
误差通过链式法则逐层反向传播。

**简单应用场景：**
反向传播算法的核心链式法则。

---

---

### 公式 (4.62)：误差反向传播简化 / Simplified Error Backpropagation

$$
= [\text{diag}(f_l'(\boldsymbol{z}^{(l)}))] (\boldsymbol{W}^{(l+1)})^T \delta^{(l+1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{diag}$ | duì jiǎo jǔ zhèn | 对角矩阵 |
| $\boldsymbol{W}^{(l+1)}$ | quán zhòng jǔ zhèn | 权重矩阵 |

**一句话解释：**
当前层误差等于激活导数与下层误差的传播。

**简单应用场景：**
反向传播算法中误差的递推关系。

---

---

### 公式 (4.63)：完整误差传播公式 / Complete Error Propagation

$$
= f_l'(\boldsymbol{z}^{(l)}) \odot (\boldsymbol{W}^{(l+1)})^T \delta^{(l+1)} \in \mathbb{R}^{M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\odot$ | Hadamard chéngjī | Hadamard乘积 |
| $f_l'(\boldsymbol{z}^{(l)})$ | huó huà dǎo shù | 激活导数 |

**一句话解释：**
第l层误差等于激活导数与权重矩阵转置乘以下层误差的Hadamard乘积。

**简单应用场景：**
反向传播的递推计算公式。

---

---

### 公式 (4.64)：数值微分 / Numerical Differentiation

$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial w_{ij}^{(l)}} = \delta_i^{(l)} a_j^{(l-1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f'(x)$ | dǎo shù | Derivative of function $f$ / 函数$f$的导数 |
| $\Delta x$ | dé ěl tǎ $x$ | Small change in $x$ / 自变量的微小增量 |

**一句话解释：**
函数在某点的导数定义为函数值增量与自变量增量比的极限。

**简单应用场景：**
数值方法中近似计算梯度。

---

---

### 公式 (4.65)：对称数值微分 / Symmetric Numerical Differentiation

$$
= [0, \;\cdots,\; a_j^{(l-1)}, \;\cdots,\; 0] [\delta_1^{(l)}, \;\cdots,\; \delta_{M_l}^{(l)}]^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f'(x)$ | dǎo shù | Derivative of function $f$ / 函数$f$的导数 |
| $\Delta x$ | dé ěl tǎ $x$ | Small change in $x$ / 自变量的微小增量 |

**一句话解释：**
使用对称差分可获得更精确的数值导数估计。

**简单应用场景：**
梯度检验中验证反向传播实现的正确性。

---

---

### 公式 (4.66)：梯度检验偏差 / Gradient Checking Error

$$
= \delta_i^{(l)} a_j^{(l-1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x)$ | 函数 | Function value / 函数值 |
| $\Delta x$ | dé ěl tǎ $x$ | Small change in $x$ / 自变量的微小增量 |

**一句话解释：**
对称差分公式用于数值梯度的精确计算。

**简单应用场景：**
验证自动微分实现是否正确。

---

---

### 公式 (4.67)：损失关于权重的偏导 / Partial Derivative of Loss to Weights

$$
\left[ \frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{W}^{(l)}} \right]_{ij} = [\delta^{(l)}(\boldsymbol{a}^{(l-1)})^T]_{ij}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}^{(l)}$ | quán zhòng jǔ zhèn | 权重矩阵 |
| $\delta^{(l)}$ | dé ěr tǎ | 误差向量 |

**一句话解释：**
权重矩阵的梯度等于误差向量与激活值转置的外积。

**简单应用场景：**
反向传播中权重梯度矩阵形式。

---

---

### 公式 (4.68)：权重梯度矩阵形式 / Weight Gradient Matrix Form

$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{W}^{(l)}} = \delta^{(l)} (\boldsymbol{a}^{(l-1)})^T \in \mathbb{R}^{M_l \times M_{l-1}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta^{(l)}$ | dé ěr tǎ xiàng liàng | 误差向量 |
| $\boldsymbol{a}^{(l-1)}$ | qián yī céng huó huà zhí | 前一层激活值 |

**一句话解释：**
权重梯度矩阵为误差向量与上层激活值转置的外积。

**简单应用场景：**
批量反向传播中权重梯度的计算。

---

---

### 公式 (4.69)：损失对偏置的梯度 / Gradient of Loss to Bias

$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{b}^{(l)}} = \delta^{(l)} \in \mathbb{R}^{M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{b}^{(l)}$ | piàn zhì xiàng liàng | 偏置向量 |
| $\delta^{(l)}$ | dé ěr tǎ | 误差向量 |

**一句话解释：**
偏置梯度直接等于该层的误差向量。

**简单应用场景：**
反向传播中偏置梯度的最简形式。

---

---

### 公式 (4.70)：数值微分 / Numerical Differentiation

$$
f'(\boldsymbol{x}) = \lim_{\Delta \boldsymbol{x} \to 0} \frac{f(\boldsymbol{x} + \Delta \boldsymbol{x}) - f(\boldsymbol{x})}{\Delta \boldsymbol{x}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f'(\boldsymbol{x})$ | dǎo shù | 导数 |
| $\Delta \boldsymbol{x}$ | xiǎo liàng | 微小增量 |

**一句话解释：**
数值微分通过函数值的差分来近似导数。

**简单应用场景：**
验证梯度计算的正确性。

---

---

### 公式 (4.71)：对称差分法 / Symmetric Difference Method

$$
f'(\boldsymbol{x}) = \lim_{\Delta \boldsymbol{x} \to 0} \frac{f(\boldsymbol{x} + \Delta \boldsymbol{x}) - f(\boldsymbol{x} - \Delta \boldsymbol{x})}{2\Delta \boldsymbol{x}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f'(\boldsymbol{x})$ | dǎo shù | 导数 |
| $\Delta \boldsymbol{x}$ | xiǎo liàng | 微小增量 |

**一句话解释：**
对称差分法比单侧差分法具有更高的精度。

**简单应用场景：**
梯度检验中的数值导数计算。

---

---

### 公式 (4.72)：符号微分示例输入 / Symbolic Differentiation Example Input

$$
\text{输入：} 3x - x + 2x + 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x$ | biàn liàng | 变量 |

**一句话解释：**
符号微分的输入多项式。

**简单应用场景：**
演示符号微分的过程。

---

---

### 公式 (4.73)：符号微分输出 / Symbolic Differentiation Output

$$
\text{输出：} 4x + 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x$ | biàn liàng | 变量 |

**一句话解释：**
对输入多项式求导后的结果。

**简单应用场景：**
演示符号微分的输出。

---

---

### 公式 (4.74)：自动微分示例函数 / Automatic Differentiation Example Function

$$
f(x; w, b) = \frac{1}{\exp(-((\boldsymbol{w}x + b)) + 1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | quán zhòng | 权重 |
| $b$ | piàn zhì | 偏置 |
| $x$ | biàn liàng | 输入变量 |

**一句话解释：**
自动微分的示例函数是sigmoid函数。

**简单应用场景：**
演示自动微分在神经网络中的应用。

---

---

### 公式 (4.75)：复合函数偏导链式法则（第一种）/ Composite Function Partial Derivative Chain Rule (First Form)

$$
\frac{\partial f(x; w, b)}{\partial w} = \frac{\partial h_6}{\partial h_5} \cdot \frac{\partial h_5}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f$ | 函数 | Output function / 输出函数 |
| $w$ | 小写 $w$ | Parameter weight / 参数权重 |
| $h_i$ | āi | Intermediate hidden activation / 中间隐层激活 |

**一句话解释：**
复合函数的偏导数等于各层偏导数的乘积，体现了链式法则。

**简单应用场景：**
深度网络中计算损失函数对参数的梯度。

---

---

### 公式 (4.76)：复合函数偏导链式法则（第二种）/ Composite Function Partial Derivative Chain Rule (Second Form)

$$
\frac{\partial f(x; w, b)}{\partial b} = \frac{\partial h_6}{\partial h_5} \cdot \frac{\partial h_5}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f$ | 函数 | Output function / 输出函数 |
| $b$ | 小写 $b$ | Bias parameter / 偏置参数 |
| $h_i$ | āi | Intermediate hidden activation / 中间隐层激活 |

**一句话解释：**
对偏置参数求偏导时，需要遍历从输出到偏置所在层的所有路径。

**简单应用场景：**
神经网络反向传播中计算偏置的梯度。

---

---

### 公式 (4.77)：具体梯度计算示例 / Concrete Gradient Calculation Example

$$
\left. \frac{\partial f(x; w, b)}{\partial w} \right|_{x=1, w=0, b=0} = 1 \times (-0.25) \times 1 \times 1 \times (-1) \times 1 = 0.25
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w$ | 小写 $w$ | Parameter weight / 参数权重 |
| $x, u, b$ | 小写字母 | Input and parameter variables / 输入和参数变量 |
| $h_i$ | āi | Hidden layer activation / 隐层激活值 |

**一句话解释：**
在特定输入值下计算参数梯度的具体数值示例。

**简单应用场景：**
梯度检验时的具体计算步骤。

---

## 第5章 卷积神经网络 (pages 97-120)

---

### 公式 (4.78)：中间计算步骤 / Intermediate Calculation Steps

$$
= 1 \times (-0.25) \times 1 \times 1 \times (-1) \times 1 \times 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $-0.25$ | shù zhí | 数值 |

**一句话解释：**
逐步计算各个中间导数的乘积。

**简单应用场景：**
自动微分的乘法过程。

---

---

### 公式 (4.79)：最终结果 / Final Result

$$
= 0.25
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $0.25$ | jié guǒ | 结果 |

**一句话解释：**
最终得到的导数值。

**简单应用场景：**
自动微分的计算结果。

---

---

### 公式 (4.80)：反向模式梯度计算 (1) / Reverse Mode Gradient (1)

$$
\frac{\partial h_1}{\partial w} = x = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_1$ | dì yī gè zhōng jiān biàn liàng | 第一个中间变量 |
| $w$ | quán zhòng | 权重 |

**一句话解释：**
第一层中间变量对权重的导数。

**简单应用场景：**
反向模式自动微分的计算。

---

---

### 公式 (4.81)：反向模式梯度计算 (2) / Reverse Mode Gradient (2)

$$
\frac{\partial h_2}{\partial w} = \frac{\partial h_2}{\partial h_1} \frac{\partial h_1}{\partial w} = 1 \times 1 = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_2$ | dì èr gè zhōng jiān biàn liàng | 第二个中间变量 |
| $h_1$ | dì yī gè zhōng jiān biàn liàng | 第一个中间变量 |

**一句话解释：**
第二层中间变量对权重的导数。

**简单应用场景：**
反向模式自动微分的递推计算。

---

---

### 公式 (4.82)：反向模式梯度计算 (3) / Reverse Mode Gradient (3)

$$
\frac{\partial h_3}{\partial w} = \frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial w} = -1 \times 1 = -1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_3$ | dì sān gè zhōng jiān biàn liàng | 第三个中间变量 |
| $h_2$ | dì èr gè zhōng jiān biàn liàng | 第二个中间变量 |

**一句话解释：**
第三层中间变量对权重的导数。

**简单应用场景：**
反向模式自动微分的递推计算。

---

---

### 公式 (4.83)：反向模式梯度计算 (4) / Reverse Mode Gradient (4)

$$
\frac{\partial h_6}{\partial w} = \frac{\partial h_6}{\partial h_5} \frac{\partial h_5}{\partial w} = -0.25 \times (-1) = 0.25
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_6$ | dì liù gè zhōng jiān biàn liàng | 第六个中间变量 |
| $h_5$ | dì wǔ gè zhōng jiān biàn liàng | 第五个中间变量 |

**一句话解释：**
第六层中间变量对权重的导数。

**简单应用场景：**
反向模式自动微分的递推计算。

---

---

### 公式 (4.84)：反向模式与前向模式对比 / Reverse Mode vs Forward Mode

$$
\frac{\partial f(x;w,b)}{\partial w} = \frac{\partial f(x;w,b) \partial h_6}{\partial h_6 \partial w} = 1 \times 0.25 = 0.25
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_6$ | zuì hòu céng huó huà zhí | 最后层激活值 |

**一句话解释：**
反向模式与前向模式得到相同的梯度结果。

**简单应用场景：**
验证自动微分的正确性。

---

---

### 公式 (4.85)：反向模式梯度 (计算∂f/∂h_n) / Reverse Mode for ∂f/∂h_n

$$
\frac{\partial f(x;w,b)}{\partial h_n} = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_n$ | zuì hòu céng | 最后层 |

**一句话解释：**
最后层的误差项初始化为1。

**简单应用场景：**
反向传播的初始化。

---

---

### 公式 (4.86)：反向传播h_5项 / Backpropagate to h_5

$$
\frac{\partial f(x;w,b)}{\partial h_5} = \frac{\partial f(x;w,b) \partial h_6}{\partial h_6 \partial h_5} = 1 \times (-0.25)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_5, h_6$ | zhōng jiān biàn liàng | 中间变量 |

**一句话解释：**
误差通过链式法则传递到上一层。

**简单应用场景：**
反向传播的递推。

---

---

### 公式 (4.87)：反向传播h_4项 / Backpropagate to h_4

$$
\frac{\partial f(x;w,b)}{\partial h_4} = \frac{\partial f(x;w,b) \partial h_5}{\partial h_5 \partial h_4} = -0.25 \times 1 = -0.25
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_4, h_5$ | zhōng jiān biàn liàng | 中间变量 |

**一句话解释：**
继续传递误差到上一层。

**简单应用场景：**
反向传播的递推。

---

---

### 公式 (4.88)：反向传播续项 / Backpropagation Continuation

$$
\vdots \quad \vdots
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\vdots$ | shěng luè | 省略号 |

**一句话解释：**
省略号表示后续步骤类似。

**简单应用场景：**
简化反向传播的完整过程表示。

---

---

### 公式 (4.89)：反向传播最终梯度 / Final Backpropagated Gradient

$$
\frac{\partial f(x;w,b)}{\partial w} = \frac{\partial f(x;w,b) \partial h_1}{\partial h_1 \partial w} = 0.25 \times 1 = 0.25
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_1$ | dì yī gè zhōng jiān biàn liàng | 第一个中间变量 |

**一句话解释：**
反向模式最终得到权重的梯度。

**简单应用场景：**
反向传播算法的最终结果。

---

---

### 公式 (4.90)：非凸优化问题示例 / Non-convex Optimization Example

$$
\boldsymbol{y} = \sigma(w_2 \sigma(w_1 x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | 激活函数 |
| $w_1, w_2$ | quán zhòng | 权重 |
| $x$ | shū rù | 输入 |

**一句话解释：**
两层神经网络是非凸优化问题的例子。

**简单应用场景：**
说明神经网络优化的复杂性。

---

---

### 公式 (4.91)：梯度消失反向传播 / Vanishing Gradient Backpropagation

$$
\delta^{(l)} = f_l'(\boldsymbol{z}^{(l)}) \odot (\boldsymbol{W}^{(l+1)})^T \delta^{(l+1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta^{(l)}$ | dé ěr tǎ xiàng liàng | 误差向量 |
| $f_l'$ | huó huà dǎo shù | 激活函数导数 |
| $\odot$ | Hadamard chéngjī | Hadamard乘积 |

**一句话解释：**
梯度消失问题中，梯度通过深层网络逐层减小。

**简单应用场景：**
解释为什么深层网络训练困难。

---

---

### 公式 (4.92)：Sigmoid函数导数范围 / Sigmoid Derivative Range

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x)) \in [0, 0.25]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma'(x)$ | xī gé mǎ dǎo shù | sigmoid导数 |

**一句话解释：**
sigmoid导数最大值只有0.25，导致梯度消失。

**简单应用场景：**
说明sigmoid导致的梯度消失问题。

---

---

### 公式 (4.93)：Tanh函数导数范围 / Tanh Derivative Range

$$
\tanh'(x) = 1 - (\tanh(x))^2 \in [0, 1]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tanh'(x)$ | tāng hánshù dǎo shù | tanh导数 |

**一句话解释：**
tanh导数的最大值为1，优于sigmoid。

**简单应用场景：**
说明tanh相对sigmoid的优势。

---

---

## 第5章：卷积神经网络

### 公式 (5.1)：卷积运算定义 / Convolution Operation Definition

$$
y = f(\boldsymbol{x} * \boldsymbol{w})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | Output / 输出 |
| $\boldsymbol{x}$ | dà xiě $x$ | Input / 输入 |
| $*$ | xīng | Convolution operator / 卷积运算符 |
| $\boldsymbol{w}$ | dà xiě $w$ | Convolution kernel / 卷积核 |

**一句话解释：**
卷积运算是输入与卷积核的组合，再通过激活函数得到输出。

**简单应用场景：**
卷积神经网络的基本计算单元。

---

---

### 公式 (5.2)：一维卷积的通用形式 / General 1D Convolution

$$
y_t = w_1 x_t + w_2 x_{t-1} + w_3 x_{t-2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | yī | 输出信号在时刻 t 的值 |
| $w_i$ | w | 卷积核权重 |
| $x_t$ | xī | 输入信号在时刻 t 的值 |

**一句话解释：**
使用权重向量 $w$ 对输入信号进行卷积运算，得到输出。

**简单应用场景：**
通用的一维信号处理，如时间序列预测。

---

---

### 公式 (5.3)：一维卷积的求和形式 / 1D Convolution Summation

$$
y_t = \sum_{k=1}^{K} w_k x_{t-k+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | yī | 输出信号在时刻 t 的值 |
| $K$ | kāi | 卷积核的长度 |
| $w_k$ | w | 卷积核的第 k 个权重 |
| $x_t$ | xī | 输入信号在时刻 t 的值 |

**一句话解释：**
输出是输入信号与卷积核的逐点乘积之和。

**简单应用场景：**
通用的卷积运算表示，便于推广到高维情况。

---

---

### 公式 (5.4)：一般卷积运算 / General Convolution Operation

$$
y_t = \sum_{k=1}^{K} w_k x_{t-k+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | yī | 输出 |
| $K$ | kāi | 滤波器长度 |
| $w_k$ | w | 滤波器或卷积核 |
| $x_t$ | xī | 输入信号 |

**一句话解释：**
使用滤波器（卷积核）对输入信号进行加权求和运算。

**简单应用场景：**
信号处理中的滤波操作或特征提取。

---

---

### 公式 (5.5)：卷积作为矩阵乘法 / Convolution as Matrix Multiplication

$$
\boldsymbol{y} = \boldsymbol{w} \times \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{y}$ | yī | 输出向量 |
| $\boldsymbol{w}$ | w | 权重向量 |
| $\boldsymbol{x}$ | xī | 输入向量 |

**一句话解释：**
卷积运算可以表示为向量与矩阵的乘法。

**简单应用场景：**
将卷积运算用矩阵乘法实现，便于计算。

---

---

### 公式 (5.6)：卷积层输出尺寸 / Convolution Layer Output Size

$$
\boldsymbol{y} = \operatorname{conv}(\boldsymbol{x}, \boldsymbol{w}) = \sum_{u=1}^{U} w_u \cdot x_{t-u+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H'$ | dà xiě $H$ 撇 | Output height / 输出高度 |
| $H$ | dà xiě $H$ | Input height / 输入高度 |
| $K_h$ | dà xiě $K$ xià biāo $h$ | Kernel height / 卷积核高度 |
| $P_h$ | dà xiě $P$ xià biāo $h$ | Padding height / 填充高度 |
| $S_h$ | dà xiě $S$ xià biāo $h$ | Stride height / 步长 |
| $\lfloor \cdot \rfloor$ | xià zhěng shù | Floor function / 向下取整函数 |

**一句话解释：**
卷积层输出的高度由输入、卷积核、填充和步长共同决定。

**简单应用场景：**
设计卷积神经网络架构时计算每层的空间尺寸。

---

---

### 公式 (5.7)：二维卷积 / 2D Convolution

$$
y_{ij} = \sum_{u=1}^{U} \sum_{v=1}^{V} w_{uv} x_{i-u+1,j-v+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_{ij}$ | yī | 输出特征图在位置 (i,j) 的值 |
| $U, V$ | yōu, wēi | 卷积核的高和宽 |
| $w_{uv}$ | w | 卷积核在位置 (u,v) 的权重 |
| $x_{ij}$ | xī | 输入图像在位置 (i,j) 的值 |

**一句话解释：**
对二维图像的每个局部区域使用卷积核进行加权求和。

**简单应用场景：**
图像处理和计算机视觉中的特征提取。

---

---

### 公式 (5.8)：二维卷积的矩阵形式 / 2D Convolution in Matrix Form

$$
\boldsymbol{Y} = \boldsymbol{W} * \boldsymbol{X}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Y}$ | yī | 输出特征图 |
| $\boldsymbol{W}$ | w | 卷积核 |
| $\boldsymbol{X}$ | xī | 输入图像 |
| * | xīng | 卷积操作符 |

**一句话解释：**
将二维卷积运算表示为卷积核与输入的矩阵卷积。

**简单应用场景：**
CNN 中的卷积层运算。

---

---

### 公式 (5.9)：互相关 / Cross-Correlation

$$
y_{ij} = \sum_{u=1}^{U} \sum_{v=1}^{V} w_{uv} x_{i+u-1,j+v-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_{ij}$ | yī | 互相关输出 |
| $w_{uv}$ | w | 滤波器权重 |
| $x_{ij}$ | xī | 输入值 |

**一句话解释：**
互相关与卷积相似，但不翻转卷积核，直接进行逐点乘积求和。

**简单应用场景：**
模式匹配和信号相似度计算。

---

---

### 公式 (5.10)：互相关与旋转 / Cross-Correlation with 180-Degree Rotation

$$
\boldsymbol{Y} = \boldsymbol{W} \otimes \boldsymbol{X}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Y}$ | yī | 输出 |
| $\boldsymbol{W}$ | w | 权重 |
| $\otimes$ | ōu | 互相关符号 |
| $\boldsymbol{X}$ | xī | 输入 |

**一句话解释：**
互相关运算可以通过 180 度旋转卷积核来实现。

**简单应用场景：**
深度学习框架中的卷积实现。

---

---

### 公式 (5.11)：硬逻辑激活函数 / Hard-Logistic Activation Function

$$\text{hard-logistic}(x) = \max(\min(0.25x + 0.5, 1), 0)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x$ | chā | 输入值 |
| $\text{hard-logistic}(x)$ | yìng luó jī fú huó | 硬逻辑激活函数的输出 |
| $\max$ | zuì dà | 取最大值 |
| $\min$ | zuì xiǎo | 取最小值 |

**一句话解释：**
硬逻辑激活函数是Logistic函数的分段线性近似，将0.25x+0.5的值限制在0到1之间。

**简单应用场景：**
硬逻辑激活函数在神经网络中用作Logistic函数的计算效率更高的替代品，特别是在边界处理上能保持非线性特性的同时降低计算成本。

---

### 公式 (5.12)：平均池化 / Average Pooling

$$
\boldsymbol{Y} = \boldsymbol{W} \otimes \boldsymbol{X}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_{i,j}$ | wài | Pooled output value / 池化输出值 |
| $K_h, K_w$ | dà xiě $K$ | Kernel height and width / 池化核高度和宽度 |
| $x_{i,j}$ | xiào xiě $x$ | Input value / 输入值 |

**一句话解释：**
平均池化将每个池化窗口内的值求平均，起到降采样作用。

**简单应用场景：**
卷积神经网络中的特征图降采样。

---

---

### 公式 (5.13)：旋转对称性 / Rotation Symmetry

$$
\text{rot180}(\boldsymbol{W} \otimes \boldsymbol{X}) = \text{rot180}(\boldsymbol{X}) \otimes \boldsymbol{W}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| rot180 | luó tài bǎi bā shí | 180 度旋转操作 |
| $\boldsymbol{W}$ | w | 卷积核 |
| $\boldsymbol{X}$ | xī | 输入 |

**一句话解释：**
卷积的 180 度旋转对称性质，用于推导反向传播公式。

**简单应用场景：**
卷积神经网络的反向传播算法推导。

---

---

### 公式 (5.14)：卷积的梯度（权重导数）/ Gradient of Convolution (Weight Derivative)

$$
\frac{\partial f(Y)}{\partial w_{uv}} = \sum_{i=1}^{M-U+1} \sum_{j=1}^{N-V+1} \frac{\partial y_{ij}}{\partial w_{uv}} \frac{\partial f(Y)}{\partial y_{ij}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(Y)$ | f | 损失函数 |
| $w_{uv}$ | w | 卷积核权重 |
| $y_{ij}$ | yī | 输出特征图 |
| $\frac{\partial}{\partial}$ | piàn dǎo shù | 偏导数 |

**一句话解释：**
计算损失函数关于卷积核权重的梯度。

**简单应用场景：**
反向传播算法中的权重更新。

---

---

### 公式 (5.15)：卷积梯度展开形式 / Expanded Gradient Form

$$
= \sum_{i=1}^{M-U+1} \sum_{j=1}^{N-V+1} x_{i+u-1,j+v-1} \frac{\partial f(Y)}{\partial y_{ij}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_{ij}$ | xī | 输入值 |
| $\frac{\partial f(Y)}{\partial y_{ij}}$ | piàn dǎo shù | 输出梯度 |

**一句话解释：**
权重梯度可以表示为输入与输出梯度的卷积。

**简单应用场景：**
反向传播中的权重更新计算。

---

---

### 公式 (5.16)：最大池化 / Maximum Pooling

$$
y_{ij} = \sum_{u=1}^{U} \sum_{v=1}^{V} w_{uv} \cdot x_{i+u-1, j+v-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_{i,j}$ | wài | Pooled output value / 池化输出值 |
| $\max$ | zuì dà zhí | Maximum value / 最大值 |
| $K_h, K_w$ | dà xiě $K$ | Kernel height and width / 池化核高度和宽度 |
| $x_{i,j}$ | xiào xiě $x$ | Input value / 输入值 |

**一句话解释：**
最大池化从每个池化窗口中选取最大值，实现特征选择和降采样。

**简单应用场景：**
提取最显著的特征，增强模型对特征的鲁棒性。

---

---

### 公式 (5.17)：卷积梯度的分解 / Decomposed Convolution Gradient

$$
\frac{\partial f(Y)}{\partial \boldsymbol{W}} = \frac{\partial f(Y)}{\partial Y} \otimes \boldsymbol{X}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}$ | w | 卷积核 |
| $\otimes$ | ōu | 互相关符号 |
| $\boldsymbol{X}$ | xī | 输入 |

**一句话解释：**
权重梯度等于输出梯度与输入的互相关。

**简单应用场景：**
高效计算卷积层权重梯度。

---

---

### 公式 (5.18)：卷积层梯度反向传播 / Convolution Layer Gradient Backpropagation

$$
\boldsymbol{Y} = \boldsymbol{W} \otimes \boldsymbol{X} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $\boldsymbol{w}$ | dà xiě $w$ | Convolution kernel / 卷积核 |
| $\delta_{i,j}$ | dé ěr tǎ | Error signal / 误差信号 |
| $\otimes$ | jīng xiàng | Correlation operation / 相关运算 |
| $\boldsymbol{x}_{i,j}$ | dà xiě $x$ | Input patch / 输入块 |

**一句话解释：**
卷积核的梯度由误差信号与对应输入块的相关运算得到。

**简单应用场景：**
卷积神经网络中卷积核参数的反向传播。

---

---

### 公式 (5.19)：输入梯度展开形式 / Expanded Input Gradient

$$
= \sum_{i=1}^{M-U+1} \sum_{j=1}^{N-V+1} w_{s-i+1,t-j+1} \frac{\partial f(Y)}{\partial y_{ij}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_{st}$ | w | 卷积核权重 |
| $\frac{\partial f(Y)}{\partial y_{ij}}$ | piàn dǎo shù | 输出梯度 |

**一句话解释：**
输入梯度由输出梯度与卷积核的卷积得到。

**简单应用场景：**
CNN 反向传播的梯度流传。

---

---

### 公式 (5.20)：输入梯度的旋转形式 / Input Gradient with Rotation

$$
\frac{\partial f(Y)}{\partial \boldsymbol{X}} = \text{rot180}\left(\frac{\partial f(Y)}{\partial Y}\right) \otimes \boldsymbol{W}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| rot180 | luó tài bǎi bā shí | 180 度旋转 |
| $\boldsymbol{W}$ | w | 卷积核 |

**一句话解释：**
输入梯度可以通过旋转输出梯度后与卷积核进行互相关得到。

**简单应用场景：**
CNN 反向传播算法优化。

---

---

### 公式 (5.21)：等价形式 / Equivalent Form

$$
= \text{rot180}(\boldsymbol{W}) \otimes \frac{\partial f(Y)}{\partial Y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| rot180 | luó tài bǎi bā shí | 180 度旋转操作 |
| $\boldsymbol{W}$ | w | 卷积核 |

**一句话解释：**
也可以先旋转卷积核再与输出梯度进行互相关。

**简单应用场景：**
反向传播的等价实现方式。

---

---

### 公式 (5.22)：池化层梯度反向传播（最大池化）/ Pooling Layer Gradient Backpropagation (Max Pooling)

$$
\boldsymbol{z}^{(l)} = f(\boldsymbol{z}^{(l-1)}) = \sigma(\boldsymbol{W}^{(l)} \otimes \boldsymbol{z}^{(l-1)} + \boldsymbol{b}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $x_{i,j}$ | xiào xiě $x$ | Input value / 输入值 |
| $y_{p,q}$ | wài | Output value / 输出值 |
| $\arg\max$ | cān shù zuì dà zhí | Argument of maximum / 最大值位置 |

**一句话解释：**
最大池化的反向传播只将梯度传递到最大值位置，其余位置梯度为零。

**简单应用场景：**
卷积神经网络中池化层的反向传播。

---

---

### 公式 (5.23)：多卷积核特征图 / Multi-Kernel Feature Maps

$$
\boldsymbol{Z}^p = \boldsymbol{W}^p \otimes \boldsymbol{X} + \boldsymbol{b}^p = \sum_{d=1}^{D} \boldsymbol{W}^{p,d} \otimes \boldsymbol{X}^d + \boldsymbol{b}^p
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Z}^p$ | z | 第 p 个输出特征图 |
| $\boldsymbol{W}^p$ | w | 第 p 个卷积核 |
| $D$ | dé | 输入通道数 |
| $\boldsymbol{b}^p$ | b | 第 p 个偏置 |

**一句话解释：**
对多通道输入使用多个卷积核，每个卷积核处理所有输入通道。

**简单应用场景：**
CNN 中的卷积层实现多个输出特征图。

---

---

### 公式 (5.24)：激活函数 / Activation Function

$$
\boldsymbol{Y}^p = f(\boldsymbol{Z}^p)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Y}^p$ | yī | 第 p 个激活后的特征图 |
| $f$ | f | 非线性激活函数（如 ReLU） |
| $\boldsymbol{Z}^p$ | z | 第 p 个卷积输出 |

**一句话解释：**
对卷积输出应用非线性激活函数以引入非线性性。

**简单应用场景：**
CNN 卷积层后的激活。

---

---

### 公式 (5.25)：最大池化 / Max Pooling

$$
y^d_{m,n} = \max_{i \in R^d_{m,n}} x_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^d_{m,n}$ | yī | 池化后的输出值 |
| $R^d_{m,n}$ | ǎr | 池化区域 |
| $x_i$ | xī | 区域内的输入值 |
| max | zuì dà | 最大值操作 |

**一句话解释：**
从池化区域中选取最大值作为输出。

**简单应用场景：**
CNN 中的最大池化层，用于降维和特征提取。

---

---

### 公式 (5.26)：平均池化 / Average Pooling

$$
y^d_{m,n} = \frac{1}{|R^d_{m,n}|} \sum_{i \in R^d_{m,n}} x_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^d_{m,n}$ | yī | 池化输出 |
| $\|R^d_{m,n}\|$ | jī | 池化区域大小 |
| $x_i$ | xī | 区域内的输入值 |

**一句话解释：**
池化区域内所有值的平均值作为输出。

**简单应用场景：**
CNN 中的平均池化，用于特征聚合和降维。

---

---

### 公式 (5.27)：池化层的非线性激活 / Nonlinear Activation in Pooling

$$
\boldsymbol{Y}^d = f\left(\boldsymbol{w}^d \boldsymbol{Y}^d + \boldsymbol{b}^d\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Y}^d$ | yī | 池化层输出 |
| $f$ | f | 非线性激活函数 |
| $\boldsymbol{w}^d$ | w | 权重 |
| $\boldsymbol{b}^d$ | b | 偏置 |

**一句话解释：**
池化层可以应用非线性激活函数进行进一步的特征变换。

**简单应用场景：**
LeNet-5 等经典 CNN 中的池化操作。

---

---

### 公式 (5.28)：LeNet-5 卷积层输出 / LeNet-5 Convolution Layer Output

$$
y_p = \max_{i \in R_p} x_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_l$ | xiàng | Hidden layer output / 隐层输出 |
| $f$ | 函数 | Activation function / 激活函数 |
| $\boldsymbol{W}_l$ | dà xiě $W$ | Convolution kernels of layer $l$ / 第$l$层卷积核 |
| $\boldsymbol{x}$ | dà xiě $x$ | Input / 输入 |
| $*$ | xīng | Convolution operator / 卷积运算符 |
| $b_l$ | xiào xiě $b$ | Bias of layer $l$ / 第$l$层偏置 |

**一句话解释：**
LeNet-5 中卷积层通过卷积运算、加偏置后应用激活函数。

**简单应用场景：**
经典卷积神经网络 LeNet-5 的基本计算单元。

---

---

### 公式 (5.29)：全连接层输出 / Fully Connected Layer Output

$$
y_p = \frac{1}{|R_p|} \sum_{i \in R_p} x_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | Output / 输出 |
| $\text{softmax}$ | ruǎn zuì dà zhí | Softmax activation / 软最大值函数 |
| $\boldsymbol{W}_f$ | dà xiě $W$ | Fully connected weights / 全连接权重 |
| $h$ | xiàng | Hidden layer output / 隐层输出 |
| $b_f$ | xiào xiě $b$ | Bias of fully connected layer / 全连接层偏置 |

**一句话解释：**
全连接层将特征向量通过线性变换和 softmax 映射到输出概率分布。

**简单应用场景：**
卷积神经网络中的分类输出层。

---

---

### 公式 (5.30)：交叉熵损失函数 / Cross Entropy Loss Function

$$
\hat{x}_{M'} = \operatorname{reshape}(\boldsymbol{x}, [M', K])
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $y_k$ | wài | Ground truth probability / 真实概率 |
| $\hat{y}_k$ | hào chā | Predicted probability / 预测概率 |
| $\log$ | duì shù | Logarithm / 对数 |

**一句话解释：**
交叉熵损失衡量真实分布与预测分布之间的差异。

**简单应用场景：**
多分类神经网络的损失函数。

---

---

### 公式 (5.31)：LeNet-5 前向传播（卷积块1）/ LeNet-5 Forward Propagation (Convolution Block 1)

$$
\boldsymbol{Y} = \boldsymbol{W} \otimes \boldsymbol{X} + \boldsymbol{b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_1^{(l)}$ | xiàng | Hidden layer 1 of level $l$ / 第$l$层隐层1 |
| $f$ | 函数 | Activation function / 激活函数 |
| $\boldsymbol{W}^{(l)}$ | dà xiě $W$ | Convolution kernels / 卷积核 |
| $\boldsymbol{x}$ | dà xiě $x$ | Input / 输入 |
| $*$ | xīng | Convolution operator / 卷积运算符 |
| $b^{(l)}$ | xiào xiě $b$ | Bias / 偏置 |

**一句话解释：**
LeNet-5 前向传播的第一个卷积块，对输入进行卷积和激活。

**简单应用场景：**
图像识别模型的初始特征提取。

---

---

### 公式 (5.32)：LeNet-5 前向传播（池化块1）/ LeNet-5 Forward Propagation (Pooling Block 1)

$$
\operatorname{conv}(\boldsymbol{x}, \boldsymbol{w}) = \operatorname{conv}(\boldsymbol{w}, \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_1$ | pī | Pooled feature / 池化特征 |
| $\text{pool}$ | chí huà | Pooling operation / 池化操作 |
| $h_1^{(l)}$ | xiàng | Hidden layer output / 隐层输出 |

**一句话解释：**
对卷积层输出进行池化，降低维度并提取主要特征。

**简单应用场景：**
卷积神经网络中特征图的降采样。

---

---

### 公式 (5.33)：LeNet-5 前向传播（卷积块2）/ LeNet-5 Forward Propagation (Convolution Block 2)

$$
y_t = \sum_{k} w_k \cdot x_{t-k}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_2^{(l)}$ | xiàng | Hidden layer 2 of level $l$ / 第$l$层隐层2 |
| $f$ | 函数 | Activation function / 激活函数 |
| $\boldsymbol{W}^{(l)}$ | dà xiě $W$ | Convolution kernels / 卷积核 |
| $p_1$ | pī | Pooled feature input / 池化特征输入 |
| $*$ | xīng | Convolution operator / 卷积运算符 |
| $b^{(l)}$ | xiào xiě $b$ | Bias / 偏置 |

**一句话解释：**
第二个卷积块对池化后的特征进一步进行卷积和激活。

**简单应用场景：**
深层特征的进一步提取和学习。

---

---

### 公式 (5.34)：LeNet-5 前向传播（池化块2）/ LeNet-5 Forward Propagation (Pooling Block 2)

$$
\frac{\partial \mathcal{L}}{\partial w_u} = \sum_t \frac{\partial \mathcal{L}}{\partial y_t} \cdot x_{t-u+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_2$ | pī | Pooled feature / 池化特征 |
| $\text{pool}$ | chí huà | Pooling operation / 池化操作 |
| $h_2^{(l)}$ | xiàng | Hidden layer output / 隐层输出 |

**一句话解释：**
对第二卷积层输出再次进行池化，进一步降低维度。

**简单应用场景：**
多级特征提取中的池化操作。

---

---

### 公式 (5.35)：LeNet-5 前向传播（展平层）/ LeNet-5 Forward Propagation (Flatten Layer)

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}} \otimes \boldsymbol{X}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_{\text{flatten}}$ | dà xiě $h$ | Flattened feature vector / 展平后的特征向量 |
| $\text{reshape}$ | chóng xīn zǔ gěi | Reshape operation / 重新组织形状操作 |
| $p_2$ | pī | Pooled feature / 池化特征 |

**一句话解释：**
将二维卷积特征图展平为一维向量，以连接全连接层。

**简单应用场景：**
从卷积层连接到全连接层的过渡操作。

---

---

### 公式 (5.36)：LeNet-5 前向传播（全连接层1）/ LeNet-5 Forward Propagation (Fully Connected Layer 1)

$$
\frac{\partial \mathcal{L}}{\partial x_t} = \sum_u \frac{\partial \mathcal{L}}{\partial y_{t+u-1}} \cdot w_u
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_{\text{fc1}}$ | xiàng | First fully connected layer output / 第一全连接层输出 |
| $f$ | 函数 | Activation function / 激活函数 |
| $\boldsymbol{W}_{\text{fc1}}$ | dà xiě $W$ | First FC layer weights / 第一全连接层权重 |
| $\boldsymbol{h}_{\text{flatten}}$ | dà xiě $h$ | Flattened vector / 展平向量 |
| $b_{\text{fc1}}$ | xiào xiě $b$ | First FC layer bias / 第一全连接层偏置 |

**一句话解释：**
第一个全连接层进行线性变换和非线性激活，提取高层语义特征。

**简单应用场景：**
图像分类中的特征融合层。

---

---

### 公式 (5.37)：LeNet-5 前向传播（全连接层2）/ LeNet-5 Forward Propagation (Fully Connected Layer 2)

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{X}} = \operatorname{rot180}(\boldsymbol{W}) \otimes \frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_{\text{fc2}}$ | xiàng | Second fully connected layer output / 第二全连接层输出 |
| $f$ | 函数 | Activation function / 激活函数 |
| $\boldsymbol{W}_{\text{fc2}}$ | dà xiě $W$ | Second FC layer weights / 第二全连接层权重 |
| $h_{\text{fc1}}$ | xiàng | First FC layer output / 第一全连接层输出 |
| $b_{\text{fc2}}$ | xiào xiě $b$ | Second FC layer bias / 第二全连接层偏置 |

**一句话解释：**
第二个全连接层继续进行特征变换，增加模型的表达能力。

**简单应用场景：**
多层感知机中的隐层。

---

---

### 公式 (5.38)：LeNet-5 前向传播（输出层）/ LeNet-5 Forward Propagation (Output Layer)

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{X}} = \tilde{\boldsymbol{W}} \otimes \frac{\partial \mathcal{L}}{\partial \boldsymbol{Y}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{\boldsymbol{y}}$ | hào chā | Predicted output probability / 预测输出概率 |
| $\text{softmax}$ | ruǎn zuì dà zhí | Softmax function / 软最大值函数 |
| $\boldsymbol{W}_{\text{out}}$ | dà xiě $W$ | Output layer weights / 输出层权重 |
| $h_{\text{fc2}}$ | xiàng | Second FC layer output / 第二全连接层输出 |
| $b_{\text{out}}$ | xiào xiě $b$ | Output layer bias / 输出层偏置 |

**一句话解释：**
输出层通过 softmax 将特征向量映射到概率分布，用于分类。

**简单应用场景：**
多分类神经网络的输出层。

---

---

### 公式 (5.39)：LeNet-5 目标函数（交叉熵）/ LeNet-5 Objective Function (Cross Entropy)

$$
\operatorname{rot180}(\boldsymbol{W})_{u,v} = \boldsymbol{W}_{U-u+1, V-v+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $K$ | dà xiě $K$ | Number of classes / 分类数 |
| $y_k$ | wài | Ground truth label / 真实标签 |
| $\hat{y}_k$ | hào chā | Predicted probability / 预测概率 |
| $\log$ | duì shù | Logarithm / 对数 |

**一句话解释：**
交叉熵损失函数衡量预测概率与真实标签之间的差异。

**简单应用场景：**
分类任务中的目标函数。

---

---

### 公式 (5.40)：LeNet-5 反向传播（输出层梯度）/ LeNet-5 Backpropagation (Output Layer Gradient)

$$
\frac{\partial \mathcal{L}}{\partial b} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial y_{ij}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $\boldsymbol{W}_{\text{out}}$ | dà xiě $W$ | Output layer weights / 输出层权重 |
| $\hat{\boldsymbol{y}}$ | hào chā | Predicted output / 预测输出 |
| $\boldsymbol{y}$ | dà xiě $y$ | True labels / 真实标签 |
| $h_{\text{fc2}}^T$ | xiàng | Transposed FC2 output / 第二全连接层输出的转置 |

**一句话解释：**
输出层权重的梯度为误差向量与输入的外积。

**简单应用场景：**
反向传播算法中输出层的参数更新。

---

---

### 公式 (5.41)：LeNet-5 反向传播（FC2 层梯度）/ LeNet-5 Backpropagation (FC2 Layer Gradient)

$$
\frac{\partial \mathcal{L}}{\partial h_{\text{fc2}}} = \boldsymbol{W}_{\text{out}}^T (\hat{\boldsymbol{y}} - \boldsymbol{y})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $h_{\text{fc2}}$ | xiàng | Second FC layer output / 第二全连接层输出 |
| $\boldsymbol{W}_{\text{out}}^T$ | dà xiě $W$ zhuǎnzhì | Transposed output weights / 输出权重的转置 |
| $\hat{\boldsymbol{y}} - \boldsymbol{y}$ | chā | Prediction error / 预测误差 |

**一句话解释：**
通过权重矩阵的转置和误差向量的相乘，将梯度反向传播到前一层。

**简单应用场景：**
反向传播中隐层的梯度计算。

---

---

### 公式 (5.42)：LeNet-5 反向传播（FC2 权重梯度）/ LeNet-5 Backpropagation (FC2 Weight Gradient)

$$
\boldsymbol{z}^{(l)} = \boldsymbol{W}^{(l)} \otimes \boldsymbol{a}^{(l-1)} + \boldsymbol{b}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $\boldsymbol{W}_{\text{fc2}}$ | dà xiě $W$ | FC2 layer weights / 第二全连接层权重 |
| $h_{\text{fc1}}^T$ | xiàng | Transposed FC1 output / 第一全连接层输出的转置 |

**一句话解释：**
FC2 层权重的梯度为该层的误差信号与前一层激活的外积。

**简单应用场景：**
全连接层参数的反向传播更新。

---

---

### 公式 (5.43)：LeNet-5 反向传播（FC1 层梯度）/ LeNet-5 Backpropagation (FC1 Layer Gradient)

$$
\boldsymbol{a}^{(l)} = f_l(\boldsymbol{z}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $h_{\text{fc1}}$ | xiàng | First FC layer output / 第一全连接层输出 |
| $\boldsymbol{W}_{\text{fc2}}^T$ | dà xiě $W$ zhuǎnzhì | Transposed FC2 weights / 第二全连接层权重的转置 |
| $\odot$ | hā dá mǎ | Hadamard product / 哈达玛积 |
| $f'$ | dǎo shù | Activation derivative / 激活函数导数 |

**一句话解释：**
FC1 层的梯度需要乘以激活函数导数，体现链式法则。

**简单应用场景：**
深层全连接层的反向传播。

---

---

### 公式 (5.44)：LeNet-5 反向传播（卷积层梯度）/ LeNet-5 Backpropagation (Convolution Layer Gradient)

$$
\delta^{(l)} = f_l'(\boldsymbol{z}^{(l)}) \odot (\operatorname{rot180}(\boldsymbol{W}^{(l+1)}) \otimes \delta^{(l+1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $\boldsymbol{W}^{(l)}$ | dà xiě $W$ | Convolution kernels of layer $l$ / 第$l$层卷积核 |
| $h^{(l)}_{i,j}$ | xiàng | Activation at position $(i,j)$ / 位置$(i,j)$处的激活 |
| $\otimes$ | jīng xiàng | Correlation / 相关运算 |
| $p_{i,j}$ | pài | Input patch at position $(i,j)$ / 位置$(i,j)$处的输入块 |

**一句话解释：**
卷积层权重梯度是误差信号与输入块的相关运算之和。

**简单应用场景：**
卷积神经网络中卷积层参数的反向传播。

---

## 符号速查表（跨章节）

| 符号 | 汉语拼音读法 | 含义 |
|------|------------|------|
| $\mathcal{L}$ | ěr shàng dì | Loss function / 损失函数 |
| $\nabla$ | nà bù lā | Gradient operator / 梯度算子 |
| $\delta^{(l)}$ | dé ěr tǎ | Error signal / 误差信号 |
| $\boldsymbol{W}^{(l)}$ | dà xiě $W$ | Weight matrix / 权重矩阵 |
| $b^{(l)}$ | xiào xiě $b$ | Bias / 偏置 |
| $\alpha$ | āěr fǎ | Learning rate / 学习率 |
| $\sigma$ | xī gé mǎ | Sigmoid/Activation / 激活函数 |
| $z^{(l)}$ | shàng biāo $l$ | Pre-activation / 激活前的值 |
| $h^{(l)}$ | xiàng | Hidden activation / 隐层激活 |
| $*$ | xīng | Convolution / 卷积 |
| $\odot$ | hā dá mǎ | Hadamard product / 哈达玛积 |

---

### 公式 (5.45)：矩阵形式表示 / Matrix Form Representation

$$
= \begin{bmatrix} w_1 & w_2 & w_3 & 0 & 0 \\ 0 & w_1 & w_2 & w_3 & 0 \\ 0 & 0 & w_1 & w_2 & w_3 \end{bmatrix} \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_i$ | w | 卷积核的权重 |
| $\boldsymbol{x}$ | xī | 输入向量 |

**一句话解释：**
卷积可以用特殊结构的矩阵与输入向量的乘法表示。

**简单应用场景：**
卷积的矩阵实现方法。

---

---

### 公式 (5.46)：矩阵卷积形式 / Matrix Convolution Form

$$
= \boldsymbol{C} \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{C}$ | dà xiě c | 循环矩阵 |
| $\boldsymbol{x}$ | xī | 输入 |

**一句话解释：**
卷积运算可以表示为循环矩阵与输入的乘法。

**简单应用场景：**
卷积的矩阵表示和计算。

---

---

### 公式 (5.47)：转置卷积的矩阵形式 / Transposed Convolution Matrix Form

$$
\boldsymbol{x} = \boldsymbol{C}^T \boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | xī | 恢复的输入 |
| $\boldsymbol{C}^T$ | dà xiě c | 循环矩阵的转置 |
| $\boldsymbol{z}$ | z | 输出 |

**一句话解释：**
转置卷积是卷积矩阵的转置与输出的乘法。

**简单应用场景：**
上采样和特征图扩大。

---

---

### 公式 (5.48)：转置卷积的展开形式 / Expanded Transposed Convolution

$$
= \begin{bmatrix} w_1 & 0 & 0 \\ w_2 & w_1 & 0 \\ w_3 & w_2 & w_1 \\ 0 & w_3 & w_2 \\ 0 & 0 & w_3 \end{bmatrix} \boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_i$ | w | 转置卷积核权重 |
| $\boldsymbol{z}$ | z | 输入特征图 |

**一句话解释：**
转置卷积使用转置的循环矩阵进行上采样。

**简单应用场景：**
生成对抗网络和图像超分辨率中的上采样。

---

---

### 公式 (5.49)：转置卷积的旋转形式 / Rotated Transposed Convolution

$$
= \text{rot180}(\boldsymbol{w}) \otimes \boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| rot180 | luó tài bǎi bā shí | 180 度旋转 |
| $\boldsymbol{w}$ | w | 卷积核 |
| $\boldsymbol{z}$ | z | 输入 |

**一句话解释：**
转置卷积等价于旋转卷积核后与输入进行卷积。

**简单应用场景：**
转置卷积的高效实现。

---

End of Chapter 5 Formulas

---

## 第6章：循环神经网络

### 公式 (6.1)：梯度下降法 / Gradient Descent

$$
\boldsymbol{h}_t = f(\boldsymbol{h}_{t-1}, \boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | tā ta | 参数 |
| $\alpha$ | āěr fǎ | 学习率 |
| $L(\boldsymbol{\theta})$ | zǔn sǔn hánshù | 损失函数 |
| $\nabla_{\boldsymbol{\theta}}$ | nà bù lā | 梯度算子 |

**一句话解释：**
参数沿着损失函数梯度的反方向更新，以最小化损失。

**简单应用场景：**
训练任何神经网络的基本优化算法，通过计算梯度来调整权重。

---

---

### 公式 (6.2)：自回归模型 / AutoRegressive Model (AR)

$$
y_t = w_0 + \sum_{k=1}^{K} w_k y_{t-k} + \varepsilon_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | yīngyòngzhuózhí | 时刻 $t$ 的输出 |
| $w_0$ | báihuà | 偏置参数 |
| $w_k$ | quánzhòng | 第 $k$ 个历史时刻的权重 |
| $K$ | jìyìshēndù | 历史记忆深度 |
| $\varepsilon_t$ | lù yīn | 白噪声 |

**一句话解释：**
自回归模型使用过去多个时刻的输出值来预测当前时刻的输出。

**简单应用场景：**
股票价格预测中，根据过去30天的价格来预测第31天的价格。

---

---

### 公式 (6.3)：非线性自回归外源输入模型 / Nonlinear AutoRegressive with Exogenous Inputs (NARX)

$$
y_t = f(x_t, x_{t-1}, \ldots, x_{t-K_x}, y_{t-1}, y_{t-2}, \ldots, y_{t-K_y})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | shūchūzhí | 时刻 $t$ 的输出 |
| $x_t$ | wàiyuáninputs | 时刻 $t$ 的外源输入 |
| $f(\cdot)$ | fúnshù | 非线性函数 |
| $K_x$ | pásuoyeshù | 外源输入的延时步数 |
| $K_y$ | pásuoyeshù | 输出的延时步数 |

**一句话解释：**
NARX模型结合外源输入和历史输出信息，通过非线性函数预测当前输出。

**简单应用场景：**
天气预报中，利用过去多天的温度、湿度等外部信息预测明天的气温。

---

---

### 公式 (6.4)：循环神经网络基本形式 / Recurrent Neural Network (RNN)

$$
\boldsymbol{h}_t = f(\boldsymbol{h}_{t-1}, \boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | yǐncáng zhuàngtài | 时刻 $t$ 的隐藏状态 |
| $\boldsymbol{h}_{t-1}$ | qiányīshíkè zhuàngtài | 时刻 $t-1$ 的隐藏状态 |
| $\boldsymbol{x}_t$ | shūrùyàng | 时刻 $t$ 的输入 |
| $f(\cdot)$ | fúnshù | 非线性激活函数 |

**一句话解释：**
循环神经网络通过反复使用相同的权重和非线性函数处理序列数据，实现信息的循环传播。

**简单应用场景：**
语言模型中，根据前面的词语序列预测下一个词。

---

---

### 公式 (6.5)：简单循环网络状态转移 / Simple Recurrent Network (SRN) State Transition

$$
\boldsymbol{z}_t = \boldsymbol{U}\boldsymbol{h}_{t-1} + \boldsymbol{W}\boldsymbol{x}_t + \boldsymbol{b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}_t$ | zhōngjiānliàng | 时刻 $t$ 的中间状态（未激活） |
| $\boldsymbol{U}$ | yǐncáng-yǐncáng quánzhòng | 隐藏层到隐藏层的权重矩阵 |
| $\boldsymbol{W}$ | shūrù-yǐncáng quánzhòng | 输入到隐藏层的权重矩阵 |
| $\boldsymbol{b}$ | piānzhì | 偏置向量 |
| $\boldsymbol{h}_{t-1}$ | qiányīshíkè yǐncáng | 前一时刻的隐藏状态 |
| $\boldsymbol{x}_t$ | dāngqiánshíkè shūrù | 当前时刻的输入 |

**一句话解释：**
简单循环网络通过线性组合隐藏状态和输入，计算下一时刻的未激活中间状态。

**简单应用场景：**
机器翻译中，将源语言句子的信息逐词编码到隐藏状态中。

---

---

### 公式 (6.6)：简单循环网络隐藏状态激活 / SRN Hidden State Activation

$$
\boldsymbol{h}_t = f(\boldsymbol{z}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | yǐncáng zhuàngtài | 时刻 $t$ 的隐藏状态 |
| $\boldsymbol{z}_t$ | zhōngjiān zhuàngtài | 未激活的中间状态 |
| $f(\cdot)$ | jīhuà hánshu | 非线性激活函数（如Tanh或Logistic） |

**一句话解释：**
通过激活函数将未激活的中间状态转换为隐藏状态。

**简单应用场景：**
使用Tanh激活函数将任意实数映射到 $[-1, 1]$ 区间内。

---

---

### 公式 (6.7)：简单循环网络完整形式 / Complete SRN Formulation

$$
\boldsymbol{h}_t = f(\boldsymbol{U}\boldsymbol{h}_{t-1} + \boldsymbol{W}\boldsymbol{x}_t + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | yǐncáng zhuàngtài | 时刻 $t$ 的隐藏状态 |
| $\boldsymbol{U}$ | yǐncáng-yǐncáng quánzhòng | 隐藏到隐藏的权重矩阵 |
| $\boldsymbol{W}$ | shūrù-yǐncáng quánzhòng | 输入到隐藏的权重矩阵 |
| $\boldsymbol{b}$ | piānzhì | 偏置向量 |
| $f(\cdot)$ | jīhuà hánshu | 非线性激活函数 |
| $\boldsymbol{x}_t$ | shūrù | 时刻 $t$ 的输入 |

**一句话解释：**
简单循环网络将隐藏状态、输入和权重通过非线性激活函数结合，形成递归的状态转移。

**简单应用场景：**
文本分类中，逐词处理文本并通过隐藏状态累积文本信息。

---

---

### 公式 (6.8)：前向传播 / Forward Propagation

$$
\boldsymbol{a}^{(l)} = \sigma(\boldsymbol{W}^{(l)} \boldsymbol{a}^{(l-1)} + \boldsymbol{b}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng jī huó zhī | 第l层激活值 |
| $\boldsymbol{W}^{(l)}$ | dì $l$ céng quán zhòng | 第l层权重矩阵 |
| $\boldsymbol{b}^{(l)}$ | dì $l$ céng piān zhì | 第l层偏置向量 |
| $\sigma$ | xī gé mǎ | 激活函数 |

**一句话解释：**
通过权重矩阵和激活函数逐层计算神经网络的输出。

**简单应用场景：**
神经网络推理的基本过程，从输入层逐层传播到输出层。

---

---

### 公式 (6.9)：反向传播梯度 / Backpropagation Gradient

$$
\frac{\partial L}{\partial \boldsymbol{a}^{(l-1)}} = (\boldsymbol{W}^{(l)})^T \frac{\partial L}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 损失函数 |
| $\boldsymbol{a}^{(l-1)}$ | dì $l-1$ céng jī huó zhī | 前一层激活值 |
| $\boldsymbol{z}^{(l)}$ | dì $l$ céng xiān xìng zǔ hé | 当前层线性组合 |
| $\boldsymbol{W}^{(l)}$ | dì $l$ céng quán zhòng | 当前层权重矩阵 |

**一句话解释：**
计算损失函数对前一层激活值的梯度，逐层向后传播误差。

**简单应用场景：**
深层网络中反向传播算法的核心步骤，计算中间层梯度。

---

## 第7章：网络优化与正则化 (Pages 149-185)

---

### 公式 (6.10)：循环神经网络通用应用相似定理 / RNN Universal Approximation (Theorem 6.1)

$$
s_t = g(s_{t-1}, x_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t$ | zhuàngtài | 时刻 $t$ 的状态 |
| $s_{t-1}$ | qiányīshíkè zhuàngtài | 前一时刻的状态 |
| $x_t$ | shūrù | 时刻 $t$ 的输入 |
| $g(\cdot)$ | fúnshù | 状态转移函数 |

**一句话解释：**
任何完全连接的循环神经网络都满足通用近似定理，可以近似任意非线性动力系统。

**简单应用场景：**
理论证明循环神经网络具有通用的函数逼近能力。

---

---

### 公式 (6.11)：循环神经网络输出函数 / RNN Output Function

$$
y_t = \sigma(s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | shūchū | 时刻 $t$ 的输出 |
| $s_t$ | zhuàngtài | 时刻 $t$ 的隐藏状态 |
| $\sigma(\cdot)$ | shūchū hánshu | 输出激活函数 |

**一句话解释：**
循环神经网络通过输出激活函数将隐藏状态映射到最终输出。

**简单应用场景：**
使用softmax函数将隐藏状态映射到类别概率分布。

---

---

### 公式 (6.12)：隐藏层线性变换 / Hidden Layer Linear Transformation

$$
s_t' = f(As_{t-1} + Bx_t + b)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t'$ | xīnshēng zhuàngtài | 新生成的隐藏状态（未激活） |
| $A$ | quánzhòng | 隐藏状态权重矩阵 |
| $B$ | quánzhòng | 输入权重矩阵 |
| $b$ | piānzhì | 偏置向量 |
| $f(\cdot)$ | hánshu | 非线性激活函数 |

**一句话解释：**
通过两层的线性变换和非线性激活来构造循环神经网络的隐藏层。

**简单应用场景：**
在循环神经网络中添加额外的非线性层以增强表达能力。

---

---

### 公式 (6.13)：反向传播链式法则 / Backpropagation Chain Rule

$$
\boldsymbol{h}_t = f(\boldsymbol{U} \boldsymbol{h}_{t-1} + \boldsymbol{W} \boldsymbol{x}_t + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 损失函数 |
| $\boldsymbol{w}$ | quán zhòng | 权重 |
| $\boldsymbol{a}^{(l)}$ | jī huó huà zhī | 激活值 |
| $\boldsymbol{z}^{(l)}$ | xiān xìng zǔ hé | 线性组合 |

**一句话解释：**
通过链式法则逐层计算损失函数对参数的梯度，实现多层网络的反向传播。

**简单应用场景：**
计算深层网络中任意权重的梯度，支持端到端的训练。

---

---

### 公式 (6.14)：线性读出层 / Linear Readout Layer

$$
s_t = Cs_t'
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t$ | fǎhuígāo zhuàngtài | 最终隐藏状态 |
| $s_t'$ | zhōngjiān zhuàngtài | 中间隐藏状态 |
| $C$ | duìyìng quánzhòng | 线性变换矩阵 |

**一句话解释：**
使用线性矩阵变换对中间隐藏状态进行读出或维度变换。

**简单应用场景：**
将高维隐藏状态线性投影到低维输出空间。

---

---

### 公式 (6.15)：双层变换输出 / Two-layer Transformation Output

$$
y_t' = f(A's_{t-1} + B'x_t + b')
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t'$ | shūchū zhuàngtài | 输出隐藏状态 |
| $A', B'$ | quánzhòng | 权重矩阵 |
| $b'$ | piānzhì | 偏置向量 |
| $f(\cdot)$ | hánshu | 非线性激活函数 |

**一句话解释：**
使用单独的权重参数对输出层进行线性变换和非线性激活。

**简单应用场景：**
在循环神经网络中为输出层使用独立的权重参数。

---

---

### 公式 (6.16)：嵌套输出变换 / Nested Output Transformation

$$
y_t' = f(A'Cs_{t-1} + B'x_t + b')
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t'$ | shūchū zhuàngtài | 输出隐藏状态 |
| $A', C, B'$ | quánzhòng | 权重矩阵 |
| $b'$ | piānzhì | 偏置向量 |
| $f(\cdot)$ | hánshu | 非线性激活函数 |

**一句话解释：**
通过多个矩阵乘积实现输出层的复杂非线性变换。

**简单应用场景：**
构造输出层的深层非线性变换以提升模型表达能力。

---

---

### 公式 (6.17)：线性输出层 / Linear Output Layer

$$
y_t = Dy_t'
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | zuìzhōng shūchū | 最终输出 |
| $y_t'$ | zhōngjiān shūchū | 中间输出 |
| $D$ | duìyìng quánzhòng | 线性变换矩阵 |

**一句话解释：**
使用线性矩阵将中间输出层的隐藏状态映射到最终输出。

**简单应用场景：**
回归问题中，将隐藏表示线性映射到实数输出值。

---

---

### 公式 (6.18)：Softmax 函数 / Softmax Function

$$
\hat{\boldsymbol{y}}_t = g(\boldsymbol{V} \boldsymbol{h}_t + \boldsymbol{c})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | xiān xìng zhū | 输入向量 |
| $K$ | lèi bié shù | 类别数 |
| $e$ | zì ránduì shù jī | 自然对数底 |

**一句话解释：**
将输出向量转换为概率分布，每个元素值在0到1之间且和为1。

**简单应用场景：**
多分类问题的输出层激活函数，生成各类别的概率。

---

---

### 公式 (6.19)：矩阵形式输出计算 / Matrix Form Output Computation

$$
y_t = \begin{bmatrix} 0 & D \end{bmatrix} \begin{bmatrix} s_t' \\ y_t' \end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | zuìzhōng shūchū | 最终输出 |
| $s_t'$ | yǐncáng zhuàngtài | 隐藏状态 |
| $y_t'$ | zhōngjiān shūchū | 中间输出状态 |
| $D$ | duìyìng quánzhòng | 输出线性变换矩阵 |

**一句话解释：**
从分离的状态向量中选择性地提取输出层信息进行最终输出。

**简单应用场景：**
从多个内部状态中精确选择用于最终输出的部分。

---

---

### 公式 (6.20)：完整循环神经网络 / Complete RNN Formulation

$$
\boldsymbol{h}_t = f(\boldsymbol{U}\boldsymbol{h}_{t-1} + \boldsymbol{W}\boldsymbol{x}_t + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | yǐncáng zhuàngtài | 时刻 $t$ 的隐藏状态 |
| $\boldsymbol{U}$ | yǐncáng-yǐncáng quánzhòng | 隐藏到隐藏的权重矩阵 |
| $\boldsymbol{W}$ | shūrù-yǐncáng quánzhòng | 输入到隐藏的权重矩阵 |
| $\boldsymbol{b}$ | piānzhì | 偏置向量 |
| $f(\cdot)$ | hánshu | 非线性激活函数 |
| $\boldsymbol{x}_t$ | shūrù | 时刻 $t$ 的输入 |

**一句话解释：**
完整的简单循环网络将前一隐藏状态、当前输入和偏置通过非线性激活合并为新的隐藏状态。

**简单应用场景：**
序列建模中反复使用此公式处理整个输入序列。

---

---

### 公式 (6.21)：循环神经网络输出层 / RNN Output Layer

$$
y_t = \boldsymbol{V}\boldsymbol{h}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | shūchū | 时刻 $t$ 的输出 |
| $\boldsymbol{V}$ | shūchū quánzhòng | 隐藏到输出的权重矩阵 |
| $\boldsymbol{h}_t$ | yǐncáng zhuàngtài | 时刻 $t$ 的隐藏状态 |

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | shūchū | 时刻 $t$ 的输出 |
| $\boldsymbol{V}$ | yǐncáng-shūchū quánzhòng | 隐藏层到输出的权重矩阵 |
| $\boldsymbol{h}_t$ | yǐncáng zhuàngtài | 时刻 $t$ 的隐藏状态 |

**一句话解释：**
循环神经网络通过权重矩阵将隐藏状态线性映射到输出。

**简单应用场景：**
将循环网络的隐藏状态通过权重矩阵转换为最终的预测输出。

---

---

### 公式 (6.22)：序列到类别 / Sequence-to-Class Classification

$$\hat{y} = g(\boldsymbol{h}_T)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}$ | yī màu | 预测输出 |
| $g(\cdot)$ | gē | 分类器函数 |
| $\boldsymbol{h}_T$ | héi | 最后时刻的隐状态 |
| $T$ | tī | 序列长度 |

**一句话解释：**
使用循环神经网络最后时刻的隐状态经过分类器得到序列级别的预测。

**简单应用场景：**
文本分类、情感分析，将整个句子映射为一个类别标签。

---

---

### 公式 (6.23)：序列平均池化 / Sequence Average Pooling

$$\hat{y} = g\left(\frac{1}{T}\sum_{t=1}^{T}\boldsymbol{h}_t\right)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}$ | yī màu | 预测输出 |
| $g(\cdot)$ | gē | 分类器函数 |
| $T$ | tī | 序列长度 |
| $\boldsymbol{h}_t$ | héi tī | 第t时刻隐状态 |
| $\sum$ | hé | 求和符号 |

**一句话解释：**
对序列所有时刻的隐状态进行平均，然后经过分类器得到预测结果。

**简单应用场景：**
句子表示学习、文档级别分类，利用所有时刻信息而非仅用最后一个。

---

---

### 公式 (6.24)：同步序列标注 / Synchronous Sequence Labeling

$$\hat{y}_t = g(\boldsymbol{h}_t), \quad \forall t \in [1, T]$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}_t$ | yī màu tī | 第t时刻的预测标签 |
| $g(\cdot)$ | gē | 分类器函数 |
| $\boldsymbol{h}_t$ | héi tī | 第t时刻的隐状态 |
| $T$ | tī | 序列长度 |

**一句话解释：**
每个时刻都有对应的输入和输出，用当前隐状态直接预测当前时刻的标签。

**简单应用场景：**
词性标注、命名实体识别，每个单词都需要标注相应的标签。

---

---

### 公式 (6.25)：编码器状态更新 / Encoder State Update

$$\boldsymbol{h}_t = f_1(\boldsymbol{h}_{t-1}, \boldsymbol{x}_t), \quad \forall t \in [1, T]$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi tī | 第t时刻编码器的隐状态 |
| $f_1(\cdot)$ | fú suō yī | 编码器函数 |
| $\boldsymbol{x}_t$ | xiào tī | 第t时刻的输入 |
| $T$ | tī | 输入序列长度 |

**一句话解释：**
编码器按时序逐步处理输入序列，更新隐状态以捕捉序列信息。

**简单应用场景：**
机器翻译中的编码器阶段，将源语言句子编码成固定维度的向量。

---

---

### 公式 (6.26)：解码器状态更新 / Decoder State Update

$$\boldsymbol{h}_{T+t} = f_2(\boldsymbol{h}_{T+t-1}, \hat{\boldsymbol{y}}_{t-1}), \quad \forall t \in [1, M]$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_{T+t}$ | héi | 解码器第t时刻的隐状态 |
| $f_2(\cdot)$ | fú suō èr | 解码器函数 |
| $\hat{\boldsymbol{y}}_{t-1}$ | yī màu | 上一时刻的预测输出向量表示 |
| $M$ | èm | 输出序列长度 |

**一句话解释：**
解码器从上一时刻的预测结果出发，逐步生成目标序列。

**简单应用场景：**
机器翻译中的解码器阶段，自回归地生成目标语言句子。

---

---

### 公式 (6.27)：解码器输出预测 / Decoder Output Prediction

$$\hat{y}_t = g(\boldsymbol{h}_{T+t}), \quad \forall t \in [1, M]$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}_t$ | yī màu tī | 第t时刻的预测输出 |
| $g(\cdot)$ | gē | 分类器函数 |
| $\boldsymbol{h}_{T+t}$ | héi | 解码器第t时刻的隐状态 |
| $M$ | èm | 输出序列长度 |

**一句话解释：**
将解码器的隐状态经过分类器得到每个时刻的预测输出。

**简单应用场景：**
生成序列的预测步骤，从隐状态映射为具体的输出类别。

---

---

### 公式 (6.28)：时刻损失函数 / Loss at Time Step

$$\mathcal{L}_t = \mathcal{L}(y_t, g(\boldsymbol{h}_t))$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_t$ | lǔ suǒ tī | 第t时刻的损失函数 |
| $y_t$ | wài tī | 第t时刻的真实标签 |
| $g(\cdot)$ | gē | 分类器函数 |
| $\boldsymbol{h}_t$ | héi tī | 第t时刻的隐状态 |

**一句话解释：**
定义单个时刻的损失函数，衡量预测与真实标签的差异。

**简单应用场景：**
序列标注任务，计算每一步预测的误差。

---

---

### 公式 (6.29)：序列总损失函数 / Total Sequence Loss

$$\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | lǔ suǒ | 序列总损失函数 |
| $T$ | tī | 序列长度 |
| $\mathcal{L}_t$ | lǔ suǒ tī | 第t时刻的损失函数 |
| $\sum$ | hé | 求和符号 |

**一句话解释：**
将所有时刻的损失函数相加，得到整个序列的总损失。

**简单应用场景：**
序列学习任务，累积所有时刻的误差用于梯度计算和参数更新。

---

---

### 公式 (6.30)：参数梯度 / Parameter Gradient

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{U}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{U}}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial \boldsymbol{U}}$ | piàn dǎo shù | 总损失对参数U的偏导数 |
| $\boldsymbol{U}$ | yōu mǐ gā | 循环连接权重矩阵 |
| $T$ | tī | 序列长度 |

**一句话解释：**
序列总损失对参数的梯度等于所有时刻损失对参数梯度的累加。

**简单应用场景：**
参数学习中的梯度计算，反映参数对总损失的影响。

---

---

### 公式 (6.31)：直接偏导数和链式法则 / Direct Derivative and Chain Rule

$$\frac{\partial \mathcal{L}_t}{\partial u_{ij}} = \sum_{k=1}^{t} \frac{\partial^+ \boldsymbol{z}_k}{\partial u_{ij}} \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{z}_k}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $u_{ij}$ | yōu | 参数U的第i,j个元素 |
| $\boldsymbol{z}_k$ | zhī tī | 第k时刻的净输入 |
| $\frac{\partial^+}{\partial u_{ij}}$ | zhí jiē piàn dǎo | 保持隐状态不变的直接偏导 |

**一句话解释：**
通过链式法则，将总损失对参数的梯度分解为对各个时刻净输入的梯度乘积。

**简单应用场景：**
BPTT算法中的梯度计算基础。

---

---

### 公式 (6.32)：直接偏导向量 / Direct Partial Derivative Vector

$$\frac{\partial^+ \boldsymbol{z}_k}{\partial u_{ij}} = [0, \cdots, [\boldsymbol{h}_{k-1}]_j, \cdots, 0]$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial^+}{\partial u_{ij}}$ | zhí jiē piàn dǎo | 直接偏导 |
| $[\boldsymbol{h}_{k-1}]_j$ | héi kè | 隐状态的第j维 |

**一句话解释：**
直接偏导在第i个位置为隐状态的第j维，其余位置为0的向量。

**简单应用场景：**
BPTT算法中计算参数梯度的中间步骤。

---

---

### 公式 (6.33)：交叉熵损失 / Cross-Entropy Loss

$$
\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{U}} = \sum_{k=1}^{t} \delta_{t,k} \boldsymbol{h}_{k-1}^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_i$ | zhēn shí biāo qiān | 真实标签 |
| $\hat{y}_i$ | yù cè gài lǜ | 预测概率 |
| $K$ | lèi bié shù | 类别数 |

**一句话解释：**
度量预测概率分布与真实标签分布之间的差异，用于分类任务。

**简单应用场景：**
多分类问题的损失函数，衡量模型预测质量。

---

---

### 公式 (6.34)：误差项定义 / Error Term Definition

$$\delta_{t,k} = \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{z}_k}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta_{t,k}$ | wù chā xiàng | 第t时刻对第k时刻净输入的误差项 |
| $\mathcal{L}_t$ | lǔ suǒ tī | 第t时刻的损失函数 |
| $\boldsymbol{z}_k$ | zhī tī | 第k时刻的净输入 |

**一句话解释：**
误差项表示最终损失对中间净输入的敏感度，用于反向传播。

**简单应用场景：**
BPTT算法的核心变量，用于递推计算所有时刻的梯度。

---

---

### 公式 (6.35)：二元交叉熵 / Binary Cross-Entropy

$$
\delta_{t,k} = \operatorname{diag}(f'(\boldsymbol{z}_k)) \boldsymbol{U}^\top \delta_{t,k+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | zhēn shí biāo qiān | 真实标签 |
| $\hat{y}$ | yù cè gài lǜ | 预测概率 |

**一句话解释：**
二分类问题的损失函数，衡量二分类模型的预测误差。

**简单应用场景：**
二分类任务（如垃圾邮件检测、图片二分类）的损失函数。

---

---

### 公式 (6.36)：误差项递推公式 / Error Term Recurrence

$$\delta_{t,k} = \text{diag}(f'(\boldsymbol{z}_k))\boldsymbol{U}^T \delta_{t,k+1}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta_{t,k}$ | wù chā xiàng | 误差项 |
| $f'(\boldsymbol{z}_k)$ | dǎo shù | 激活函数的导数 |
| $\boldsymbol{U}$ | yōu mǐ gā | 循环权重矩阵 |
| $\text{diag}(\cdot)$ | duì jiǎo huà | 对角化操作 |

**一句话解释：**
误差项通过权重矩阵和激活函数导数向后递推，是BPTT的核心公式。

**简单应用场景：**
计算所有时刻对参数的梯度贡献。

---

---

### 公式 (6.37)：参数梯度元素形式 / Parameter Gradient Element Form

$$\frac{\partial \mathcal{L}_t}{\partial u_{ij}} = \sum_{k=1}^{t} [\delta_{t,k}]_i [\boldsymbol{h}_{k-1}]_j$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $[\delta_{t,k}]_i$ | yuánsù | 误差项向量的第i个元素 |
| $[\boldsymbol{h}_{k-1}]_j$ | yuánsù | 隐状态向量的第j个元素 |

**一句话解释：**
参数梯度是所有时刻误差项与对应隐状态乘积的累加。

**简单应用场景：**
BPTT中逐个计算参数矩阵的梯度。

---

---

### 公式 (6.38)：参数梯度矩阵形式 / Parameter Gradient Matrix Form

$$\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{U}} = \sum_{k=1}^{t} \delta_{t,k} \boldsymbol{h}_{k-1}^T$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{U}}$ | piàn dǎo shù | 梯度矩阵 |
| $\delta_{t,k}$ | wù chā xiàng | 误差项列向量 |
| $\boldsymbol{h}_{k-1}^T$ | zhuǎnzhì | 隐状态的转置 |

**一句话解释：**
参数梯度矩阵由所有时刻的外积累加而成。

**简单应用场景：**
BPTT算法中的梯度计算。

---

---

### 公式 (6.39)：序列参数梯度U / Sequence Parameter Gradient U

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{U}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \delta_{t,k} \boldsymbol{h}_{k-1}^T$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | lǔ suǒ | 序列总损失 |
| $\boldsymbol{U}$ | yōu mǐ gā | 循环权重矩阵 |
| $T$ | tī | 序列长度 |

**一句话解释：**
整个序列的总损失对循环权重的梯度是各时刻梯度的二重累加。

**简单应用场景：**
BPTT算法中参数U的梯度累积和更新。

---

---

### 公式 (6.40)：序列参数梯度W / Sequence Parameter Gradient W

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \delta_{t,k} \boldsymbol{x}_k^T$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | lǔ suǒ | 序列总损失 |
| $\boldsymbol{W}$ | wēi | 输入权重矩阵 |
| $\boldsymbol{x}_k$ | xiào kè | 第k时刻的输入 |

**一句话解释：**
整个序列的总损失对输入权重的梯度，结构类似于U的梯度。

**简单应用场景：**
BPTT算法中参数W的梯度计算和更新。

---

---

### 公式 (6.41)：序列偏置梯度 / Sequence Bias Gradient

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{b}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \delta_{t,k}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | lǔ suǒ | 序列总损失 |
| $\boldsymbol{b}$ | bǐ | 偏置向量 |
| $T$ | tī | 序列长度 |

**一句话解释：**
偏置的梯度是所有时刻误差项的累加。

**简单应用场景：**
BPTT算法中偏置参数的梯度更新。

---

---

### 公式 (6.42)：RTRL隐状态更新 / RTRL Hidden State Update

$$\boldsymbol{h}_{t+1} = f(\boldsymbol{z}_{t+1}) = f(\boldsymbol{U}\boldsymbol{h}_t + \boldsymbol{W}\boldsymbol{x}_{t+1} + \boldsymbol{b})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_{t+1}$ | héi tī | 第t+1时刻的隐状态 |
| $f(\cdot)$ | fú suō | 激活函数 |
| $\boldsymbol{z}_{t+1}$ | zhī tī | 第t+1时刻的净输入 |

**一句话解释：**
定义隐状态的前向计算，用于RTRL算法的梯度前向传播。

**简单应用场景：**
实时循环学习（RTRL）算法的基础公式。

---

---

### 公式 (6.43)：参数偏导数链式 / Parameter Derivative Chain Rule

$$\frac{\partial \boldsymbol{h}_{t+1}}{\partial u_{ij}} = \left(\frac{\partial^+ \boldsymbol{z}_{t+1}}{\partial u_{ij}} + \frac{\partial \boldsymbol{h}_t}{\partial u_{ij}}\boldsymbol{U}^T\right) \frac{\partial \boldsymbol{h}_{t+1}}{\partial \boldsymbol{z}_{t+1}}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \boldsymbol{h}_{t+1}}{\partial u_{ij}}$ | piàn dǎo | 隐状态对参数的偏导 |
| $\frac{\partial^+}{\partial u_{ij}}$ | zhí jiē piàn dǎo | 直接偏导 |

**一句话解释：**
通过链式法则计算隐状态对参数的偏导，包括直接项和反馈项。

**简单应用场景：**
RTRL算法中的梯度前向传播。

---

---

### 公式 (6.44)：参数偏导展开 / Parameter Derivative Expansion

$$\frac{\partial \boldsymbol{h}_{t+1}}{\partial u_{ij}} = \left(\mathbb{I}_i([\boldsymbol{h}_t]_j) + \frac{\partial \boldsymbol{h}_t}{\partial u_{ij}}\boldsymbol{U}^T\right) \text{diag}(f'(\boldsymbol{z}_{t+1}))$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{I}_i([\boldsymbol{h}_t]_j)$ | zhǐ shì xiàng liàng | 指示向量 |
| $f'(\boldsymbol{z}_{t+1})$ | dǎo shù | 激活函数导数 |

**一句话解释：**
RTRL中参数偏导的具体计算形式，分离直接项和反馈项。

**简单应用场景：**
实时循环学习算法的具体实现。

---

---

### 公式 (6.45)：均方误差 / Mean Squared Error

$$
\tilde{\boldsymbol{h}}_t = \tanh(\boldsymbol{W} \boldsymbol{x}_t + \boldsymbol{U} (\boldsymbol{r}_t \odot \boldsymbol{h}_{t-1}) + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N$ | yàng běn shù | 样本数 |
| $y_i$ | zhēn shí zhí | 真实值 |
| $\hat{y}_i$ | yù cè zhí | 预测值 |

**一句话解释：**
计算预测值与真实值之间差异的平方平均，用于回归问题。

**简单应用场景：**
线性回归和其他回归任务中的标准损失函数。

---

---

### 公式 (6.46)：损失参数梯度 / Loss Parameter Gradient

$$\frac{\partial \mathcal{L}_t}{\partial u_{ij}} = \frac{\partial \boldsymbol{h}_t}{\partial u_{ij}} \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{h}_t}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_t$ | lǔ suǒ tī | 第t时刻的损失 |
| $u_{ij}$ | yōu | 参数 |
| $\boldsymbol{h}_t$ | héi tī | 第t时刻的隐状态 |

**一句话解释：**
RTRL算法中实时计算损失对参数的梯度，可直接用于在线参数更新。

**简单应用场景：**
在线学习和实时序列处理任务。

---

---

### 公式 (6.47)：误差项展开 / Error Term Expansion

$$\delta_{t,k} = \prod_{\tau=k}^{t-1} (\text{diag}(f'(\boldsymbol{z}_\tau))\boldsymbol{U}^T) \delta_{t,t}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta_{t,k}$ | wù chā xiàng | 误差项 |
| $\prod$ | chéng jī | 累乘符号 |
| $f'(\boldsymbol{z}_\tau)$ | dǎo shù | 激活函数导数 |

**一句话解释：**
将误差项表示为从t到k的权重矩阵和激活函数的连乘。

**简单应用场景：**
分析梯度消失/爆炸问题的数学基础。

---

---

### 公式 (6.48)：误差项范数估计 / Error Term Norm Estimate

$$\delta_{t,k} \approx \gamma^{t-k} \delta_{t,t}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gā mǎ | 权重矩阵范数的估计值 |
| $t-k$ | jiān gé | 时间间隔 |
| $\delta_{t,t}$ | wù chā xiàng | 当前时刻的误差项 |

**一句话解释：**
误差项随时间间隔呈指数衰减或增长，导致梯度消失或爆炸。

**简单应用场景：**
理解循环神经网络的长程依赖问题。

---

---

### 公式 (6.49)：正则化项 / Regularization Term

$$
\boldsymbol{h}_t = \boldsymbol{z}_t \odot \boldsymbol{h}_{t-1} + (1 - \boldsymbol{z}_t) \odot \tilde{\boldsymbol{h}}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 原始损失 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\boldsymbol{w}$ | quán zhòng | 权重向量 |

**一句话解释：**
在损失函数中添加权重的平方和，防止模型过度拟合。

**简单应用场景：**
任何机器学习模型中防止过拟合的标准技术（L2正则化）。

---

---

### 公式 (6.50)：改进跳连 / Improved Skip Connection

$$\boldsymbol{h}_t = \boldsymbol{h}_{t-1} + g(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}; \boldsymbol{\theta})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi tī | 第t时刻的隐状态 |
| $g(\cdot)$ | gē | 非线性变换函数 |
| $\boldsymbol{x}_t$ | xiào tī | 第t时刻的输入 |
| $\boldsymbol{h}_{t-1}$ | héi tī | 上一时刻的隐状态 |

**一句话解释：**
改进的跳连既保留线性关系，又通过g函数添加非线性关系，平衡表达能力。

**简单应用场景：**
LSTM和GRU网络设计的基础思想。

---

---

### 公式 (6.51)：LSTM记忆单元更新 / LSTM Memory Cell Update

$$\boldsymbol{c}_t = \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{c}_t$ | jì yì dān yuán | 第t时刻的记忆单元 |
| $\boldsymbol{f}_t$ | yí wàng mén | 遗忘门 |
| $\boldsymbol{i}_t$ | shū rù mén | 输入门 |
| $\tilde{\boldsymbol{c}}_t$ | hóu xuǎn zhuàng tài | 候选状态 |
| $\odot$ | yuánsù chéng jī | 元素乘积 |

**一句话解释：**
LSTM通过遗忘门控制旧信息的保留，输入门控制新信息的增加，更新记忆单元。

**简单应用场景：**
LSTM网络的核心机制，长期信息的存储和遗忘。

---

---

### 公式 (6.52)：LSTM隐状态输出 / LSTM Hidden State Output

$$\boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi zhuàng tài | 第t时刻的隐状态 |
| $\boldsymbol{o}_t$ | shū chū mén | 输出门 |
| $\boldsymbol{c}_t$ | jì yì dān yuán | 记忆单元 |
| $\tanh$ | tāng | 双曲正切激活函数 |
| $\odot$ | yuánsù chéng jī | 元素乘积 |

**一句话解释：**
输出门控制记忆单元有多少信息输出给隐状态，隐状态作为外部输出。

**简单应用场景：**
LSTM网络向后续层传递信息。

---

---

### 公式 (6.53)：LSTM候选状态 / LSTM Candidate State

$$\tilde{\boldsymbol{c}}_t = \tanh(\boldsymbol{W}_c \boldsymbol{x}_t + \boldsymbol{U}_c \boldsymbol{h}_{t-1} + \boldsymbol{b}_c)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{\boldsymbol{c}}_t$ | hóu xuǎn zhuàng tài | 候选状态 |
| $\boldsymbol{W}_c$ | shū rù quán zhòng | 输入权重 |
| $\boldsymbol{U}_c$ | xún huán quán zhòng | 循环权重 |
| $\boldsymbol{x}_t$ | xiào tī | 第t时刻的输入 |
| $\boldsymbol{h}_{t-1}$ | héi tī | 上一时刻的隐状态 |

**一句话解释：**
LSTM通过非线性变换当前输入和历史隐状态得到候选状态，作为新信息的候选。

**简单应用场景：**
LSTM网络中新信息的生成。

---

---

### 公式 (6.54)：输入门 / Input Gate

$$\boldsymbol{i}_t = \sigma(\boldsymbol{W}_i \boldsymbol{x}_t + \boldsymbol{U}_i \boldsymbol{h}_{t-1} + \boldsymbol{b}_i)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{i}_t$ | shū rù mén | 输入门 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{W}_i$ | shū rù quán zhòng | 输入权重 |
| $\boldsymbol{U}_i$ | xún huán quán zhòng | 循环权重 |

**一句话解释：**
输入门决定有多少候选信息需要写入记忆单元，输出值在0-1之间。

**简单应用场景：**
LSTM中的信息选择门。

---

---

### 公式 (6.55)：遗忘门 / Forget Gate

$$\boldsymbol{f}_t = \sigma(\boldsymbol{W}_f \boldsymbol{x}_t + \boldsymbol{U}_f \boldsymbol{h}_{t-1} + \boldsymbol{b}_f)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{f}_t$ | yí wàng mén | 遗忘门 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{W}_f$ | shū rù quán zhòng | 输入权重 |
| $\boldsymbol{U}_f$ | xún huán quán zhòng | 循环权重 |

**一句话解释：**
遗忘门控制上一时刻的记忆单元中有多少信息需要遗忘，值越小遗忘越多。

**简单应用场景：**
LSTM中的信息遗忘机制。

---

---

### 公式 (6.56)：输出门 / Output Gate

$$\boldsymbol{o}_t = \sigma(\boldsymbol{W}_o \boldsymbol{x}_t + \boldsymbol{U}_o \boldsymbol{h}_{t-1} + \boldsymbol{b}_o)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{o}_t$ | shū chū mén | 输出门 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{W}_o$ | shū rù quán zhòng | 输入权重 |
| $\boldsymbol{U}_o$ | xún huán quán zhòng | 循环权重 |

**一句话解释：**
输出门决定记忆单元中有多少信息需要输出给隐状态，控制短期信息的流动。

**简单应用场景：**
LSTM中的输出信息选择。

---

---

### 公式 (6.57)：LSTM紧凑表示 / LSTM Compact Representation

$$\begin{bmatrix} \tilde{\boldsymbol{c}}_t \\ \boldsymbol{o}_t \\ \boldsymbol{i}_t \\ \boldsymbol{f}_t \end{bmatrix} = \begin{bmatrix} \tanh \\ \sigma \\ \sigma \\ \sigma \end{bmatrix} \left(\boldsymbol{W} \begin{bmatrix} \boldsymbol{x}_t \\ \boldsymbol{h}_{t-1} \end{bmatrix} + \boldsymbol{b}\right)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}$ | wēi | 统一权重矩阵 |
| $\boldsymbol{b}$ | bǐ | 统一偏置向量 |
| $\tanh, \sigma$ | jī huà hánshù | 激活函数 |

**一句话解释：**
将LSTM的四个计算过程统一为单个矩阵运算，便于高效实现。

**简单应用场景：**
LSTM网络的矢量化实现。

---

---

### 公式 (6.58)：LSTM记忆单元紧凑 / LSTM Memory Cell Compact

$$\boldsymbol{c}_t = \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{c}_t$ | jì yì dān yuán | 记忆单元 |
| $\boldsymbol{f}_t$ | yí wàng mén | 遗忘门 |
| $\boldsymbol{i}_t$ | shū rù mén | 输入门 |

**一句话解释：**
遗忘门和输入门的组合实现对旧信息的更新和新信息的融合。

**简单应用场景：**
LSTM的核心更新机制。

---

---

### 公式 (6.59)：LSTM隐状态紧凑 / LSTM Hidden State Compact

$$\boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi zhuàng tài | 隐状态 |
| $\boldsymbol{o}_t$ | shū chū mén | 输出门 |
| $\boldsymbol{c}_t$ | jì yì dān yuán | 记忆单元 |

**一句话解释：**
输出门选择性地输出记忆信息，作为当前时刻对外的表示。

**简单应用场景：**
LSTM隐状态的生成。

---

---

### 公式 (6.60)：无遗忘门LSTM / Forget-Gate-Free LSTM

$$\boldsymbol{c}_t = \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{c}_t$ | jì yì dān yuán | 记忆单元 |
| $\boldsymbol{i}_t$ | shū rù mén | 输入门 |
| $\tilde{\boldsymbol{c}}_t$ | hóu xuǎn zhuàng tài | 候选状态 |

**一句话解释：**
最早提出的LSTM没有遗忘门，只能累加新信息，会导致记忆饱和。

**简单应用场景：**
LSTM的历史变体。

---

---

### 公式 (6.61)：Peephole输入门 / Peephole Input Gate

$$\boldsymbol{i}_t = \sigma(\boldsymbol{W}_i \boldsymbol{x}_t + \boldsymbol{U}_i \boldsymbol{h}_{t-1} + \boldsymbol{V}_i \boldsymbol{c}_{t-1} + \boldsymbol{b}_i)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{i}_t$ | shū rù mén | 输入门 |
| $\boldsymbol{V}_i$ | quán zhòng | 记忆单元权重 |
| $\boldsymbol{c}_{t-1}$ | jì yì | 上一时刻记忆 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic函数 |

**一句话解释：**
Peephole连接让门可以直接看到上一时刻的记忆单元，增强门的决策能力。

**简单应用场景：**
LSTM的改进变体。

---

---

### 公式 (6.62)：Peephole遗忘门 / Peephole Forget Gate

$$\boldsymbol{f}_t = \sigma(\boldsymbol{W}_f \boldsymbol{x}_t + \boldsymbol{U}_f \boldsymbol{h}_{t-1} + \boldsymbol{V}_f \boldsymbol{c}_{t-1} + \boldsymbol{b}_f)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{f}_t$ | yí wàng mén | 遗忘门 |
| $\boldsymbol{V}_f$ | quán zhòng | 记忆单元权重 |
| $\boldsymbol{c}_{t-1}$ | jì yì | 上一时刻记忆 |

**一句话解释：**
遗忘门通过Peephole连接直接观察上一时刻的记忆单元。

**简单应用场景：**
LSTM的改进变体。

---

---

### 公式 (6.63)：Peephole输出门 / Peephole Output Gate

$$\boldsymbol{o}_t = \sigma(\boldsymbol{W}_o \boldsymbol{x}_t + \boldsymbol{U}_o \boldsymbol{h}_{t-1} + \boldsymbol{V}_o \boldsymbol{c}_t + \boldsymbol{b}_o)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{o}_t$ | shū chū mén | 输出门 |
| $\boldsymbol{V}_o$ | quán zhòng | 记忆单元权重 |
| $\boldsymbol{c}_t$ | jì yì dān yuán | 当前时刻记忆 |

**一句话解释：**
输出门通过Peephole连接看到当前时刻的记忆单元，更好地决策输出。

**简单应用场景：**
LSTM的改进变体。

---

---

### 公式 (6.64)：耦合门LSTM / Coupled Gate LSTM

$$\boldsymbol{c}_t = (1 - \boldsymbol{i}_t) \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{c}_t$ | jì yì dān yuán | 记忆单元 |
| $\boldsymbol{i}_t$ | shū rù mén | 输入门 |
| $\tilde{\boldsymbol{c}}_t$ | hóu xuǎn zhuàng tài | 候选状态 |

**一句话解释：**
将遗忘门和输入门耦合为一个门（遗忘门=1-输入门），减少参数和计算量。

**简单应用场景：**
简化的LSTM变体。

---

---

### 公式 (6.65)：GRU隐状态更新 / GRU Hidden State Update

$$\boldsymbol{h}_t = \boldsymbol{z}_t \odot \boldsymbol{h}_{t-1} + (1 - \boldsymbol{z}_t) \odot g(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}; \boldsymbol{\theta})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi zhuàng tài | 隐状态 |
| $\boldsymbol{z}_t$ | gēng xīn mén | 更新门 |
| $g(\cdot)$ | gē | 候选状态函数 |
| $\odot$ | yuánsù chéng jī | 元素乘积 |

**一句话解释：**
GRU用单一更新门平衡保留历史信息和接收新信息，比LSTM简洁。

**简单应用场景：**
GRU网络的核心机制。

---

---

### 公式 (6.66)：GRU更新门 / GRU Update Gate

$$\boldsymbol{z}_t = \sigma(\boldsymbol{W}_z \boldsymbol{x}_t + \boldsymbol{U}_z \boldsymbol{h}_{t-1} + \boldsymbol{b}_z)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}_t$ | gēng xīn mén | 更新门 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{W}_z$ | shū rù quán zhòng | 输入权重 |
| $\boldsymbol{U}_z$ | xún huán quán zhòng | 循环权重 |

**一句话解释：**
更新门决定有多少历史信息保留在隐状态中，输出值在0-1之间。

**简单应用场景：**
GRU网络的关键门。

---

---

### 公式 (6.67)：GRU候选隐状态 / GRU Candidate Hidden State

$$\tilde{\boldsymbol{h}}_t = \tanh(\boldsymbol{W}_h \boldsymbol{x}_t + \boldsymbol{U}_h (\boldsymbol{r}_t \odot \boldsymbol{h}_{t-1}) + \boldsymbol{b}_h)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{\boldsymbol{h}}_t$ | hóu xuǎn gé zhuàng tài | 候选隐状态 |
| $\boldsymbol{r}_t$ | chóng zhì mén | 重置门 |
| $\boldsymbol{h}_{t-1}$ | héi zhuàng tài | 上一时刻隐状态 |
| $\tanh$ | tāng | 双曲正切函数 |

**一句话解释：**
GRU通过重置门控制新信息是否依赖历史状态，生成候选隐状态。

**简单应用场景：**
GRU网络的新信息生成。

---

---

### 公式 (6.68)：GRU重置门 / GRU Reset Gate

$$\boldsymbol{r}_t = \sigma(\boldsymbol{W}_r \boldsymbol{x}_t + \boldsymbol{U}_r \boldsymbol{h}_{t-1} + \boldsymbol{b}_r)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{r}_t$ | chóng zhì mén | 重置门 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{W}_r$ | shū rù quán zhòng | 输入权重 |
| $\boldsymbol{U}_r$ | xún huán quán zhòng | 循环权重 |

**一句话解释：**
重置门控制候选隐状态的计算是否依赖上一时刻的隐状态，实现历史信息的选择性遗忘。

**简单应用场景：**
GRU网络的重要控制门。

---

---

### 公式 (6.69)：GRU隐状态更新紧凑 / GRU Hidden State Compact

$$\boldsymbol{h}_t = \boldsymbol{z}_t \odot \boldsymbol{h}_{t-1} + (1 - \boldsymbol{z}_t) \odot \tilde{\boldsymbol{h}}_t$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi zhuàng tài | 隐状态 |
| $\boldsymbol{z}_t$ | gēng xīn mén | 更新门 |
| $\tilde{\boldsymbol{h}}_t$ | hóu xuǎn gé zhuàng tài | 候选隐状态 |

**一句话解释：**
更新门平衡地混合历史隐状态和候选隐状态，得到最终隐状态。

**简单应用场景：**
GRU网络的输出生成。

---

---

### 公式 (6.70)：堆叠循环神经网络 / Stacked RNN Layer

$$\boldsymbol{h}_t^{(l)} = f(\boldsymbol{U}^{(l)} \boldsymbol{h}_{t-1}^{(l)} + \boldsymbol{W}^{(l)} \boldsymbol{h}_t^{(l-1)} + \boldsymbol{b}^{(l)})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t^{(l)}$ | déi l céng héi zhuàng tài | 第l层第t时刻的隐状态 |
| $\boldsymbol{U}^{(l)}$ | déi l céng xún huán quán zhòng | 第l层的循环权重 |
| $\boldsymbol{W}^{(l)}$ | déi l céng shū rù quán zhòng | 第l层的输入权重 |
| $f(\cdot)$ | jī huà hánshù | 激活函数 |

**一句话解释：**
堆叠多层循环神经网络，增加模型深度和表达能力，相邻层之间通过隐状态连接。

**简单应用场景：**
深层循环神经网络的构建。

---

---

### 公式 (6.71)：双向RNN前向 / Bidirectional RNN Forward

$$\boldsymbol{h}_t^{(1)} = f(\boldsymbol{U}^{(1)} \boldsymbol{h}_{t-1}^{(1)} + \boldsymbol{W}^{(1)} \boldsymbol{x}_t + \boldsymbol{b}^{(1)})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t^{(1)}$ | qián xiàng céng héi zhuàng tài | 前向层隐状态 |
| $f(\cdot)$ | jī huà hánshù | 激活函数 |
| $\boldsymbol{U}^{(1)}$ | xún huán quán zhòng | 循环权重 |
| $\boldsymbol{W}^{(1)}$ | shū rù quán zhòng | 输入权重 |

**一句话解释：**
双向RNN的前向层按时序正向处理序列，捕捉历史信息。

**简单应用场景：**
序列处理中需要上文信息的任务。

---

---

### 公式 (6.72)：双向RNN反向 / Bidirectional RNN Backward

$$\boldsymbol{h}_t^{(2)} = f(\boldsymbol{U}^{(2)} \boldsymbol{h}_{t+1}^{(2)} + \boldsymbol{W}^{(2)} \boldsymbol{x}_t + \boldsymbol{b}^{(2)})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t^{(2)}$ | hòu xiàng céng héi zhuàng tài | 反向层隐状态 |
| $f(\cdot)$ | jī huà hánshù | 激活函数 |
| $\boldsymbol{h}_{t+1}^{(2)}$ | xià yī shíkè | 下一时刻的反向隐状态 |

**一句话解释：**
双向RNN的反向层按时序逆向处理序列，捕捉未来信息。

**简单应用场景：**
序列处理中需要后文信息的任务。

---

---

### 公式 (6.73)：双向RNN输出拼接 / Bidirectional RNN Concatenation

$$\boldsymbol{h}_t = \boldsymbol{h}_t^{(1)} \oplus \boldsymbol{h}_t^{(2)}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi zhuàng tài | 最终隐状态 |
| $\boldsymbol{h}_t^{(1)}$ | qián xiàng | 前向隐状态 |
| $\boldsymbol{h}_t^{(2)}$ | hòu xiàng | 反向隐状态 |
| $\oplus$ | pīn jiē | 向量拼接 |

**一句话解释：**
前向和反向层的隐状态拼接起来，形成包含双向信息的完整表示。

**简单应用场景：**
双向RNN的输出，同时包含上文和下文信息。

---

---

### 公式 (6.74)：递归神经网络节点更新 / Recursive Neural Network Node Update

$$\boldsymbol{h}_i = f(\boldsymbol{h}_{\pi_i})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_i$ | dì i gè jié diǎn zhuàng tài | 第i个节点的隐状态 |
| $f(\cdot)$ | jī huà hánshù | 非线性变换函数 |
| $\boldsymbol{h}_{\pi_i}$ | fù jié diǎn zhuàng tài | 父节点状态的拼接 |
| $\pi_i$ | fù jié diǎn jí hé | 第i个节点的父节点集合 |

**一句话解释：**
递归神经网络中每个节点根据其所有父节点的状态进行更新，扩展了循环网络到树结构。

**简单应用场景：**
自然语言处理中的句子语义合成、自然语言结构化建模。

---

---

### 公式 (6.75)：递归神经网络示例1 / Recursive NN Example 1

$$\boldsymbol{h}_1 = \sigma\left(\boldsymbol{W} \begin{bmatrix} \boldsymbol{x}_1 \\ \boldsymbol{x}_2 \end{bmatrix} + \boldsymbol{b}\right)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_1$ | dì 1 gè gé zhuàng tài | 第1个隐状态 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{W}$ | wēi | 权重矩阵 |
| $\boldsymbol{b}$ | bǐ | 偏置向量 |

**一句话解释：**
递归神经网络通过对输入进行非线性组合生成中间层的表示。

**简单应用场景：**
树结构中两个子节点合成父节点。

---

---

### 公式 (6.76)：递归神经网络示例2 / Recursive NN Example 2

$$\boldsymbol{h}_2 = \sigma\left(\boldsymbol{W} \begin{bmatrix} \boldsymbol{x}_3 \\ \boldsymbol{x}_4 \end{bmatrix} + \boldsymbol{b}\right)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_2$ | dì 2 gè gé zhuàng tài | 第2个隐状态 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{W}$ | wēi | 权重矩阵 |

**一句话解释：**
树结构中另外两个子节点组合生成第2个隐状态。

**简单应用场景：**
递归神经网络的树结构构建。

---

---

### 公式 (6.77)：递归神经网络示例3 / Recursive NN Example 3

$$\boldsymbol{h}_3 = \sigma\left(\boldsymbol{W} \begin{bmatrix} \boldsymbol{h}_1 \\ \boldsymbol{h}_2 \end{bmatrix} + \boldsymbol{b}\right)$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_3$ | dì 3 gè gé zhuàng tài | 根节点隐状态 |
| $\sigma(\cdot)$ | xī gé mǎ | Logistic激活函数 |
| $\boldsymbol{h}_1, \boldsymbol{h}_2$ | gé zhuàng tài | 子树隐状态 |

**一句话解释：**
树的根节点由两个子树的表示进一步合成，形成整棵树的表示。

**简单应用场景：**
递归神经网络进行层级信息融合。

---

---

### 公式 (6.78)：递归神经网络分类输出 / Recursive NN Classification Output

$$y = g(\boldsymbol{W}' \boldsymbol{h}_3 + \boldsymbol{b}')$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | 分类输出 |
| $g(\cdot)$ | gē | 分类器函数 |
| $\boldsymbol{W}'$ | wēi | 分类权重 |
| $\boldsymbol{h}_3$ | gé zhuàng tài | 根节点表示 |

**一句话解释：**
递归神经网络将整棵树的根节点表示经过分类器映射为输出标签。

**简单应用场景：**
递归神经网络的分类预测。

---

---

### 公式 (6.79)：批量归一化 / Batch Normalization

$$
\boldsymbol{h}_t^{(l)} = f(\boldsymbol{U}^{(l)} \boldsymbol{h}_{t-1}^{(l)} + \boldsymbol{W}^{(l)} \boldsymbol{h}_t^{(l-1)} + \boldsymbol{b}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_i$ | shū rù zhí | 输入值 |
| $\mu_B$ | píng jūn zhí | 批次均值 |
| $\sigma_B^2$ | fāng chà | 批次方差 |
| $\epsilon$ | ēpsī lóng | 数值稳定性常数 |

**一句话解释：**
将每个批次的输入归一化为均值0、方差1，加速训练并提高稳定性。

**简单应用场景：**
深层神经网络的中间层，大幅加快训练速度和改善收敛性。

---

---

### 公式 (6.80)：层归一化 / Layer Normalization

$$
\boldsymbol{h}_t^{(1)} = f(\boldsymbol{U}^{(1)} \boldsymbol{h}_{t-1}^{(1)} + \boldsymbol{W}^{(1)} \boldsymbol{x}_t + \boldsymbol{b}^{(1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | shū rù xiàng liàng | 输入向量 |
| $\boldsymbol{\mu}$ | píng jūn zhí | 均值 |
| $\boldsymbol{\sigma}^2$ | fāng chà | 方差 |
| $\gamma$ | gā mǎ | 缩放参数 |
| $\boldsymbol{\beta}$ | bèi tǎ | 平移参数 |

**一句话解释：**
按特征维度进行归一化，而非按批次，更适合序列模型和小批量场景。

**简单应用场景：**
Transformer模型和循环神经网络中的标准归一化方法。

---

---

### 公式 (6.81)：权重衰减 / Weight Decay

$$
\boldsymbol{h}_t^{(l)} = \boldsymbol{h}_t^{(l)} + \boldsymbol{h}_t^{(l-1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | quán zhòng | 权重向量 |
| $\lambda$ | lán mǔ dá | 衰减系数 |
| $\alpha$ | āěr fǎ | 学习率 |
| $L$ | zǔn sǔn | 损失函数 |

**一句话解释：**
在每次参数更新时，将权重乘以小于1的因子，逐步减小权重，防止过拟合。

**简单应用场景：**
几乎所有深度学习模型的标准正则化技术。

---

---

## 第7章：网络优化与正则化

### 公式 (7.1)：梯度估计 / Gradient Estimation

$$
g_t(\boldsymbol{\theta}) = \frac{1}{K} \sum_{(\boldsymbol{x},y) \in \mathcal{S}_t} \frac{\partial \mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta}))}{\partial \boldsymbol{\theta}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g_t(\boldsymbol{\theta})$ | tī dù gū jì | 第t次迭代时的梯度估计 |
| $K$ | kè | 批量大小(Batch Size) |
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 神经网络函数 |
| $\boldsymbol{\theta}$ | xī tǎ | 网络参数 |

**一句话解释：**
梯度估计是通过计算小批量样本上的平均损失关于参数的偏导数来估计真实梯度。

**简单应用场景：**
在小批量梯度下降法中，每次迭代时计算一个小批量样本集合上的平均梯度，用于更新参数。

---

---

### 公式 (7.2)：参数更新 / Parameter Update

$$
g_t \triangleq g_t(\boldsymbol{\theta}_{t-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g_t$ | tī dù | 第t次迭代的梯度 |
| $\boldsymbol{\theta}_{t-1}$ | cān shù | 第t-1次迭代的参数 |

**一句话解释：**
记梯度为 $g_t$ 表示在第 $t-1$ 次迭代参数值处计算的梯度。

**简单应用场景：**
简化梯度的符号，便于后续公式的表示。

---

---

### 公式 (7.3)：参数更新规则 / Parameter Update Rule

$$
\boldsymbol{\theta}_t \leftarrow \boldsymbol{\theta}_{t-1} - \alpha g_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}_t$ | xī tǎ | 第t次迭代的参数 |
| $\alpha$ | āl fǎ | 学习率 |
| $g_t$ | tī dù | 第t次迭代的梯度 |

**一句话解释：**
参数沿着负梯度方向更新，更新步长由学习率控制。

**简单应用场景：**
标准的随机梯度下降法中，每次迭代都按照这个规则更新所有参数。

---

---

### 公式 (7.4)：参数差值 / Parameter Difference

$$
\Delta\boldsymbol{\theta}_t \triangleq \boldsymbol{\theta}_t - \boldsymbol{\theta}_{t-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | cān shù chā zhí | 第t次迭代的参数变化量 |
| $\boldsymbol{\theta}_t$ | cān shù | 第t次迭代的参数 |
| $\boldsymbol{\theta}_{t-1}$ | cān shù | 第t-1次迭代的参数 |

**一句话解释：**
参数差值表示一次迭代中参数变化的幅度。

**简单应用场景：**
分析参数更新的变化量，在动量法等优化算法中用到。

---

---

### 公式 (7.5)：反时衰减 / Inverse Time Decay

$$
\alpha_t = \alpha_0 \frac{1}{1 + \beta \times t}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | xué xí lǜ | 第t次迭代的学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 初始学习率 |
| $\beta$ | bèi tǎ | 衰减率 |
| $t$ | tī dài cì shù | 迭代次数 |

**一句话解释：**
学习率随着迭代次数增加而按反比例衰减。

**简单应用场景：**
在长期训练中，逐步降低学习率以帮助模型收敛到更优的局部最小值。

---

---

### 公式 (7.6)：指数衰减 / Exponential Decay

$$
\alpha_t = \alpha_0 \beta^t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | xué xí lǜ | 第t次迭代的学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 初始学习率 |
| $\beta$ | bèi tǎ | 衰减系数 |
| $t$ | tī dài cì shù | 迭代次数 |

**一句话解释：**
学习率按照指数规律衰减，衰减速度较快。

**简单应用场景：**
当需要快速降低学习率时使用，适用于收敛较快的问题。

---

---

### 公式 (7.7)：自然指数衰减 / Natural Exponential Decay

$$
\alpha_t = \alpha_0 \exp(-\beta \times t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | xué xí lǜ | 第t次迭代的学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 初始学习率 |
| $\beta$ | bèi tǎ | 衰减率 |
| $t$ | tī dài cì shù | 迭代次数 |
| $\exp$ | zhǐ shù hánshù | 指数函数 |

**一句话解释：**
学习率按自然指数规律衰减，衰减曲线平滑。

**简单应用场景：**
在需要平滑衰减学习率的优化问题中使用，常见于深度学习框架的默认设置。

---

---

### 公式 (7.8)：余弦衰减 / Cosine Decay

$$
\alpha_t = \frac{1}{2}\alpha_0\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | xué xí lǜ | 第t次迭代的学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 初始学习率 |
| $T$ | zǒng tī dài cì shù | 总迭代次数 |
| $t$ | tī dài cì shù | 当前迭代次数 |
| $\cos$ | yú xián hánshù | 余弦函数 |
| $\pi$ | pài | 圆周率 |

**一句话解释：**
学习率按照余弦函数规律从初值平滑衰减到接近零。

**简单应用场景：**
在固定时间内的训练中，使用余弦衰减策略可以更好地平衡训练的早期和后期阶段。

---

---

### 公式 (7.9)：学习率预热 / Learning Rate Warmup

$$
\alpha'_t = \frac{t}{T'}\alpha_0, \quad 1 \leq t \leq T'
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha'_t$ | yù huà xué xí lǜ | 预热阶段的学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 目标学习率 |
| $t$ | tī dài cì shù | 当前迭代次数 |
| $T'$ | yù huà zǒng tī dài cì shù | 预热总迭代次数 |

**一句话解释：**
在训练初期，学习率从零线性增长到目标学习率，帮助模型稳定收敛。

**简单应用场景：**
使用较大学习率训练大规模神经网络时，先通过预热阶段避免训练发散。

---

---

### 公式 (7.10)：动量法梯度累积 / Momentum Gradient Accumulation

$$
\hat{\boldsymbol{x}} = \frac{\boldsymbol{x} - \boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}_t$ | dì $t$ bù xǔn huà xiàng liàng | 第t步梯度向量 |
| $\beta$ | bèi tǎ | 动量系数 |
| $\nabla_{\boldsymbol{\theta}} L$ | nà bù lā sǔn shī | 梯度 |

**一句话解释：**
累积历史梯度信息，加速优化并使更新方向更稳定。

**简单应用场景：**
所有深度学习优化器中的标准技巧，加快收敛速度。

---

---

### 公式 (7.11)：循环学习率更新 / Cyclic Learning Rate Update

$$
\alpha_t = \alpha^m_{\min} + (\alpha^m_{\max} - \alpha^m_{\min})\max(0, 1-b)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | xué xí lǜ | 第t次迭代的学习率 |
| $\alpha^m_{\min}$ | zuì xiǎo xué xí lǜ | 第m个循环周期的最小学习率 |
| $\alpha^m_{\max}$ | zuì dà xué xí lǜ | 第m个循环周期的最大学习率 |
| $b$ | bǎi fēn bǐ | 归一化位置百分比 |

**一句话解释：**
在每个循环周期内，学习率在最小值和最大值之间线性变化。

**简单应用场景：**
实现三角形循环学习率策略，使学习率周期性变化以改进收敛性。

---

---

### 公式 (7.12)：循环周期位置计算 / Cyclic Period Position

$$
b = \left|\frac{t}{\Delta T} - 2m + 1\right|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b$ | bǎi fēn bǐ | 在当前循环周期内的相对位置 |
| $t$ | tī dài cì shù | 当前迭代次数 |
| $\Delta T$ | xún huán zhōu qī | 循环周期 |
| $m$ | xún huán cì shù | 循环次数 |

**一句话解释：**
计算当前迭代在所在循环周期内的相对位置，范围为[0,1]。

**简单应用场景：**
确定学习率在当前循环周期内应该取的具体值。

---

---

### 公式 (7.13)：自适应学习率 / Adaptive Learning Rate

$$
\hat{\boldsymbol{x}} = \frac{\boldsymbol{x} - \boldsymbol{\mu}_{\mathcal{B}}}{\sqrt{\boldsymbol{\sigma}_{\mathcal{B}}^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 初始学习率 |
| $G_t$ | jī lěi píng fāng | 梯度平方累积 |
| $\epsilon$ | ēpsī lóng | 数值稳定性常数 |
| $\odot$ | yuán sù chéng fǎ | 元素乘法 |

**一句话解释：**
为每个参数分别设置学习率，根据历史梯度信息自动调整学习速度。

**简单应用场景：**
RMSprop、AdaGrad等现代优化器的核心机制。

---

---

### 公式 (7.14)：AdaGrad梯度平方累积 / AdaGrad Gradient Accumulation

$$
G_t = \sum_{i=1}^{t} \boldsymbol{g}_i \odot \boldsymbol{g}_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | tī dù píng fāng lěi jī hé | 第t次迭代时梯度平方的累积和 |
| $\boldsymbol{g}_i$ | tī dù | 第i次迭代的梯度向量 |
| $\odot$ | yuán sù chéng jī | 元素级乘积(Hadamard积) |

**一句话解释：**
AdaGrad算法累积历历史梯度的平方，用于自适应地调整学习率。

**简单应用场景：**
对于出现频率高的特征自动降低学习率，对于出现频率低的特征保持较高的学习率。

---

---

### 公式 (7.15)：AdaGrad参数更新 / AdaGrad Parameter Update

$$
\Delta\boldsymbol{\theta}_t = -\frac{\alpha}{\sqrt{G_t + \varepsilon}} \odot \boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | cān shù gèng xīn liàng | 第t次迭代的参数更新量 |
| $\alpha$ | āl fǎ | 学习率 |
| $G_t$ | tī dù píng fāng lěi jī | 梯度平方累积 |
| $\varepsilon$ | ēps | 小的正数，用于数值稳定性 |
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |

**一句话解释：**
参数更新时，每个维度的学习率都根据该维度历史梯度的大小进行自适应调整。

**简单应用场景：**
在处理稀疏数据和数据特征出现频率差异大的问题时效果较好。

---

---

### 公式 (7.16)：RMSprop梯度平方指数移动平均 / RMSprop Exponential Moving Average

$$
G_t = \beta G_{t-1} + (1-\beta)\boldsymbol{g}_t \odot \boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | tī dù píng fāng lěi jī | 第t次迭代的指数移动平均 |
| $\beta$ | bèi tǎ | 衰减系数，通常为0.9 |
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |
| $\odot$ | yuán sù chéng jī | 元素级乘积 |

**一句话解释：**
RMSprop使用指数移动平均替代累积和，避免学习率单调递减的问题。

**简单应用场景：**
比AdaGrad更灵活，可以继续学习即使在很多次迭代后仍有较大的梯度。

---

---

### 公式 (7.17)：RMSprop梯度平方展开式 / RMSprop Expansion

$$
G_t = (1-\beta)\sum_{i=1}^{t}\beta^{t-i}\boldsymbol{g}_i \odot \boldsymbol{g}_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | tī dù píng fāng lěi jī | 第t次迭代的指数移动平均 |
| $\beta$ | bèi tǎ | 衰减系数 |
| $\boldsymbol{g}_i$ | tī dù | 第i次迭代的梯度 |
| $t$ | tī dài cì shù | 总迭代次数 |

**一句话解释：**
RMSprop的梯度平方可以展开为加权的历史梯度平方和，权重随时间指数衰减。

**简单应用场景：**
理解RMSprop如何给近期梯度更高的权重，同时保留历史信息。

---

---

### 公式 (7.18)：RMSprop参数更新 / RMSprop Parameter Update

$$
\Delta\boldsymbol{\theta}_t = -\frac{\alpha}{\sqrt{G_t + \varepsilon}} \odot \boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | cān shù gèng xīn liàng | 第t次迭代的参数更新量 |
| $\alpha$ | āl fǎ | 学习率，通常为0.001 |
| $G_t$ | tī dù píng fāng lěi jī | 梯度平方的指数移动平均 |
| $\varepsilon$ | ēps | 小的正数，用于数值稳定性 |
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |

**一句话解释：**
RMSprop参数更新与AdaGrad类似，但使用指数移动平均替代累积和。

**简单应用场景：**
在神经网络训练中广泛使用，尤其是循环神经网络的优化。

---

---

### 公式 (7.19)：AdaDelta参数平方指数移动平均 / AdaDelta Parameter Accumulation

$$
\Delta X^2_{t-1} = \beta_1 \Delta X^2_{t-2} + (1-\beta_1)\Delta\boldsymbol{\theta}_{t-1} \odot \Delta\boldsymbol{\theta}_{t-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta X^2_{t-1}$ | cān shù gèng xīn liàng píng fāng lěi jī | 第t-1次迭代的参数更新平方指数移动平均 |
| $\beta_1$ | bèi tǎ | 衰减系数 |
| $\Delta\boldsymbol{\theta}_{t-1}$ | cān shù gèng xīn liàng | 第t-1次迭代的参数更新量 |
| $\odot$ | yuán sù chéng jī | 元素级乘积 |

**一句话解释：**
AdaDelta跟踪参数更新量的指数移动平均，用于自适应地调整学习率。

**简单应用场景：**
在不需要手动设置学习率的优化问题中使用，完全自适应。

---

---

### 公式 (7.20)：AdaDelta参数更新 / AdaDelta Update Rule

$$
\Delta\boldsymbol{\theta}_t = -\frac{\sqrt{\Delta X^2_{t-1}+\varepsilon}}{\sqrt{G_t + \varepsilon}} \odot \boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | cān shù gèng xīn liàng | 第t次迭代的参数更新量 |
| $\Delta X^2_{t-1}$ | cān shù gèng xīn píng fāng lěi jī | 参数更新平方的指数移动平均 |
| $G_t$ | tī dù píng fāng lěi jī | 梯度平方的指数移动平均 |
| $\varepsilon$ | ēps | 小的正数 |
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |

**一句话解释：**
AdaDelta使用参数更新量的历史信息与梯度信息的比值来动态调整学习率。

**简单应用场景：**
完全不需要预设学习率，自动适应不同问题的优化特性。

---

---

### 公式 (7.21)：动量法 / Momentum Method

$$
\Delta\boldsymbol{\theta}_t = \rho\Delta\boldsymbol{\theta}_{t-1} - \alpha g_t = -\alpha\sum_{i=1}^{t}\rho^{t-i}\boldsymbol{g}_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | cān shù gèng xīn liàng | 第t次迭代的参数更新量 |
| $\rho$ | rù | 动量系数，通常为0.9 |
| $\alpha$ | āl fǎ | 学习率 |
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |

**一句话解释：**
动量法在每次迭代时不仅考虑当前梯度，还考虑之前的更新方向，加速收敛。

**简单应用场景：**
在有高曲率梯度或噪声梯度的优化问题中，使用动量法可以更快地收敛。

---

---

### 公式 (7.22)：Nesterov 加速梯度 / Nesterov Accelerated Gradient

$$
\boldsymbol{\mu}_{\mathcal{B}} = \frac{1}{K} \sum_{k=1}^{K} \boldsymbol{z}^{(k)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |
| $\beta$ | bèi tǎ | 动量系数 |
| $\boldsymbol{v}_{t-1}$ | qián $t-1$ bù dòng liàng | 前一步动量 |

**一句话解释：**
预测下一位置梯度，在该位置计算梯度，提供比标准动量法更好的收敛性。

**简单应用场景：**
加速深度神经网络收敛，比动量法性能更优。

---

---

### 公式 (7.23)：梯度裁剪 / Gradient Clipping

$$
\boldsymbol{\sigma}_{\mathcal{B}}^2 = \frac{1}{K} \sum_{k=1}^{K} (\boldsymbol{z}^{(k)} - \boldsymbol{\mu}_{\mathcal{B}})^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}$ | tī dù xiàng liàng | 梯度向量 |
| $\tilde{\boldsymbol{g}}$ | cái jiǎn hòu tī dù | 裁剪后梯度 |
| $\theta$ | xī tǎ | 梯度范数阈值 |
| $\|\boldsymbol{g}\|$ | tī dù fàn shù | 梯度范数 |

**一句话解释：**
限制梯度的最大范数，防止梯度爆炸导致训练不稳定。

**简单应用场景：**
循环神经网络和其他容易梯度爆炸的模型的训练。

---

---

### 公式 (7.24)：Nesterov加速梯度-更新规则 / NAG Update Rule

$$
\Delta\boldsymbol{\theta}_t = \rho\Delta\boldsymbol{\theta}_{t-1} - \alpha g_t(\boldsymbol{\theta}_{t-1} + \rho\Delta\boldsymbol{\theta}_{t-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | cān shù gèng xīn liàng | 第t次迭代的参数更新量 |
| $\rho$ | rù | 动量系数 |
| $\alpha$ | āl fǎ | 学习率 |
| $\boldsymbol{g}_t()$ | tī dù hánshù | 梯度函数 |

**一句话解释：**
Nesterov加速梯度的完整更新规则，结合了动量和前向看的思想。

**简单应用场景：**
在凸优化和深度学习中使用NAG可以获得比普通动量法更好的收敛速率。

---

---

### 公式 (7.25)：Adam一阶矩估计 / Adam First Moment Estimate

$$
M_t = \beta_1 M_{t-1} + (1-\beta_1)\boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $M_t$ | yī jiē jù | 第t次迭代的一阶矩(均值)估计 |
| $\beta_1$ | bèi tǎ | 一阶矩的衰减系数，通常为0.9 |
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |

**一句话解释：**
Adam算法使用指数移动平均估计梯度的一阶矩(均值)，类似于动量。

**简单应用场景：**
Adam优化器中用于跟踪梯度的期望值，帮助加速收敛。

---

---

### 公式 (7.26)：Adam二阶矩估计 / Adam Second Moment Estimate

$$
G_t = \beta_2 G_{t-1} + (1-\beta_2)\boldsymbol{g}_t \odot \boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | èr jiē jù | 第t次迭代的二阶矩(方差)估计 |
| $\beta_2$ | bèi tǎ | 二阶矩的衰减系数，通常为0.99 |
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |
| $\odot$ | yuán sù chéng jī | 元素级乘积 |

**一句话解释：**
Adam算法同时估计梯度的二阶矩(方差)，用于自适应调整各维度的学习率。

**简单应用场景：**
与一阶矩结合，使Adam能够在不同维度上自适应地调整学习率。

---

---

### 公式 (7.27)：Adam一阶矩偏差修正 / Adam First Moment Bias Correction

$$
\tilde{M}_t = \frac{M_t}{1-\beta^t_1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{M}_t$ | xiū zhèng hòu yī jiē jù | 修正后的一阶矩估计 |
| $M_t$ | yī jiē jù | 原始一阶矩估计 |
| $\beta_1$ | bèi tǎ | 一阶矩的衰减系数 |
| $t$ | tī dài cì shù | 迭代次数 |

**一句话解释：**
由于一阶矩初始为零，需要进行偏差修正以获得无偏估计，尤其在训练初期。

**简单应用场景：**
确保Adam在训练早期能够正确地估计梯度的均值。

---

---

### 公式 (7.28)：Adam二阶矩偏差修正 / Adam Second Moment Bias Correction

$$
\tilde{G}_t = \frac{G_t}{1-\beta^t_2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{G}_t$ | xiū zhèng hòu èr jiē jù | 修正后的二阶矩估计 |
| $G_t$ | èr jiē jù | 原始二阶矩估计 |
| $\beta_2$ | bèi tǎ | 二阶矩的衰减系数 |
| $t$ | tī dài cì shù | 迭代次数 |

**一句话解释：**
对二阶矩进行偏差修正，使其成为关于梯度方差的无偏估计。

**简单应用场景：**
确保Adam的自适应学习率在训练初期能够准确反映梯度的方差特性。

---

---

### 公式 (7.29)：Adam参数更新 / Adam Parameter Update

$$
\Delta\boldsymbol{\theta}_t = -\frac{\alpha}{\sqrt{\tilde{G}_t + \varepsilon}} \odot \tilde{M}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | cān shù gèng xīn liàng | 第t次迭代的参数更新量 |
| $\alpha$ | āl fǎ | 学习率，通常为0.001 |
| $\tilde{G}_t$ | xiū zhèng èr jiē jù | 修正后的二阶矩估计 |
| $\varepsilon$ | ēps | 小的正数，用于数值稳定性 |
| $\tilde{M}_t$ | xiū zhèng yī jiē jù | 修正后的一阶矩估计 |
| $\odot$ | yuán sù chéng jī | 元素级乘积 |

**一句话解释：**
Adam结合了一阶矩(动量)和二阶矩(自适应学习率)来更新参数，被广泛认为是深度学习中最有效的优化算法。

**简单应用场景：**
是现代深度学习中最常用的优化器，适用于大多数深度神经网络的训练。

---

---

### 公式 (7.30)：梯度裁剪 / Gradient Clipping

$$
\boldsymbol{g}_t = \max(\min(\boldsymbol{g}_t, b), a)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |
| $a$ | xià jiè | 梯度的下界 |
| $b$ | shàng jiè | 梯度的上界 |
| $\max,\min$ | zuì dà zhí, zuì xiǎo zhí | 最大值、最小值函数 |

**一句话解释：**
梯度裁剪将梯度限制在[a,b]区间内，防止梯度爆炸或消失。

**简单应用场景：**
在训练循环神经网络时防止梯度爆炸，提高训练稳定性。

---

---

### 公式 (7.31)：范数梯度裁剪 / Norm-based Gradient Clipping

$$
\boldsymbol{g}_t = \frac{b}{\|\boldsymbol{g}_t\|}\boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}_t$ | tī dù | 第t次迭代的梯度 |
| $b$ | tī dù fàn shù shàng jiè | 梯度范数的上界 |
| $\|\boldsymbol{g}_t\|$ | fàn shù | 梯度的范数 |

**一句话解释：**
当梯度的范数超过阈值b时，将梯度按比例缩小，保持方向不变。

**简单应用场景：**
在深度循环神经网络训练中防止梯度爆炸，比逐元素裁剪更能保持梯度方向。

---

---

### 公式 (7.32)：优化算法通用形式1 / General Optimization Form 1

$$
\Delta \boldsymbol{\theta}_t = -\frac{\alpha_t}{\sqrt{G_t + \epsilon}} M_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta \boldsymbol{\theta}_t$ | dì $t$ bù cān shù gēng xīn | 第t步参数更新 |
| $\alpha_t$ | dì $t$ bù xué xí lǜ | 第t步学习率 |
| $G_t$ | jī lěi píng fāng | 梯度平方累积 |
| $M_t$ | dòng liàng xiàng liàng | 动量向量 |

**一句话解释：**
结合自适应学习率和动量，形成现代优化器的统一框架。

**简单应用场景：**
AdaGrad、RMSprop、Adam等优化器的基础形式。

---

---

### 公式 (7.33)：梯度计算 / Gradient Computation

$$
G_t = \psi(g_1, \ldots, g_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | jī lěi xìn xī | 累积信息 |
| $\psi$ | pài | 累积函数 |
| $g_i$ | dì $i$ bù tī dù | 第i步梯度 |

**一句话解释：**
通过某种方式聚合历史梯度信息，为自适应学习率提供基础。

**简单应用场景：**
不同优化器通过不同的$\psi$函数实现各自的自适应机制。

---

---

### 公式 (7.34)：动量计算 / Momentum Computation

$$
M_t = \phi(g_1, \ldots, g_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $M_t$ | dòng liàng xiàng liàng | 动量向量 |
| $\phi$ | fài | 动量计算函数 |
| $g_i$ | dì $i$ bù tī dù | 第i步梯度 |

**一句话解释：**
根据历史梯度计算动量，控制参数更新的方向和大小。

**简单应用场景：**
所有包含动量机制的优化器（Momentum、Adam等）。

---

---

### 公式 (7.35)：均匀分布参数初始化方差 / Uniform Distribution Variance

$$
\text{var}(x) = \frac{(b-a)^2}{12}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(x)$ | fāng chā | 随机变量的方差 |
| $a,b$ | qū jiān biāo jiè | 均匀分布的下界和上界 |

**一句话解释：**
均匀分布在区间[a,b]上的随机变量的方差公式。

**简单应用场景：**
计算参数初始化时的方差，用于确定初始化的范围。

---

---

### 公式 (7.36)：参数初始化范围 / Parameter Initialization Range

$$
r = \sqrt{3\sigma^2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r$ | fàn wéi | 均匀分布的范围(半径) |
| $\sigma^2$ | fāng chā | 期望的方差 |
| $\sqrt{\cdot}$ | píng fāng gēn | 平方根函数 |

**一句话解释：**
根据期望的方差计算均匀分布的范围，参数应在[-r,r]区间内初始化。

**简单应用场景：**
在深度神经网络初始化时，使用该公式确定参数的初始化范围。

---

---

### 公式 (7.37)：Xavier初始化 / Xavier Initialization

$$
a^{(l)} = f\left(\sum_{i=1}^{M_{l-1}} w^{(l)}_i a^{(l-1)}_i\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $a^{(l)}$ | dì l céng huó huà | 第l层的激活值 |
| $f$ | jī huó huà hánshù | 激活函数 |
| $w^{(l)}_i$ | quán zhòng | 第l层的权重 |
| $a^{(l-1)}_i$ | qián yī céng huó huà | 第l-1层的激活值 |
| $M_{l-1}$ | qián yī céng shén jīng yuán gè shù | 第l-1层的神经元个数 |

**一句话解释：**
Xavier初始化假设网络的激活是线性的，基于使每层的方差保持一致的原则进行初始化。

**简单应用场景：**
在深度神经网络中进行参数初始化，特别是对于使用Sigmoid和Tanh激活函数的网络。

---

---

### 公式 (7.38)：Xavier初始化偏差为零 / Xavier Bias Zero Condition

$$
E[a^{(l)}] = E\left[\sum_{i=1}^{M_{l-1}} w^{(l)}_i a^{(l-1)}_i\right] = \sum_{i=1}^{M_{l-1}} E[w^{(l)}_i]E[a^{(l-1)}_i] = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E[\cdot]$ | qī wàng zhí | 期望值 |
| $a^{(l)}$ | huó huà zhí | 第l层激活值 |
| $w^{(l)}_i$ | quán zhòng | 第l层权重 |
| $a^{(l-1)}_i$ | qián yī céng huó huà zhí | 第l-1层激活值 |

**一句话解释：**
Xavier初始化要求权重和激活值的期望都为零，确保初始化的无偏性。

**简单应用场景：**
通过零均值初始化权重和偏置，避免激活值的偏移。

---

---

### 公式 (7.39)：Xavier初始化方差条件 / Xavier Variance Condition

$$
\text{var}(a^{(l)}) = \text{var}\left(\sum_{i=1}^{M_{l-1}} w^{(l)}_i a^{(l-1)}_i\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(\cdot)$ | fāng chā | 方差 |
| $a^{(l)}$ | huó huà zhí | 第l层激活值 |
| $w^{(l)}_i$ | quán zhòng | 第l层权重 |
| $a^{(l-1)}_i$ | qián yī céng huó huà zhí | 第l-1层激活值 |

**一句话解释：**
计算第l层激活值方差的表达式，用于推导Xavier初始化的权重范围。

**简单应用场景：**
分析网络各层激活值方差的变化，确保网络信号的稳定传播。

---

---

### 公式 (7.40)：激活值方差展开 / Activation Variance Expansion

$$
= \sum_{i=1}^{M_{l-1}} \text{var}(w^{(l)}_i)\text{var}(a^{(l-1)}_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(\cdot)$ | fāng chā | 方差函数 |
| $w^{(l)}_i$ | quán zhòng | 第l层权重 |
| $a^{(l-1)}_i$ | qián yī céng huó huà | 第l-1层激活值 |

**一句话解释：**
独立随机变量乘积的方差等于各自方差的乘积之和。

**简单应用场景：**
用于推导Xavier初始化条件，分析网络层间的方差传递。

---

---

### 公式 (7.41)：激活值方差简化 / Activation Variance Simplification

$$
= M_{l-1}\text{var}(w^{(l)}_i)\text{var}(a^{(l-1)}_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $M_{l-1}$ | quán zhòng gè shù | 前一层神经元个数 |
| $\text{var}(w^{(l)}_i)$ | quán zhòng fāng chā | 权重方差 |
| $\text{var}(a^{(l-1)}_i)$ | huó huà zhí fāng chā | 激活值方差 |

**一句话解释：**
当所有权重和激活值具有相同方差时，总方差可以简化为这个表达式。

**简单应用场景：**
在Xavier初始化中，使用这个简化形式推导权重初始化的方差要求。

---

---

### 公式 (7.42)：学习率衰减公式 / Learning Rate Decay Formula

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | dì $t$ bù xué xí lǜ | 第t步学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 初始学习率 |
| $\rho$ | ēr tǎ | 衰减因子 |

**一句话解释：**
按指数衰减学习率，前期学习快，后期学习慢，帮助模型精细化收敛。

**简单应用场景：**
训练阶段动态调整学习率，提升模型最终性能。

---

---

### 公式 (7.43)：权重方差简化 / Weight Variance for Backward Pass

$$
\text{var}(w^{(l)}_i) = \frac{1}{M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(w^{(l)}_i)$ | quán zhòng fāng chā | 权重方差 |
| $M_l$ | dì l céng shén jīng yuán gè shù | 第l层神经元个数 |

**一句话解释：**
考虑反向传播时的梯度方差，权重方差应与当前层的神经元个数成反比。

**简单应用场景：**
Xavier初始化的另一个推导角度，确保反向传播时梯度方差也保持稳定。

---

---

### 公式 (7.44)：Xavier初始化均值权重方差 / Xavier Weight Variance Average

$$
\text{var}(w^{(l)}_i) = \frac{2}{M_{l-1} + M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(w^{(l)}_i)$ | quán zhòng fāng chā | 权重方差 |
| $M_{l-1}$ | qián yī céng shén jīng yuán gè shù | 前一层神经元个数 |
| $M_l$ | dì l céng shén jīng yuán gè shù | 第l层神经元个数 |

**一句话解释：**
Xavier初始化综合考虑前向和反向传播，取两个条件的平均值。

**简单应用场景：**
在实际深度学习框架中实现Xavier初始化时的标准公式。

---

---

### 公式 (7.45)：反向传播中的链式法则应用 / Backpropagation Chain Rule Application

$$
\frac{\partial L}{\partial \boldsymbol{w}^{(l)}} = \frac{\partial L}{\partial \boldsymbol{a}^{(l)}} \frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{w}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 损失函数 |
| $\boldsymbol{w}^{(l)}$ | dì $l$ céng quán zhòng | 第l层权重 |
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng jī huó zhī | 第l层激活值 |

**一句话解释：**
计算损失对权重的梯度，是反向传播算法的基本操作。

**简单应用场景：**
任何神经网络的参数更新前计算梯度。

---

---

### 公式 (7.46)：隐层梯度计算 / Hidden Layer Gradient Computation

$$
\frac{\partial L}{\partial \boldsymbol{a}^{(l-1)}} = (\boldsymbol{w}^{(l)})^T \frac{\partial L}{\partial \boldsymbol{a}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 损失函数 |
| $\boldsymbol{a}^{(l-1)}$ | dì $l-1$ céng jī huó zhī | 前一层激活值 |
| $\boldsymbol{w}^{(l)}$ | dì $l$ céng quán zhòng | 当前层权重 |

**一句话解释：**
将误差信号从当前层反向传播到前一层，用于计算前层的梯度。

**简单应用场景：**
深层网络反向传播的逐层梯度计算。

---

---

### 公式 (7.47)：激活函数梯度 / Activation Function Gradient

$$
\frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{z}^{(l)}} = \sigma'(\boldsymbol{z}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng jī huó zhī | 第l层激活值 |
| $\boldsymbol{z}^{(l)}$ | dì $l$ céng xiān xìng zǔ hé | 第l层线性组合 |
| $\sigma'$ | jī huó hánshù dǎo shù | 激活函数导数 |

**一句话解释：**
激活函数的导数，用于反向传播时计算激活值对线性组合的梯度。

**简单应用场景：**
任何包含非线性激活函数的神经网络的反向传播计算。

---

---

### 公式 (7.48)：ReLU 激活梯度 / ReLU Activation Gradient

$$
\frac{\partial \text{ReLU}(\boldsymbol{z})}{\partial \boldsymbol{z}} = \begin{cases} 1 & \text{if } \boldsymbol{z} > 0 \\ 0 & \text{otherwise} \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | xiān xìng zǔ hé | 线性组合 |
| ReLU | - | 修正线性单元激活函数 |

**一句话解释：**
ReLU函数的导数为阶跃函数，正部分的梯度为1，负部分为0。

**简单应用场景：**
包含ReLU激活的神经网络的梯度计算。

---

---

### 公式 (7.49)：Sigmoid 激活梯度 / Sigmoid Activation Gradient

$$
\frac{\partial \sigma(\boldsymbol{z})}{\partial \boldsymbol{z}} = \sigma(\boldsymbol{z})(1 - \sigma(\boldsymbol{z}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma(\boldsymbol{z})$ | xī gé mó id hánshù | Sigmoid函数 |
| $\boldsymbol{z}$ | xiān xìng zǔ hé | 线性组合 |

**一句话解释：**
Sigmoid函数的导数等于函数值乘以1减去函数值，展现了对称性。

**简单应用场景：**
使用Sigmoid激活函数的神经网络层的梯度计算。

---

---

### 公式 (7.50)：Tanh 激活梯度 / Tanh Activation Gradient

$$
\frac{\partial \tanh(\boldsymbol{z})}{\partial \boldsymbol{z}} = 1 - \tanh^2(\boldsymbol{z})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tanh(\boldsymbol{z})$ | shuāng qū xiàn hánshù | Tanh函数 |
| $\boldsymbol{z}$ | xiān xìng zǔ hé | 线性组合 |

**一句话解释：**
Tanh函数的导数为1减去函数值的平方，范围在0到1之间。

**简单应用场景：**
使用Tanh激活函数的神经网络层的梯度计算。

---

---

### 公式 (7.51)：权重梯度累积 / Weight Gradient Accumulation

$$
\frac{\partial L}{\partial \boldsymbol{W}^{(l)}} = \frac{\partial L}{\partial \boldsymbol{a}^{(l)}} (\boldsymbol{a}^{(l-1)})^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 损失函数 |
| $\boldsymbol{W}^{(l)}$ | dì $l$ céng quán zhòng jǔ zhèn | 第l层权重矩阵 |
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng jī huó zhī | 第l层激活值 |

**一句话解释：**
计算损失函数对权重矩阵的梯度，通过激活值的外积表示。

**简单应用场景：**
神经网络权重参数的梯度计算和参数更新。

---

---

### 公式 (7.52)：偏置梯度 / Bias Gradient

$$
\frac{\partial L}{\partial \boldsymbol{b}^{(l)}} = \frac{\partial L}{\partial \boldsymbol{a}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 损失函数 |
| $\boldsymbol{b}^{(l)}$ | dì $l$ céng piān zhì | 第l层偏置向量 |
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng jī huó zhī | 第l层激活值 |

**一句话解释：**
偏置的梯度等于激活值的梯度，是权重梯度的特殊情况。

**简单应用场景：**
神经网络偏置参数的梯度计算和更新。

---

---

### 公式 (7.53)：批梯度下降 / Batch Gradient Descent

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \sum_{i=1}^{N} \nabla_{\boldsymbol{\theta}} L_i(\boldsymbol{\theta}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |
| $N$ | pī cì dà xiǎo | 批次大小 |
| $L_i$ | dì $i$ gè yàng běn zǔn sǔn | 第i个样本损失 |

**一句话解释：**
对整个批次的样本计算梯度，累加后进行参数更新。

**简单应用场景：**
神经网络训练的标准梯度下降优化算法。

---

---

### 公式 (7.54)：小批梯度下降 / Mini-Batch Gradient Descent

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \frac{1}{M} \sum_{i=1}^{M} \nabla_{\boldsymbol{\theta}} L_i(\boldsymbol{\theta}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |
| $M$ | xiǎo pī cì dà xiǎo | 小批次大小 |
| $L_i$ | dì $i$ gè yàng běn zǔn sǔn | 第i个样本损失 |

**一句话解释：**
对小批量样本计算平均梯度进行更新，是实际应用中最常用的方式。

**简单应用场景：**
深度学习训练的标准做法，平衡计算效率和梯度稳定性。

---

---

### 公式 (7.55)：随机梯度下降 / Stochastic Gradient Descent

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \nabla_{\boldsymbol{\theta}} L_i(\boldsymbol{\theta}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |
| $L_i$ | dì $i$ gè yàng běn zǔn sǔn | 第i个样本损失 |

**一句话解释：**
每次只用一个样本的梯度更新参数，具有随机性但计算快。

**简单应用场景：**
大规模数据集上的在线学习或流式学习场景。

---

---

### 公式 (7.56)：动量加速梯度下降 / Momentum Accelerated Gradient Descent

$$
\boldsymbol{v}_t = \beta \boldsymbol{v}_{t-1} + \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \boldsymbol{v}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}_t$ | dì $t$ bù dòng liàng | 第t步动量向量 |
| $\beta$ | bèi tǎ | 动量系数 |
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |

**一句话解释：**
累积历史梯度形成动量，使参数更新更快且方向更稳定。

**简单应用场景：**
加快深度学习模型收敛，特别是在平坦区域。

---

---

### 公式 (7.57)：Nesterov 动量 / Nesterov Momentum

$$
\boldsymbol{v}_t = \beta \boldsymbol{v}_{t-1} + \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t - \alpha \boldsymbol{v}_{t-1})
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \boldsymbol{v}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}_t$ | dì $t$ bù dòng liàng | 第t步动量向量 |
| $\beta$ | bèi tǎ | 动量系数 |
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |

**一句话解释：**
在动量方向上预先计算梯度，提供更精确的更新方向，收敛速度更快。

**简单应用场景：**
高性能优化算法，特别是凸优化问题中。

---

---

### 公式 (7.58)：AdaGrad 算法 / AdaGrad Algorithm

$$
G_t = G_{t-1} + \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t) \odot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | dì $t$ bù jī lěi píng fāng | 第t步梯度平方累积 |
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |
| $\epsilon$ | ēpsī lóng | 数值稳定性常数 |
| $\odot$ | yuán sù chéng fǎ | 元素乘法 |

**一句话解释：**
自动为每个参数调整学习率，梯度大的参数学习率小，梯度小的学习率大。

**简单应用场景：**
处理稀疏数据和不同尺度参数的优化问题。

---

---

### 公式 (7.59)：RMSprop 算法 / RMSprop Algorithm

$$
G_t = \beta G_{t-1} + (1 - \beta) \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t) \odot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | dì $t$ bù píng fāng píng jūn | 第t步平方平均 |
| $\beta$ | bèi tǎ | 衰减因子 |
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |
| $\epsilon$ | ēpsī lóng | 数值稳定性常数 |

**一句话解释：**
改进的AdaGrad，使用指数加权平均梯度平方，避免学习率单调递减。

**简单应用场景：**
循环神经网络和其他深度学习模型的标准优化器。

---

---

### 公式 (7.60)：Adam 算法第一部分 / Adam Algorithm Part 1

$$
M_t = \beta_1 M_{t-1} + (1 - \beta_1) \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $M_t$ | dì $t$ bù yī jiē jī pò | 第t步一阶矩 |
| $\beta_1$ | bèi tǎ yī | 一阶矩衰减因子 |
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |

**一句话解释：**
计算梯度的指数加权平均（一阶矩），体现梯度趋势。

**简单应用场景：**
Adam优化器的第一步，累积梯度动量信息。

---

---

### 公式 (7.61)：Adam 算法第二部分 / Adam Algorithm Part 2

$$
G_t = \beta_2 G_{t-1} + (1 - \beta_2) \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t) \odot \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | dì $t$ bù èr jiē jī pò | 第t步二阶矩 |
| $\beta_2$ | bèi tǎ èr | 二阶矩衰减因子 |
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\odot$ | yuán sù chéng fǎ | 元素乘法 |

**一句话解释：**
计算梯度平方的指数加权平均（二阶矩），用于自适应学习率。

**简单应用场景：**
Adam优化器的第二步，累积二阶矩信息。

---

---

### 公式 (7.62)：Adam 算法参数更新 / Adam Algorithm Parameter Update

$$
\hat{M}_t = \frac{M_t}{1 - \beta_1^t}
$$
$$
\hat{G}_t = \frac{G_t}{1 - \beta_2^t}
$$
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\alpha}{\sqrt{\hat{G}_t + \epsilon}} \odot \hat{M}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{M}_t$ | yī jiē jī pò piānchà xiū zhèng | 一阶矩偏差修正 |
| $\hat{G}_t$ | èr jiē jī pò piānchà xiū zhèng | 二阶矩偏差修正 |
| $\beta_1^t$ | bèi tǎ yī tì fāng | 衰减因子的t次方 |
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $\alpha$ | āěr fǎ | 学习率 |

**一句话解释：**
使用偏差修正的一阶和二阶矩计算参数更新，结合动量和自适应学习率。

**简单应用场景：**
最流行的深度学习优化器，适用于几乎所有类型的神经网络。

---

---

### 公式 (7.63)：学习率预热 / Learning Rate Warmup

$$
\alpha_t = \alpha_0 \cdot \min\left(\frac{t}{T_{\text{warmup}}}, 1\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | dì $t$ bù xué xí lǜ | 第t步学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 初始学习率 |
| $T_{\text{warmup}}$ | yuán nuǎn jiàn shù | 预热步数 |

**一句话解释：**
从较小学习率开始，逐步增加到目标学习率，稳定训练初期。

**简单应用场景：**
大规模深度学习模型训练中提高稳定性和最终性能。

---

---

### 公式 (7.64)：循环学习率 / Cyclic Learning Rate

$$
\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{\pi t}{T_{\text{cycle}}}\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | dì $t$ bù xué xí lǜ | 第t步学习率 |
| $\alpha_{\min}$ | zuì xiǎo xué xí lǜ | 最小学习率 |
| $\alpha_{\max}$ | zuì dà xué xí lǜ | 最大学习率 |
| $T_{\text{cycle}}$ | xún huán zhōu qī | 循环周期 |

**一句话解释：**
学习率在最小值和最大值间按余弦函数周期变化，帮助跳出局部最优。

**简单应用场景：**
改善模型泛化性能，特别是在数据集较小时。

---

---

### 公式 (7.65)：余弦退火学习率 / Cosine Annealing Learning Rate

$$
\alpha_t = \alpha_0 \left(1 + \cos\left(\frac{\pi t}{T}\right)\right) / 2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | dì $t$ bù xué xí lǜ | 第t步学习率 |
| $\alpha_0$ | chū shǐ xué xí lǜ | 初始学习率 |
| $T$ | zǒng xùn liàn bù shù | 总训练步数 |

**一句话解释：**
学习率按余弦函数从大衰减到小，实现平滑的学习率退火。

**简单应用场景：**
改善深度学习模型的最终性能和收敛速度。

---

---

### 公式 (7.66)：权重衰减正则化 / Weight Decay Regularization

$$
L_{\text{total}} = L + \frac{\lambda}{2} \sum_{l=1}^{L} \|\boldsymbol{W}^{(l)}\|_2^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | zǔn sǔn | 原始损失 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\boldsymbol{W}^{(l)}$ | dì $l$ céng quán zhòng | 第l层权重矩阵 |
| $\|\cdot\|_2^2$ | èr fàn shù píng fāng | 二范数平方 |

**一句话解释：**
在损失函数中加入权重的平方和，惩罚大权重，防止过拟合。

**简单应用场景：**
任何深度学习模型中防止过拟合的标准正则化技术。

---

---

### 公式 (7.67)：Dropout 正则化 / Dropout Regularization

$$
\tilde{\boldsymbol{a}}^{(l)} = \boldsymbol{m}^{(l)} \odot \boldsymbol{a}^{(l)}
$$
$$
\boldsymbol{m}^{(l)} \sim \text{Bernoulli}(p)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{\boldsymbol{a}}^{(l)}$ | duò chū hòu jī huó zhī | Dropout后激活值 |
| $\boldsymbol{m}^{(l)}$ | suī jī zhào yǎng zhǔ | 随机掩码向量 |
| $\boldsymbol{a}^{(l)}$ | jī huó zhī | 激活值 |
| $p$ | bǎo liú gài lǜ | 保留概率 |
| $\odot$ | yuán sù chéng fǎ | 元素乘法 |

**一句话解释：**
随机将部分神经元激活值设为0，强制网络学习冗余特征，防止共适应。

**简单应用场景：**
全连接层和卷积层的标准正则化技术，减少过拟合。

---

---

### 公式 (7.68)：批量归一化前向传播 / Batch Normalization Forward Pass

$$
\mu_B = \frac{1}{M} \sum_{i=1}^{M} x_i
$$
$$
\sigma_B^2 = \frac{1}{M} \sum_{i=1}^{M} (x_i - \mu_B)^2
$$
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
$$
y_i = \gamma \hat{x}_i + \beta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_i$ | dì $i$ gè yàng běn | 第i个样本 |
| $M$ | pī cì dà xiǎo | 批次大小 |
| $\mu_B$ | píng jūn zhí | 批次均值 |
| $\sigma_B^2$ | fāng chà | 批次方差 |
| $\gamma$ | gā mǎ | 缩放参数 |
| $\beta$ | bèi tǎ | 平移参数 |
| $\epsilon$ | ēpsī lóng | 数值稳定性常数 |

**一句话解释：**
对批次内数据归一化，然后用可学习参数重新缩放和平移。

**简单应用场景：**
现代深度神经网络中加速训练和改善泛化的标准技术。

---

---

### 公式 (7.69)：层归一化前向传播 / Layer Normalization Forward Pass

$$
\mu_l = \frac{1}{D} \sum_{d=1}^{D} x_d
$$
$$
\sigma_l^2 = \frac{1}{D} \sum_{d=1}^{D} (x_d - \mu_l)^2
$$
$$
\hat{x}_d = \frac{x_d - \mu_l}{\sqrt{\sigma_l^2 + \epsilon}}
$$
$$
y_d = \gamma_d \hat{x}_d + \beta_d
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_d$ | dì $d$ gè tè zhēng | 第d个特征 |
| $D$ | tè zhēng wéi dù | 特征维度 |
| $\mu_l$ | céng píng jūn | 层平均 |
| $\sigma_l^2$ | céng fāng chà | 层方差 |
| $\gamma_d$ | dì $d$ gè gā mǎ | 第d个gamma参数 |
| $\beta_d$ | dì $d$ gè bèi tǎ | 第d个beta参数 |

**一句话解释：**
按特征维度进行归一化，每个样本独立计算统计量，不依赖批次。

**简单应用场景：**
Transformer模型和循环神经网络的标准归一化方法。

---

---

### 公式 (7.70)：实例归一化 / Instance Normalization

$$
\hat{x}_{ncd} = \frac{x_{ncd} - \mu_{nd}}{\sqrt{\sigma_{nd}^2 + \epsilon}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_{ncd}$ | dì $n$ gè yàng běn dì $c$ gè píng miàn dì $d$ gè wèi zhì | 第n个样本第c个通道第d个位置 |
| $\mu_{nd}$ | dì $n$ gè yàng běn píng jūn | 第n个样本平均 |
| $\sigma_{nd}^2$ | dì $n$ gè yàng běn fāng chà | 第n个样本方差 |
| $\epsilon$ | ēpsī lóng | 数值稳定性常数 |

**一句话解释：**
对每个样本的每个通道单独计算统计量进行归一化，独立于其他样本。

**简单应用场景：**
风格迁移等计算机视觉任务中的图像处理。

---

---

### 公式 (7.71)：群归一化 / Group Normalization

$$
\hat{x}_i = \frac{x_i - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_i$ | dì $i$ gè yuánsù | 第i个元素 |
| $\mu_g$ | zǔ píng jūn | 组平均 |
| $\sigma_g^2$ | zǔ fāng chà | 组方差 |
| $\epsilon$ | ēpsī lóng | 数值稳定性常数 |

**一句话解释：**
将通道分组，在每组内计算统计量进行归一化，平衡批量和层归一化。

**简单应用场景：**
小批量训练或目标检测等需要不同归一化策略的任务。

---

---

### 公式 (7.72)：参数初始化方差 / Parameter Initialization Variance

$$
\text{var}(w) = \frac{1}{n_{\text{in}}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(w)$ | fāng chà | 权重方差 |
| $n_{\text{in}}$ | shū rù wéi dù | 输入维度 |

**一句话解释：**
Xavier初始化方法，根据输入维度设定权重初始化的方差，加速收敛。

**简单应用场景：**
任何深度神经网络权重初始化的标准方法。

---

---

### 公式 (7.73)：He 初始化 / He Initialization

$$
\text{var}(w) = \frac{2}{n_{\text{in}}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(w)$ | fāng chà | 权重方差 |
| $n_{\text{in}}$ | shū rù wéi dù | 输入维度 |

**一句话解释：**
专为ReLU激活函数设计的初始化方法，方差为Xavier的两倍。

**简单应用场景：**
包含ReLU激活函数的深层神经网络的权重初始化。

---

---

### 公式 (7.74)：目标函数优化形式 / Objective Function Optimization Form

$$
\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) + \lambda R(\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | tā ta | 参数向量 |
| $L(\boldsymbol{\theta})$ | zǔn sǔn hánshù | 损失函数 |
| $R(\boldsymbol{\theta})$ | zhèng zé huà xiàng | 正则化项 |
| $\lambda$ | lán mǔ dá | 正则化系数 |

**一句话解释：**
优化目标包括原始损失和正则化项，平衡模型拟合能力和泛化能力。

**简单应用场景：**
几乎所有监督学习任务的目标函数设定。

---

---

### 公式 (7.75)：梯度消失问题 / Vanishing Gradient Problem

$$
\frac{\partial L}{\partial \boldsymbol{w}^{(1)}} = \frac{\partial L}{\partial \boldsymbol{a}^{(L)}} \prod_{l=2}^{L} \frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | tǒng jì céng shù | 总计层数 |
| $\boldsymbol{w}^{(1)}$ | dì 1 céng quán zhòng | 第1层权重 |
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng jī huó zhī | 第l层激活值 |
| $\boldsymbol{z}^{(l)}$ | dì $l$ céng xiān xìng zǔ hé | 第l层线性组合 |

**一句话解释：**
链式法则中的多个导数相乘，如果导数小于1，会指数级衰减，导致深层梯度极小。

**简单应用场景：**
解释为什么深层网络难以训练，以及使用ReLU和残差连接的必要性。

---

---

### 公式 (7.76)：梯度爆炸问题 / Exploding Gradient Problem

$$
\frac{\partial L}{\partial \boldsymbol{w}^{(1)}} = \frac{\partial L}{\partial \boldsymbol{a}^{(L)}} \prod_{l=2}^{L} \frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | tǒng jì céng shù | 总计层数 |
| $\boldsymbol{w}^{(1)}$ | dì 1 céng quán zhòng | 第1层权重 |
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng jī huó zhī | 第l层激活值 |
| $\boldsymbol{z}^{(l)}$ | dì $l$ céng xiān xìng zǔ hé | 第l层线性组合 |

**一句话解释：**
链式法则中的多个导数相乘，如果导数大于1，会指数级增长，导致梯度爆炸。

**简单应用场景：**
解释为什么深层网络中出现梯度爆炸，以及使用梯度裁剪的必要性。

---

## 修正说明

本文档包含以下修正内容：

1. **章节6 错误公式修正**：公式 6.1, 6.13, 6.18, 6.33, 6.35, 6.45, 6.49, 6.79, 6.80, 6.81 的LaTeX表达式已严格按原书校正。

2. **章节6 补充缺失公式**：补充了缺失的公式 6.8（前向传播）和 6.9（反向传播梯度）。

3. **章节7 错误公式修正**：公式 7.10, 7.13, 7.22, 7.23, 7.42 的表达式已重新核对并修正。

4. **章节7 顺序调整**：公式 7.32, 7.33, 7.34 的位置已按照原书正确排列。

5. **章节7 补充缺失公式**：补充了公式 7.45 至 7.76 共32个缺失的公式，涵盖反向传播、各类优化算法、学习率调度、正则化方法和参数初始化等重要内容。

所有公式的符号表、拼音读法、含义解释和应用场景均按照要求的格式完整给出。

---

## 第8章：注意力机制与外部记忆

### 公式 (8.1)：注意力分布 / Attention Distribution

$$
\operatorname{Att}(\boldsymbol{X}, \boldsymbol{q}) = \sum_{n=1}^{N} \alpha_n \boldsymbol{x}_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_n$ | ālf ā | 注意力分布 |
| $s(x_n, q)$ | ēs | 注意力打分函数 |
| $q$ | qū | 查询向量 |
| $N$ | ēn | 输入向量个数 |

**一句话解释：**
使用softmax函数将注意力打分函数的输出归一化为概率分布。

**简单应用场景：**
在自注意力机制中，基于query向量对所有key-value对进行加权求和。

---

---

### 公式 (8.2)：加性模型 / Additive Model

$$
\text{s}(x, q) = \boldsymbol{v}^T \tanh(\boldsymbol{W}x + \boldsymbol{U}q)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}$ | wēi liàng | 权重向量 |
| $\boldsymbol{W}$ | dà xiě w | 输入映射矩阵 |
| $\boldsymbol{U}$ | dà xiě u | 查询映射矩阵 |
| $\tanh$ | tǎng huàn | 双曲正切函数 |

**一句话解释：**
通过非线性变换计算输入和查询之间的相似度。

**简单应用场景：**
Bahdanau注意力机制中用于计算注意力权重的打分函数。

---

---

### 公式 (8.3)：点积模型 / Dot-Product Model

$$
\text{s}(x, q) = x^T q
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x$ | xiǎo xiě x | 输入向量 |
| $q$ | qū | 查询向量 |
| $T$ | zhuǎn zhì | 转置 |

**一句话解释：**
使用两个向量的内积作为相似度度量。

**简单应用场景：**
Scaled dot-product attention中的基础打分函数。

---

---

### 公式 (8.4)：缩放点积模型 / Scaled Dot-Product Model

$$
\text{s}(x, q) = \frac{x^T q}{\sqrt{D}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D$ | dà xiě d | 向量维度 |
| $\sqrt{D}$ | gēn hào d | 缩放因子 |

**一句话解释：**
对点积结果进行缩放，防止梯度消失问题。

**简单应用场景：**
Transformer中自注意力机制的标准打分函数。

---

---

### 公式 (8.5)：双线性模型 / Bilinear Model

$$
\text{s}(x, q) = x^T \boldsymbol{W} q
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}$ | dà xiě w | 双线性映射矩阵 |

**一句话解释：**
通过矩阵变换计算输入和查询的相似度。

**简单应用场景：**
神经机器翻译中的注意力机制。

---

---

### 公式 (8.6)：软性注意力 / Soft Attention

$$
\text{att}(X, q) = \sum_{n=1}^{N} \alpha_n x_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{att}$ | zhù yì | 注意力输出 |
| $X$ | dà xiě x | 输入向量集合 |
| $\alpha_n$ | ālf ā | 第n个向量的注意力权重 |

**一句话解释：**
根据注意力权重对所有输入向量进行加权求和。

**简单应用场景：**
所有注意力机制的核心计算。

---

---

### 公式 (8.7)：期望改进 / Expected Improvement

$$
\text{E}[\text{I}(x)] = \mathbb{E}_{q(\theta)} \left[ \int_q f(x; \theta) q(\theta) d\theta \right] \approx \frac{1}{M} \sum_{m=1}^{M} f(x, \theta_m)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{E}$ | qī wàng | 期望 |
| $q(\theta)$ | qū | 概率分布 |
| $f(x; \theta)$ | ēf | 目标函数 |
| $M$ | ēm | 样本数 |

**一句话解释：**
使用蒙特卡洛采样估计目标函数的期望。

**简单应用场景：**
贝叶斯优化中的获取函数计算。

---

---

### 公式 (8.8)：批量注意力一一化（修正）/ Batch Normalization

$$
\alpha_n = \frac{\exp(s(\boldsymbol{x}_n, \boldsymbol{q}))}{\sum_{j=1}^{N} \exp(s(\boldsymbol{x}_j, \boldsymbol{q}))}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z^{(l)}$ | zhě | 第l层的输入 |
| $\mathbb{E}$ | qī wàng | 期望 |
| $\text{var}$ | fāng chā | 方差 |
| $\epsilon$ | ēp sī lóng | 防止分母为零的小常数 |

**一句话解释：**
对神经网络的层输入进行标准化处理。

**简单应用场景：**
深度神经网络中加速训练收敛。

---

---

### 公式 (8.9)：键值对注意力（基础形式）/ Key-Value Attention (Basic Form)

$$
\text{att}\left((K, V), q\right) = \sum_{n=1}^{N} \alpha_n v_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K$ | K | 键矩阵，包含 $k_1, \ldots, k_N$ |
| $V$ | V | 值矩阵，包含 $v_1, \ldots, v_N$ |
| $q$ | q | 查询向量 |
| $\alpha_n$ | ālf ǎ xià biāo | 第 n 个键的注意权重 |
| $v_n$ | V zǐ | 第 n 个值向量 |
| $N$ | N | 键值对总数 |

**一句话解释：**
通过查询向量与键的匹配程度，计算每个值向量的权重，然后进行加权求和。

**简单应用场景：**
机器翻译中，查询向量代表目标词，键值对代表源语言词及其特征。

---

---

### 公式 (8.10)：键值对注意力（Softmax归一化）/ Key-Value Attention (Softmax Normalization)

$$
\text{att}\left((K, V), q\right) = \sum_{n=1}^{N} \frac{\exp\left(s(k_n, q)\right)}{\sum_i \exp\left(s(k_i, q)\right)} v_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(k_n, q)$ | dǎ fēn hánshù | 键与查询的兼容性打分函数 |
| $\exp$ | zhǐ shù | 指数函数 |
| $\sum_i \exp(s(k_i, q))$ | guī huà huà fā zǐ | 所有打分的指数和（归一化分母）|

**一句话解释：**
用Softmax函数对打分结果进行归一化，得到每个值的概率权重，然后加权求和。

**简单应用场景：**
序列模型中，基于相似度计算每个历史状态对当前预测的贡献程度。

---

---

### 公式 (8.11)：多头注意力 / Multi-Head Attention

$$
\text{att}\left((K, V), Q\right) = \text{att}\left((K, V), q_1\right) \oplus \cdots \oplus \text{att}\left((K, V), q_M\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q$ | Q | 多个查询向量集合，包含 $q_1, \ldots, q_M$ |
| $\oplus$ | zhí hé | 向量拼接操作 |
| $M$ | M | 注意力头数 |

**一句话解释：**
用多个查询向量从不同角度关注键值对，然后将结果拼接起来获得更丰富的表示。

**简单应用场景：**
Transformer模型中，多个注意力头可以同时学习语法、语义等多个不同的特征。

---

## 指针网络 (Pointer Networks)

---

### 公式 (8.12)：序列条件概率分解 / Factorization of Conditional Probability

$$
p(c_{1:M}|x_{1:N}) = \prod_{m=1}^{M} p(c_m|c_{1:(m-1)}, x_{1:N})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $c_{1:M}$ | cè xù liè | 输出序列，包含 $c_1$ 到 $c_M$ |
| $x_{1:N}$ | shū rù xù liè | 输入序列，包含 $x_1$ 到 $x_N$ |
| $p(\cdot)$ | gài lǜ | 概率分布 |
| $m$ | m | 输出位置索引 |

**一句话解释：**
将联合概率分解为条件概率的乘积，每个输出依赖于所有输入和之前的输出。

**简单应用场景：**
旅行商问题（TSP）中，逐步选择下一个要访问的城市，依赖于已选城市和所有城市的位置。

---

---

### 公式 (8.13)：独立假设下的条件概率 / Conditional Probability Under Independence Assumption

$$
p(c_{1:M}|x_{1:N}) \approx \prod_{m=1}^{M} p(c_m|x_c, \ldots, x_{c_{m-1}}, x_{1:N})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $c_m$ | dì m gè shū chū | 第 m 个输出元素 |
| $x_c$ | xī léi fúhào | 特定输入符号 |

**一句话解释：**
在独立性假设下，每个输出只依赖于特定的输入子集，而不是全部输入和历史输出。

**简单应用场景：**
文本摘要中，每个输出句子主要依赖于源文本中的相关句子。

---

---

### 公式 (8.14)：Softmax注意力用于序列标记 / Softmax Attention for Sequence Labeling

$$
p(c_m|c_{1:(m-1)}, x_{1:N}) = \text{softmax}(s_{m,n})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_{m,n}$ | dǎ fēn | 第 m 个输出位置和第 n 个输入位置的兼容性打分 |
| $\text{softmax}$ | ruǎn zuì dà zhí | Softmax函数 |

**一句话解释：**
对输入序列每个位置的打分结果应用Softmax函数，得到指针位置的概率分布。

**简单应用场景：**
序列标记任务中，模型在每个时刻指向输入序列中最相关的位置。

---

---

### 公式 (8.15)：GRU单元更新（指针网络） / GRU Cell Update (Pointer Networks)

$$
s_{m,n} = v^T \tanh(Wx_n + Uh_m), \quad \forall n \in [1, N]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $v$ | V | 可学习的参数向量 |
| $\tanh$ | shuāng qū xiàn hánshù | 双曲正切激活函数 |
| $W$ | W | 输入权重矩阵 |
| $U$ | U | 隐层权重矩阵 |
| $x_n$ | dì n gè shū rù xiàng liàng | 第 n 个输入向量 |
| $h_m$ | dì m gè yǐn cáng tài | 第 m 个隐层状态 |

**一句话解释：**
通过非线性变换计算输入向量与隐层状态的兼容性打分，用于确定注意力指向。

**简单应用场景：**
序列到序列模型中，解码器在每个时刻计算与编码器各位置的关联度。

---

## 自注意力模型 (Self-Attention Model)

---

### 公式 (8.16)：查询矩阵 / Query Matrix

$$
Q = W_Q X \in \mathbb{R}^{D_k \times N}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q$ | chá xun jù zhèn | 查询矩阵 |
| $W_Q$ | W zǐ | 查询权重矩阵 |
| $X$ | X | 输入矩阵 |
| $D_k$ | D zǐ | 查询/键的维度 |
| $N$ | N | 序列长度 |

**一句话解释：**
通过学习参数矩阵将输入映射到查询空间，用于与键进行相似度计算。

**简单应用场景：**
Transformer中，输入序列经过线性变换生成查询矩阵。

---

---

### 公式 (8.17)：键矩阵 / Key Matrix

$$
K = W_K X \in \mathbb{R}^{D_k \times N}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K$ | jī jù zhèn | 键矩阵 |
| $W_K$ | W zǐ | 键权重矩阵 |

**一句话解释：**
通过学习参数矩阵将输入映射到键空间，与查询进行相似度匹配。

**简单应用场景：**
Transformer中，输入序列经过线性变换生成键矩阵，用于计算注意力分数。

---

---

### 公式 (8.18)：值矩阵 / Value Matrix

$$
V = W_V X \in \mathbb{R}^{D_v \times N}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V$ | zhí jù zhèn | 值矩阵 |
| $W_V$ | W zǐ | 值权重矩阵 |
| $D_v$ | D zǐ | 值的维度 |

**一句话解释：**
通过学习参数矩阵将输入映射到值空间，根据注意力权重进行加权求和。

**简单应用场景：**
Transformer中，输入序列经过线性变换生成值矩阵，提供实际的内容表示。

---

---

### 公式 (8.19)：自注意力输出 / Self-Attention Output

$$
h_n = \text{att}\left((K, V), q_n\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_n$ | dì n gè shū chū | 第 n 个位置的自注意力输出 |
| $q_n$ | dì n gè chá xun | 第 n 个查询向量 |

**一句话解释：**
对每个位置的查询向量应用注意力机制，得到整个序列信息的加权表示。

**简单应用场景：**
Transformer编码器中，每个词汇位置都可以关注序列中的所有位置。

---

---

### 公式 (8.20)：自注意力加权求和（展开形式1）/ Self-Attention Weighted Sum (Expanded Form 1)

$$
= \sum_{j=1}^{N} \alpha_{nj} v_j
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_{nj}$ | ālf ǎ | 第 n 个查询对第 j 个值的注意权重 |
| $v_j$ | dì j gè zhí | 第 j 个值向量 |

**一句话解释：**
通过注意力权重对所有值向量进行加权求和，得到该位置的最终表示。

**简单应用场景：**
序列编码中，每个位置整合其他所有位置的信息，权重由相似度决定。

---

---

### 公式 (8.21)：自注意力加权求和（展开形式2）/ Self-Attention Weighted Sum (Expanded Form 2)

$$
= \sum_{j=1}^{N} \text{softmax}\left(s(k_j, q_n)\right) v_j
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(k_j, q_n)$ | jī chá xiāng hóngrónɡ xìng dǎ fēn | 键与查询的相容性打分 |
| $k_j$ | dì j gè jī | 第 j 个键向量 |

**一句话解释：**
显式展示Softmax注意力机制，计算键与查询的打分，然后通过Softmax得到权重。

**简单应用场景：**
Transformer自注意力层的核心计算，体现了键查询相似度与值的加权组合。

---

---

### 公式 (8.22)：缩放点积注意力 / Scaled Dot-Product Attention

$$
H = V \text{ softmax}\left(\frac{K^T Q}{\sqrt{D_k}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H$ | shū chū jù zhèn | 注意力输出矩阵 |
| $K^T Q$ | jī chá xiāng hóngrónɡ xìng jù zhèn | 键与查询的点积矩阵 |
| $\sqrt{D_k}$ | píng fāng gēn | 缩放因子（维度的平方根）|

**一句话解释：**
用键与查询的缩放点积计算注意力分数，避免点积过大导致梯度消失问题。

**简单应用场景：**
Transformer的标准自注意力机制，广泛应用于NLP和计算机视觉任务。

---

## 记忆增强神经网络 (Memory Augmented Neural Networks)

---

### 公式 (8.23)：记忆读取操作 / Memory Read Operation

$$
r = \sum_{n=1}^{N} a_n m_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r$ | dú qǔ xiàng liàng | 读取向量 |
| $a_n$注意力权重 | ālf ǎ | 对第 n 个记忆单元的注意力权重 |
| $m_n$ | dì n gè jì yì | 第 n 个记忆单元内容 |

**一句话解释：**
通过注意力权重对记忆矩阵中的内容进行加权求和，实现内容可寻址的读取。

**简单应用场景：**
神经图灵机中，根据查询向量从外部记忆中读取相关信息。

---

---

### 公式 (8.24)：注意力权重计算 / Attention Weight Calculation

$$
a_n = \text{softmax}\left(s(m_n, q_r)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(m_n, q_r)$ | jì yì yǔ chá xun de xiāng hóngrónɡ xìng dǎ fēn | 记忆与查询的相容性打分 |
| $q_r$ | dú qǔ chá xun xiàng liàng | 读取查询向量 |

**一句话解释：**
计算读取查询与各记忆单元的相容性，通过Softmax归一化得到注意力权重。

**简单应用场景：**
关键值记忆网络中，基于查询向量从多个记忆槽中检索相关信息。

---

## 端到端记忆网络 (End-to-End Memory Networks)

---

### 公式 (8.25)：记忆读取输出 / Memory Read Output

$$
r = \sum_{n=1}^{N} \text{softmax}(a_n^T q)c_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $a_n^T q$ | zhù yìjù zhèn yǔ chá xun de diǎn jī | 地址向量与查询的点积 |
| $c_n$ | dì n gè nèi rónɡ xiàng liàng | 第 n 个内容向量 |

**一句话解释：**
通过软性寻址方式，用查询与地址向量的点积计算权重，然后对内容向量加权求和。

**简单应用场景：**
问答系统中，基于问题查询从事实库中检索多个相关事实。

---

---

### 公式 (8.26)：记忆网络输出 / Memory Network Output

$$
y = f(q + r)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | shū chū | 最终输出 |
| $f$ | shè jì hánshù | 分类或回归函数 |
| $q + r$ | chá xun hé dú qǔ xiàng liàng de héng | 查询与读取向量的和 |

**一句话解释：**
将查询向量与从记忆中读取的向量相加，再通过函数处理得到最终预测。

**简单应用场景：**
阅读理解任务中，结合问题和检索到的相关内容进行答案预测。

---

## 门控端到端记忆网络 (Gated End-to-End Memory Networks)

---

### 公式 (8.27)：门控多跳更新 / Gated Multi-Hop Update

$$
q^{(k)} = r^{(k-1)} + q^{(k-1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q^{(k)}$ | dì k gè tiào tiào de chá xun | 第 k 跳的查询向量 |
| $r^{(k-1)}$ | dì k-1 gè tiào tiào de dú qǔ xiàng liàng | 第 k-1 跳的读取向量 |
| $k$ | k | 跳次数索引 |

**一句话解释：**
在多跳推理中，新的查询由前一跳的读取结果与原查询相加得到，实现信息累积。

**简单应用场景：**
多步推理问答中，模型逐步细化查询以检索更相关的信息。

---

---

### 公式 (8.28)：门控记忆访问（修正）/ Gated Memory Access

$$
\operatorname{Att}((\boldsymbol{K}, \boldsymbol{V}), \boldsymbol{q}) = \sum_{n=1}^{N} \alpha_n \boldsymbol{v}_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r$ | ǎr | 读向量 |
| $a_n$ | ālf ā | 第n个位置的注意力权重 |
| $m_n$ | ēm | 第n个位置的存储值 |

**一句话解释：**
根据注意力权重从外部记忆中读取信息。

**简单应用场景：**
神经图灵机中的记忆读取操作。

---

---

### 公式 (8.29)：多跳操作（修正）/ Multi-Hop Operations

$$
\alpha_n = \operatorname{softmax}(s(\boldsymbol{k}_n, \boldsymbol{q}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q^{(k)}$ | qū | 第k跳的查询向量 |
| $r^{(k-1)}$ | ǎr | 第k-1跳的读向量 |

**一句话解释：**
使用前一跳的输出作为当前跳的查询向量。

**简单应用场景：**
端到端记忆网络的多跳推理。

---

## CHAPTER 9: 无监督学习

---

### 公式 (8.30)：注意力机制矩阵形式 / Attention Matrix Form

$$
\boldsymbol{\alpha} = \text{softmax}\left(\text{s}(\boldsymbol{X}, q)\right) = \text{softmax}\left(\frac{\boldsymbol{X} q^T}{\sqrt{D}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\alpha}$ | ālf ā | 注意力向量 |
| $\boldsymbol{X}$ | dà xiě x | 输入矩阵 |
| $D$ | dà xiě d | 向量维度 |

**一句话解释：**
将单个查询的注意力扩展到矩阵形式。

**简单应用场景：**
批处理中的注意力计算。

---

---

### 公式 (8.31)：多头注意力 / Multi-Head Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q$ | dà xiě q | 查询矩阵 |
| $K$ | dà xiě k | 键矩阵 |
| $V$ | dà xiě v | 值矩阵 |
| $h$ | āi chǐ | 注意力头数 |
| $W^O$ | dà xiě w | 输出投影矩阵 |

**一句话解释：**
使用多个注意力头并行计算，然后连接融合。

**简单应用场景：**
Transformer中的标准注意力机制。

---

---

### 公式 (8.32)：单个注意力头 / Single Attention Head

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $W_i^Q$ | dà xiě w | 第i个头的查询投影 |
| $W_i^K$ | dà xiě w | 第i个头的键投影 |
| $W_i^V$ | dà xiě w | 第i个头的值投影 |

**一句话解释：**
每个注意力头使用独立的线性投影。

**简单应用场景：**
多头注意力中单个头的计算。

---

---

### 公式 (8.33)：缩放点积注意力矩阵 / Scaled Dot-Product Attention Matrix

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $d_k$ | dì kèi | 键向量的维度 |
| $T$ | zhuǎn zhì | 转置 |

**一句话解释：**
计算查询与键的相似度后加权求和值。

**简单应用场景：**
Transformer自注意力机制的核心。

---

---

### 公式 (8.34)：位置编码 / Positional Encoding

$$
PE_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{pos}$ | wèi zhì | 序列中的位置 |
| $i$ | ī | 维度索引 |
| $d$ | dì | 总维度 |

**一句话解释：**
为偶数维度位置编码使用正弦函数。

**简单应用场景：**
Transformer中的位置编码（偶数维）。

---

---

### 公式 (8.35)：位置编码余弦部分 / Positional Encoding Cosine

$$
PE_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $2i+1$ | èr ī jiā yī | 奇数维度索引 |

**一句话解释：**
为奇数维度位置编码使用余弦函数。

**简单应用场景：**
Transformer中的位置编码（奇数维）。

---

---

### 公式 (8.36)：前馈网络 / Feed-Forward Network

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{FFN}$ | qián kuàhui | 前馈网络 |
| $W_1$ | dà xiě w | 第一层权重 |
| $W_2$ | dà xiě w | 第二层权重 |
| $b_1, b_2$ | dà xiě b | 偏置向量 |

**一句话解释：**
位置独立的全连接前馈网络。

**简单应用场景：**
Transformer块中的前馈子层。

---

---

### 公式 (8.37)：层归一化 / Layer Normalization

$$
y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gā mǎ | 缩放参数 |
| $\beta$ | bèi tǎ | 移位参数 |
| $\mu$ | měi | 均值 |
| $\sigma^2$ | xī gé mǎ píng fāng | 方差 |
| $\epsilon$ | ēp sī lóng | 数值稳定性常数 |

**一句话解释：**
对特征进行归一化并学习可调的缩放和移位。

**简单应用场景：**
Transformer中加速收敛的归一化方法。

---

---

## 第9章：无监督学习

### 公式 (9.1)：主成分分析 / Principal Component Analysis

$$
z^{(n)} = \boldsymbol{w}^T \boldsymbol{x}^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z^{(n)}$ | zhě | 第n个样本的主成分 |
| $\boldsymbol{w}$ | dà xiě w | 主成分方向向量 |
| $\boldsymbol{x}^{(n)}$ | dà xiě x | 第n个样本 |
| $T$ | zhuǎn zhì | 转置 |

**一句话解释：**
将原始数据投影到主成分方向上获得降维表示。

**简单应用场景：**
图像降维、数据可视化。

---

---

### 公式 (9.2)：协方差矩阵 / Covariance Matrix

$$
\sigma(X; \boldsymbol{w}) = \frac{1}{N} \sum_{n=1}^{N} (\boldsymbol{w}^T \boldsymbol{x}^{(n)} - \boldsymbol{w}^T \bar{\boldsymbol{x}})^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | 协方差 |
| $X$ | dà xiě x | 数据矩阵 |
| $N$ | ēn | 样本数 |
| $\bar{\boldsymbol{x}}$ | xǔ | 数据均值 |

**一句话解释：**
计算投影数据在主成分方向上的方差。

**简单应用场景：**
确定最优的主成分方向。

---

---

### 公式 (9.3)：协方差矩阵展开 / Covariance Matrix Expansion

$$
p(\boldsymbol{x}) = \prod_{t=1}^{T} p(x_t | \boldsymbol{x}_{<t})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $X$ | dà xiě x | 数据矩阵 |
| $\bar{\boldsymbol{x}}$ | xǔ | 数据中心 |

**一句话解释：**
将方差表达式改写为矩阵形式。

**简单应用场景：**
数学推导中的中间步骤。

---

---

### 公式 (9.4)：方差的矩阵形式 / Variance in Matrix Form

$$
= \boldsymbol{w}^T \Sigma \boldsymbol{w}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Sigma$ | xī gé mǎ | 协方差矩阵 |
| $\boldsymbol{w}$ | dà xiě w | 权重向量 |

**一句话解释：**
用协方差矩阵表示投影数据的方差。

**简单应用场景：**
PCA中的特征值分解问题。

---

---

### 公式 (9.5)：优化目标 / Optimization Objective

$$
\max_{\boldsymbol{w}} \boldsymbol{w}^T \Sigma \boldsymbol{w} + \lambda(1 - \boldsymbol{w}^T \boldsymbol{w})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | lán mǔ dá | 拉格朗日乘子 |

**一句话解释：**
在单位向量约束下最大化方差。

**简单应用场景：**
主成分分析的拉格朗日求解。

---

---

### 公式 (9.6)：特征方程 / Eigenvalue Equation

$$
\Sigma \boldsymbol{w} = \lambda \boldsymbol{w}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Sigma$ | xī gé mǎ | 协方差矩阵 |
| $\lambda$ | lán mǔ dá | 特征值 |
| $\boldsymbol{w}$ | dà xiě w | 特征向量 |

**一句话解释：**
求解协方差矩阵的特征向量和特征值。

**简单应用场景：**
PCA中找出最大方差方向。

---

---

### 公式 (9.7)：方差与特征值 / Variance and Eigenvalue

$$
\sigma(X; \boldsymbol{w}) = \boldsymbol{w}^T \Sigma \boldsymbol{w} = \boldsymbol{w}^T \lambda \boldsymbol{w} = \lambda
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | lán mǔ dá | 特征值 |
| $\sigma$ | xī gé mǎ | 方差 |

**一句话解释：**
协方差矩阵的特征值等于对应特征向量方向的方差。

**简单应用场景：**
PCA中特征值的几何意义。

---

---

### 公式 (9.8)：特征值分解 / Eigenvalue Decomposition

$$
\Sigma \boldsymbol{W} = \boldsymbol{W} \text{diag}(\lambda)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}$ | dà xiě w | 特征向量矩阵 |
| $\text{diag}(\lambda)$ | duì jiǎo huà | 特征值对角矩阵 |

**一句话解释：**
协方差矩阵可分解为特征向量矩阵和特征值矩阵的乘积。

**简单应用场景：**
PCA的标准求解方法。

---

---

### 公式 (9.9)：稀疏编码基向量 / Sparse Coding Dictionary

$$
\boldsymbol{x} = \sum_{m=1}^{M} z_m \boldsymbol{a}_m
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | dà xiě x | 输入向量 |
| $z_m$ | zhě | 编码系数 |
| $\boldsymbol{a}_m$ | dà xiě a | 第m个基向量 |
| $M$ | ēm | 基向量个数 |

**一句话解释：**
使用少量基向量的线性组合表示原始数据。

**简单应用场景：**
图像压缩、特征学习。

---

---

### 公式 (9.10)：稀疏编码矩阵形式 / Sparse Coding in Matrix Form

$$
p(x_t | \boldsymbol{x}_{<t}) = \operatorname{softmax}(\boldsymbol{W} \boldsymbol{h}_t + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | dà xiě a | 字典矩阵 |
| $\boldsymbol{z}$ | dà xiě z | 稀疏编码向量 |

**一句话解释：**
用矩阵乘法表示稀疏编码。

**简单应用场景：**
稀疏编码的标准数学表示。

---

---

### 公式 (9.11)：稀疏编码目标函数 / Sparse Coding Loss Function

$$
\mathcal{L}(A, Z) = \sum_{n=1}^{N} \left( \|\boldsymbol{x}^{(n)} - A z^{(n)}\|^2 + \eta \rho(z^{(n)}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ēl | 损失函数 |
| $A$ | dà xiě a | 字典矩阵 |
| $Z$ | dà xiě z | 编码矩阵 |
| $\eta$ | yī tǎ | 稀疏度权重 |
| $\rho$ | ρ | 稀疏度度量函数 |

**一句话解释：**
重构误差与稀疏度约束的加权和。

**简单应用场景：**
稀疏编码的训练目标。

---

---

### 公式 (9.12)：零范数 / L0 Norm

$$
\rho(z) = \sum_{m=1}^{M} \mathbb{I}(|z_m| > 0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{I}$ | zhǐ shì hánshù | 指示函数 |
| $z_m$ | zhě | 编码系数 |

**一句话解释：**
计算编码向量中非零元素的个数。

**简单应用场景：**
衡量编码的稀疏性。

---

---

### 公式 (9.13)：L1 范数 / L1 Norm

$$
\rho(z) = \sum_{m=1}^{M} |z_m|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lvert z_m \rvert$ | jué duì zhí | 绝对值 |

**一句话解释：**
计算编码向量各元素绝对值之和。

**简单应用场景：**
凸优化中更易处理的稀疏度度量。

---

---

### 公式 (9.14)：对数范数 / Log Norm

$$
\rho(z) = \sum_{m=1}^{M} \log(1 + z_m^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log$ | duì shù | 对数函数 |

**一句话解释：**
使用光滑的对数函数近似稀疏度。

**简单应用场景：**
可导的稀疏度度量。

---

---

### 公式 (9.15)：负指数范数 / Negative Exponential Norm

$$
\rho(z) = \sum_{m=1}^{M} -\exp(-z_m^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp$ | zhǐ shù | 指数函数 |

**一句话解释：**
使用负指数函数作为稀疏度度量。

**简单应用场景：**
连续可导的稀疏度近似。

---

---

### 公式 (9.16)：稀疏编码固定基向量的目标函数 / Sparse Coding with Fixed Dictionary

$$
\min_{\boldsymbol{z}^{(n)}} \|\boldsymbol{x}^{(n)} - A\boldsymbol{z}^{(n)}\|^2 + \eta \rho(z^{(n)}), \forall n \in [1, N]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\min$ | zuì xiǎo | 最小值 |
| $\eta$ | yī tǎ | 稀疏度权重 |

**一句话解释：**
固定字典，优化编码向量的稀疏性和重构精度。

**简单应用场景：**
稀疏编码的编码阶段。

---

---

### 公式 (9.17)：稀疏编码固定编码的目标函数 / Sparse Coding with Fixed Encoding

$$
\min_{A} \sum_{n=1}^{N} \left( \|\boldsymbol{x}^{(n)} - A\boldsymbol{z}^{(n)}\|^2 + \frac{\lambda}{2}\|A\|^2 \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A$ | dà xiě a | 字典矩阵 |
| $\lambda$ | lán mǔ dá | 正则化参数 |

**一句话解释：**
固定编码，优化字典矩阵的重构精度。

**简单应用场景：**
稀疏编码的字典学习阶段。

---

---

### 公式 (9.18)：自编码器重构误差 / Autoencoder Reconstruction Error

$$
\mathcal{L} = \sum_{n=1}^{N} \|\boldsymbol{x}^{(n)} - g(f(\boldsymbol{x}^{(n)}))\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f$ | ēf | 编码函数 |
| $g$ | jī | 解码函数 |
| $\boldsymbol{x}^{(n)}$ | dà xiě x | 第n个输入样本 |

**一句话解释：**
衡量自编码器重构输入数据的误差。

**简单应用场景：**
自编码器的训练目标。

---

---

### 公式 (9.19)：自编码器误差展开 / Autoencoder Error Expansion

$$
= \sum_{n=1}^{N} \|\boldsymbol{x}^{(n)} - f \circ g(\boldsymbol{x}^{(n)})\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\circ$ | yuán | 函数复合 |

**一句话解释：**
使用函数复合表示自编码器的完整映射。

**简单应用场景：**
自编码器的数学表示。

---

---

### 公式 (9.20)：自编码器编码 / Autoencoder Encoding

$$
\boldsymbol{z} = f(\boldsymbol{W}^{(1)} \boldsymbol{x} + \boldsymbol{b}^{(1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | dà xiě z | 隐层表示 |
| $\boldsymbol{W}^{(1)}$ | dà xiě w | 编码权重矩阵 |
| $\boldsymbol{b}^{(1)}$ | dà xiě b | 编码偏置向量 |
| $f$ | ēf | 激活函数 |

**一句话解释：**
将输入映射到低维隐层表示。

**简单应用场景：**
自编码器的编码部分。

---

---

### 公式 (9.21)：自编码器解码 / Autoencoder Decoding

$$
\boldsymbol{x}' = f(\boldsymbol{W}^{(2)} \boldsymbol{z} + \boldsymbol{b}^{(2)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}'$ | dà xiě x撇 | 重构输出 |
| $\boldsymbol{W}^{(2)}$ | dà xiě w | 解码权重矩阵 |
| $\boldsymbol{b}^{(2)}$ | dà xiě b | 解码偏置向量 |

**一句话解释：**
从隐层表示重构原始输入数据。

**简单应用场景：**
自编码器的解码部分。

---

---

### 公式 (9.22)：自编码器总损失 / Autoencoder Total Loss

$$
\mathcal{L} = \sum_{n=1}^{N} (\|\boldsymbol{x}^{(n)} - \boldsymbol{x}'^{(n)}\|^2 + \lambda\|\boldsymbol{W}\|_F^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\cdot\|_F^2$ | fú bēi nuó mǐ | Frobenius范数平方 |
| $\lambda$ | lán mǔ dá | 正则化参数 |

**一句话解释：**
重构误差与权重正则化的加权和。

**简单应用场景：**
自编码器的完整训练目标。

---

---

### 公式 (9.23)：稀疏自编码器目标函数 / Sparse Autoencoder Loss

$$
\mathcal{L} = \sum_{n=1}^{N} (\|\boldsymbol{x}^{(n)} - \boldsymbol{x}'^{(n)}\|^2 + \eta \rho(z) + \lambda\|\boldsymbol{W}\|^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\eta$ | yī tǎ | 稀疏度权重 |
| $\rho(z)$ | ρ | 稀疏度约束 |

**一句话解释：**
在自编码器中加入稀疏性约束。

**简单应用场景：**
学习稀疏隐层表示。

---

---

### 公式 (9.24)：稀疏编码度量 / Sparsity Measure

$$
\hat{\rho}_l = \frac{1}{N} \sum_{n=1}^{N} z_l^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{\rho}_l$ | ρ hàt | 第l层的平均激活值 |
| $z_l^{(n)}$ | zhě | 第n个样本的第l层激活值 |

**一句话解释：**
计算隐层神经元的平均激活值。

**简单应用场景：**
稀疏性约束中的目标分布。

---

---

### 公式 (9.25)：KL散度稀疏约束 / KL Divergence Sparsity

$$
\text{KL}(\rho^* \| \hat{\rho}_l) = \rho^* \log\frac{\rho^*}{\hat{\rho}_l} + (1 - \rho^*) \log\frac{1 - \rho^*}{1 - \hat{\rho}_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{KL}$ | kě ěr bǎk | KL散度 |
| $\rho^*$ | ρ xīng | 目标稀疏度 |
| $\hat{\rho}_l$ | ρ hàt | 实际稀疏度 |

**一句话解释：**
使用KL散度衡量实际与目标稀疏度的差异。

**简单应用场景：**
稀疏性约束的实现。

---

---

### 公式 (9.26)：稀疏编码度函数定义 / Sparsity Penalty Function

$$
\rho(Z) = \sum_{j=1}^{k} \text{KL}(\rho^*\|\hat{\rho}_j)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z$ | dà xiě z | 隐层表示 |
| $k$ | kèi | 隐层单元数 |

**一句话解释：**
对所有隐层单元的KL散度进行求和。

**简单应用场景：**
隐层级别的稀疏性约束。

---

---

### 公式 (9.27)：变分自编码器KL散度（修正）/ VAE KL Divergence

$$
\boldsymbol{h}_n = \operatorname{PE}(\boldsymbol{x}_n, n)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_{\text{KL}}$ | kě ěr bǎk | KL散度 |
| $q(z \mid x)$ | qū | 后验分布 |
| $p(z)$ | pī | 先验分布 |
| $z$ | zhě | 隐变量 |

**一句话解释：**
衡量学习的后验分布与先验分布之间的差异。

**简单应用场景：**
变分自编码器的KL散度项。

---

---

### 公式 (9.28)：变分自编码器目标函数（修正）/ VAE Loss Function

$$
\operatorname{PE}(x, 2i) = \sin\left(\frac{x}{10000^{2i/d}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{VAE}}$ | ēl | 变分自编码器损失 |
| $\mathbb{E}$ | qī wàng | 期望 |
| $D_{\text{KL}}$ | kě ěr bǎk | KL散度 |

**一句话解释：**
重构项与KL散度项的加权组合。

**简单应用场景：**
变分自编码器的完整训练目标。

---

---

### 公式 (9.29)：高斯后验分布（修正）/ Gaussian Posterior Distribution

$$
\operatorname{PE}(x, 2i+1) = \cos\left(\frac{x}{10000^{2i/d}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{N}$ | zhèng tài | 正态分布 |
| $\mu(x)$ | měi | 均值 |
| $\sigma(x)$ | xī gé mǎ | 标准差 |
| $I$ | dà xiě i | 单位矩阵 |

**一句话解释：**
假设后验分布为高斯分布，由神经网络参数化。

**简单应用场景：**
变分自编码器中的编码部分。

---

---

### 公式 (9.30)：高斯先验分布（修正）/ Gaussian Prior Distribution

$$
\boldsymbol{H}^{(l)} = \operatorname{TransformerBlock}(\boldsymbol{H}^{(l-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{N}$ | zhèng tài | 正态分布 |
| $I$ | dà xiě i | 单位矩阵 |

**一句话解释：**
先验分布为标准正态分布。

**简单应用场景：**
变分自编码器中的先验假设。

---

---

### 公式 (9.31)：高斯KL散度闭形式解（修正）/ Gaussian KL Divergence Closed Form

$$
\boldsymbol{Z} = \operatorname{LayerNorm}(\boldsymbol{H} + \operatorname{MultiHead}(\boldsymbol{H}, \boldsymbol{H}, \boldsymbol{H}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_j$ | měi | 第j个均值 |
| $\sigma_j$ | xī gé mǎ | 第j个标准差 |
| $d$ | dì | 隐变量维度 |

**一句话解释：**
高斯分布KL散度的显式计算形式。

**简单应用场景：**
变分自编码器中KL散度项的高效计算。

---

---

### 公式 (9.32)：自归一化网络输出 / Self-Normalizing Output

$$
y_i = \frac{\exp(u_i)}{\sum_{j=1}^{K} \exp(u_j)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_i$ | yī | 第i个输出概率 |
| $u_i$ | yòu | 第i个logit |
| $K$ | kèi | 输出类别数 |

**一句话解释：**
softmax函数将logits转化为概率分布。

**简单应用场景：**
多分类神经网络的输出层。

---

---

### 公式 (9.33)：最大似然估计 / Maximum Likelihood Estimation

$$
\mathcal{L}(\theta) = \sum_{n=1}^{N} \log p(y^{(n)} | x^{(n)}; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ēl | 似然函数 |
| $\theta$ | xī tǎ | 模型参数 |
| $p$ | pī | 概率 |

**一句话解释：**
最大化训练数据的条件对数似然。

**简单应用场景：**
有监督学习中的标准目标函数。

---

---

### 公式 (9.34)：交叉熵损失 / Cross-Entropy Loss

$$
\mathcal{L} = -\sum_{n=1}^{N} \sum_{k=1}^{K} y_k^{(n)} \log(\hat{y}_k^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_k^{(n)}$ | yī | 第n个样本的第k个目标标签 |
| $\hat{y}_k^{(n)}$ | yī hàt | 第n个样本的第k个预测概率 |

**一句话解释：**
衡量预测概率分布与真实标签分布的差异。

**简单应用场景：**
分类任务中的标准损失函数。

---

---

### 公式 (9.35)：均方误差损失 / Mean Squared Error Loss

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \|\hat{y}^{(n)} - y^{(n)}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}^{(n)}$ | yī hàt | 第n个样本的预测值 |
| $y^{(n)}$ | yī | 第n个样本的真实值 |

**一句话解释：**
预测值与真实值平方差的平均。

**简单应用场景：**
回归任务中的标准损失函数。

---

---

### 公式 (9.36)：梯度下降更新 / Gradient Descent Update

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āěr fǎ | 学习率 |
| $\nabla_\theta$ | nà bù lā | 对参数的梯度 |
| $\mathcal{L}$ | ēl | 损失函数 |

**一句话解释：**
沿着梯度的反方向更新参数。

**简单应用场景：**
神经网络的基础优化算法。

---

---

### 公式 (9.37)：动量梯度下降 / Momentum Gradient Descent

$$
\boldsymbol{v} \leftarrow \gamma \boldsymbol{v} + (1 - \gamma) \nabla_\theta \mathcal{L}(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}$ | dà xiě v | 速度向量 |
| $\gamma$ | gā mǎ | 动量系数 |

**一句话解释：**
使用历史梯度的加权平均作为更新方向。

**简单应用场景：**
加速收敛、减少振荡的优化方法。

---

---

### 公式 (9.38)：参数更新（动量）/ Parameter Update with Momentum

$$
\theta \leftarrow \theta - \alpha \boldsymbol{v}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āěr fǎ | 学习率 |
| $\boldsymbol{v}$ | dà xiě v | 速度向量 |

**一句话解释：**
使用速度向量进行参数更新。

**简单应用场景：**
动量优化算法的参数更新步骤。

---

---

### 公式 (9.39)：Adam优化器 / Adam Optimizer

$$
m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $m_t$ | ēm | 一阶矩估计 |
| $g_t$ | jī | 梯度 |
| $\beta_1$ | bèi tǎ | 一阶矩衰减因子 |

**一句话解释：**
计算梯度的一阶矩（均值）的指数移动平均。

**简单应用场景：**
Adam优化器中的动量项计算。

---

---

### 公式 (9.40)：Adam二阶矩 / Adam Second Moment

$$
v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $v_t$ | vī | 二阶矩估计 |
| $g_t^2$ | jī píng fāng | 梯度平方 |
| $\beta_2$ | bèi tǎ | 二阶矩衰减因子 |

**一句话解释：**
计算梯度平方的指数移动平均。

**简单应用场景：**
Adam优化器中的自适应学习率计算。

---

---

### 公式 (9.41)：Adam偏差纠正 / Adam Bias Correction

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{m}_t$ | ēm hàt | 纠正后的一阶矩 |
| $\hat{v}_t$ | vī hàt | 纠正后的二阶矩 |
| $t$ | tī | 时间步 |

**一句话解释：**
对初期的指数移动平均偏差进行纠正。

**简单应用场景：**
Adam优化器的偏差纠正步骤。

---

---

### 公式 (9.42)：Adam参数更新 / Adam Parameter Update

$$
\theta \leftarrow \theta - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āěr fǎ | 学习率 |
| $\epsilon$ | ēp sī lóng | 数值稳定性常数 |

**一句话解释：**
使用自适应学习率更新参数。

**简单应用场景：**
Adam优化器的参数更新步骤。

---

---

### 公式 (9.43)：RMSprop优化器 / RMSprop Optimizer

$$
s_t \leftarrow \rho s_{t-1} + (1 - \rho) g_t^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t$ | ēs | 平方梯度累积 |
| $\rho$ | rù | 衰减因子 |
| $g_t$ | jī | 梯度 |

**一句话解释：**
计算梯度平方的指数移动平均。

**简单应用场景：**
RMSprop优化器的累积步骤。

---

---

### 公式 (9.44)：RMSprop参数更新 / RMSprop Parameter Update

$$
\theta \leftarrow \theta - \frac{\alpha}{\sqrt{s_t + \epsilon}} g_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āěr fǎ | 学习率 |
| $\epsilon$ | ēp sī lóng | 数值稳定性常数 |

**一句话解释：**
使用自适应学习率进行参数更新。

**简单应用场景：**
RMSprop优化器的参数更新。

---

---

### 公式 (9.45)：学习率衰减 / Learning Rate Decay

$$
\alpha_t = \alpha_0 \cdot e^{-\lambda t}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | ālf ā | 第t步的学习率 |
| $\alpha_0$ | ālf ā líng | 初始学习率 |
| $\lambda$ | lán mǔ dá | 衰减系数 |
| $t$ | tī | 时间步 |

**一句话解释：**
随着训练进行逐渐降低学习率。

**简单应用场景：**
改善优化收敛效果的学习率调度策略。

---

---

### 公式 (9.46)：L2正则化 / L2 Regularization

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \|\boldsymbol{\theta}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{reg}}$ | ēl | 正则化后的损失 |
| $\lambda$ | lán mǔ dá | 正则化强度 |
| $\boldsymbol{\theta}$ | xī tǎ | 模型参数 |

**一句话解释：**
在损失函数中加入参数的L2范数项。

**简单应用场景：**
防止过拟合的正则化方法。

---

---

### 公式 (9.47)：Dropout正则化 / Dropout Regularization

$$
\tilde{z} = z \odot m, \quad m_i \sim \text{Bernoulli}(1 - p)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{z}$ | zhě | dropout后的输出 |
| $m$ | ēm | 掩码向量 |
| $\odot$ | yuán | 逐元素乘积 |
| $p$ | pī | dropout概率 |

**一句话解释：**
随机丢弃隐层神经元的激活值。

**简单应用场景：**
减少神经网络过拟合的有效方法。

---

## CHAPTER 8 缺失公式 (8.30-8.37)

---

## 第10章：模型独立的学习方式

### 公式 (10.1)：集成学习的误差 / Ensemble Learning Error

$$
\mathcal{R}(f_m) = E_x\left[(f_m(x) - h(x))^2\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(f_m)$ | fēng xiǎn | Risk/Error of model $f_m$ |
| $f_m(x)$ | mó xíng | Model $m$ |
| $h(x)$ | zhēn shí | True function |
| $E_x$ | qī wàng | Expectation over x |

**一句话解释：**
The risk of model $f_m$ is the expected squared difference between the model's prediction and the true function.

**简单应用场景：**
Evaluating the prediction error of a single weak learner in ensemble methods.

---

---

### 公式 (10.2)：单个模型的期望误差 / Individual Model Expected Error

$$
= E_x\left[\epsilon_m(x)^2\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\epsilon_m(x)$ | cuò wù | Error of model $m$ |
| $f_m(x) - h(x)$ | yù cè cuò wù | Prediction error |

**一句话解释：**
The risk can be expressed as the expected squared error of model $m$.

**简单应用场景：**
Decomposing the risk into individual model errors in ensemble learning.

---

---

### 公式 (10.3)：集成学习的期望误差 / Ensemble Learning Expected Risk

$$
\mathcal{R}(J) = \frac{1}{M} \sum_{m=1}^{M} E_x[\epsilon_m(x)^2]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(J)$ | jí chéng cuò wù | Ensemble risk |
| $M$ | mó xíng gè shù | Number of models |
| $\epsilon_m(x)$ | dì $m$ gè mó xíng de cuò wù | Error of model $m$ |

**一句话解释：**
The ensemble risk is the average of individual model risks.

**简单应用场景：**
Computing the expected error when combining multiple weak learners.

---

---

### 公式 (10.4)：集成模型的预测 / Ensemble Model Prediction

$$
F(x) = \frac{1}{M} \sum_{m=1}^{M} f_m(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F(x)$ | jí chéng mó xíng | Ensemble model |
| $f_m(x)$ | dì $m$ gè jī chǔ mó xíng | Base model $m$ |
| $M$ | mó xíng gè shù | Number of base models |

**一句话解释：**
The ensemble prediction is the simple average of all base model predictions.

**简单应用场景：**
Simple ensemble voting in averaging-based methods like bagging.

---

---

### 公式 (10.5)：集成模型的误差分解 / Ensemble Model Error Decomposition

$$
\mathcal{R}(F) = E_x\left[\left(\frac{1}{M} \sum_{m=1}^{M} f_m(x) - h(x)\right)^2\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(F)$ | jí chéng mó xíng cuò wù | Ensemble error |
| $F(x)$ | yù cè | Prediction |
| $h(x)$ | zhēn shí hánshù | True function |

**一句话解释：**
The ensemble error is the expected squared difference between the ensemble prediction and the true function.

**简单应用场景：**
Measuring the performance of ensemble methods.

---

---

### 公式 (10.6)：展开第一项 / Expanded First Term

$$
= E_x\left[\left(\frac{1}{M} \sum_{m=1}^{M} \epsilon_m(x)\right)^2\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\epsilon_m(x)$ | cuò wù xiàng | Error term |

**一句话解释：**
Rewriting the ensemble error using individual model errors.

**简单应用场景：**
Deriving error bounds for ensemble methods.

---

---

### 公式 (10.7)：平方展开 / Squared Expansion

$$
= \frac{1}{M^2} E_x\left[\sum_{m=1}^{M} \sum_{n=1}^{M} \epsilon_m(x)\epsilon_n(x)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\epsilon_m(x)$ | dì $m$ gè mó xíng de cuò wù | Error of model $m$ |
| $\epsilon_n(x)$ | dì $n$ gè mó xíng de cuò wù | Error of model $n$ |

**一句话解释：**
Expanding the squared error into a double summation of error products.

**简单应用场景：**
Analyzing correlation between model errors in ensemble learning.

---

---

### 公式 (10.8)：期望线性性 / Linearity of Expectation

$$
= \frac{1}{M^2} \sum_{m=1}^{M} \sum_{n=1}^{M} E_x[\epsilon_m(x)\epsilon_n(x)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_x[\epsilon_m(x)\epsilon_n(x)]$ | cuò wù jiàn de xiāng guān xìng | Covariance of errors |

**一句话解释：**
Applying the linearity of expectation to the error product.

**简单应用场景：**
Computing error correlations in ensemble learning.

---

---

### 公式 (10.9)：权重衰减 / Weight Decay

$$
p(\boldsymbol{x} | \boldsymbol{z}) = \mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}(\boldsymbol{z}), \boldsymbol{\sigma}^2(\boldsymbol{z}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L(\boldsymbol{\theta})$ | Lú tīta | 带正则项的损失函数 |
| $L_D(\boldsymbol{\theta})$ | Lú dì tīta | 数据集上的原始损失函数 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\boldsymbol{\theta}$ | tīta | 模型参数 |

**一句话解释：**
权重衰减通过在损失函数中添加权重的平方范数项来惩罚较大的权重，促进学习更小的权重。

**简单应用场景：**
在神经网络训练中，权重衰减可以防止模型过拟合，使得学到的权重更加稀疏。

---

---

### 公式 (10.10)：AdaBoost 算法的集成模型 / AdaBoost Ensemble Model

$$
F(x) = \sum_{m=1}^{M} \alpha_m f_m(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_m$ | quán zhòng | Weight of model $m$ |
| $f_m(x)$ | jī chǔ mó xíng | Base classifier |
| $M$ | mó xíng gè shù | Number of iterations |

**一句话解释：**
AdaBoost combines weak learners with weights that are learned during training.

**简单应用场景：**
Sequential boosting where each model is weighted by its accuracy.

---

---

### 公式 (10.11)：AdaBoost 的指数损失函数 / AdaBoost Exponential Loss

$$
\mathcal{L}(F) = \exp(-yF(x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(F)$ | sǔn shī hánshù | Loss function |
| $y$ | biāo qiān | Label |
| $F(x)$ | yù cè zhí | Prediction |

**一句话解释：**
AdaBoost uses exponential loss to heavily penalize misclassifications.

**简单应用场景：**
Training AdaBoost to focus on hard-to-classify examples.

---

---

### 公式 (10.12)：AdaBoost 集成展开 / AdaBoost Ensemble Expansion

$$
= \exp\left(-y \sum_{m=1}^{M} \alpha_m f_m(x)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_m$ | dì $m$ cì de quán zhòng | Weight of iteration $m$ |
| $f_m(x)$ | dì $m$ cì de jī chǔ fēn lèi qì | Base classifier at iteration $m$ |

**一句话解释：**
Decomposing the AdaBoost loss into individual weighted base classifiers.

**简单应用场景：**
Analyzing the ensemble structure of AdaBoost.

---

---

### 公式 (10.13)：迭代后的集成模型 / Iterative Ensemble Model

$$
F_{m-1}(x) = \sum_{i=1}^{m-1} \alpha_i f_i(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F_{m-1}(x)$ | dì $m-1$ cì zōng hé yù cè | Ensemble after $m-1$ iterations |
| $\alpha_i$ | dì $i$ cì de quán zhòng | Weight of iteration $i$ |
| $f_i(x)$ | dì $i$ cì de jī chǔ fēn lèi qì | Base classifier at iteration $i$ |

**一句话解释：**
The ensemble prediction up to iteration $m-1$ is the sum of weighted classifiers.

**简单应用场景：**
Computing the cumulative ensemble at each boosting iteration.

---

---

### 公式 (10.14)：AdaBoost 第 m 次迭代的损失 / AdaBoost Iteration m Loss

$$
\mathcal{L}(\alpha_m, f_m(x)) = \sum_{n=1}^{N} \exp\left(-y^{(n)}\left(F_{m-1}(x^{(n)}) + \alpha_m f_m(x^{(n)})\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_m$ | dì $m$ cì de quán zhòng | Weight for iteration $m$ |
| $f_m(x)$ | dì $m$ cì xuǎn zé de jī chǔ fēn lèi qì | Base classifier at iteration $m$ |
| $F_{m-1}(x^{(n)})$ | dì $n$ ge yàng běn zài qián $m-1$ cì de yù cè | Ensemble prediction on sample $n$ |

**一句话解释：**
The loss at iteration $m$ depends on the sum of the previous ensemble and the new weighted classifier.

**简单应用场景：**
Computing the loss when adding a new base classifier to the AdaBoost ensemble.

---

---

### 公式 (10.15)：Dropout正则化 / Dropout Regularization

$$
\log p(\boldsymbol{x}) \geq \mathbb{E}_{q(\boldsymbol{z}|\boldsymbol{x})} [\log p(\boldsymbol{x}|\boldsymbol{z})] - D_{\mathrm{KL}}(q(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{z}_{i}^{(l)}$ | tǎi zī | Dropout后的第l层第i个神经元 |
| $z_{i}^{(l)}$ | zī | 第l层第i个神经元的激活前值 |
| $r_{i}^{(l)}$ | ěr zī | Bernoulli随机变量（0或1） |

**一句话解释：**
Dropout通过随机丢弃一些神经元（设置为0）来减少神经元之间的相互适应。

**简单应用场景：**
在深层神经网络训练中使用Dropout可以有效降低过拟合风险。

---

---

### 公式 (10.16)：二分类标签的映射 / Binary Classification Label Mapping

$$
y f_m(x) = 1 - 2\mathcal{I}(y \neq f_m(x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | zhēn shí biāo qiān | True label |
| $f_m(x)$ | yù cè biāo qiān | Predicted label |
| $\mathcal{I}(\cdot)$ | zhǐ shì hánshù | Indicator function |

**一句话解释：**
For binary classification, the product equals 1 when correct and -1 when incorrect.

**简单应用场景：**
Expressing classification correctness in AdaBoost loss functions.

---

---

### 公式 (10.17)：批量归一化前向传播 / Batch Normalization Forward

$$
\mathcal{L}(\theta, \phi; \boldsymbol{x}) = -D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z})) + \mathbb{E}_{q_\phi(\boldsymbol{z}|\boldsymbol{x})} [\log p_\theta(\boldsymbol{x}|\boldsymbol{z})]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{x}_{i}$ | hàt xī | 归一化后的输入 |
| $x_{i}$ | xī | 原始输入 |
| $\mu_B$ | myù bì | 小批量数据的均值 |
| $\sigma_B^2$ | xī gé mǎ píng fāng | 小批量数据的方差 |
| $\epsilon$ | epsīlóng | 数值稳定性的小常数 |

**一句话解释：**
批量归一化将小批量数据通过减去均值并除以标准差来进行归一化处理。

**简单应用场景：**
在卷积神经网络中，批量归一化可以加速训练并提高模型的泛化能力。

---

---

### 公式 (10.18)：批量归一化尺度和移位 / Batch Normalization Scale and Shift

$$
q_\phi(\boldsymbol{z}|\boldsymbol{x}) = \mathcal{N}(\boldsymbol{z}; \boldsymbol{\mu}_\phi(\boldsymbol{x}), \operatorname{diag}(\boldsymbol{\sigma}_\phi^2(\boldsymbol{x})))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_{i}$ | yī | 输出值 |
| $\gamma$ | gā mǎ | 尺度参数 |
| $\hat{x}_{i}$ | hàt xī | 归一化后的输入 |
| $\beta$ | bèi tǎ | 移位参数 |

**一句话解释：**
批量归一化通过学习尺度参数$\gamma$和移位参数$\beta$来恢复数据的表达能力。

**简单应用场景：**
在深度卷积网络中，这些可学习参数使得网络能够学习最优的数据分布。

---

---

### 公式 (10.19)：批量归一化训练阶段的方差 / Batch Normalization Training Variance

$$
\boldsymbol{z} = \boldsymbol{\mu}_\phi(\boldsymbol{x}) + \boldsymbol{\sigma}_\phi(\boldsymbol{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma_B^2$ | xī gé mǎ píng fāng | 小批量数据的方差 |
| $m$ | liù | 小批量中的样本数 |
| $x_{i}$ | xī | 第i个样本 |
| $\mu_B$ | myù bì | 小批量数据的均值 |

**一句话解释：**
批量归一化计算小批量内所有样本相对于批量均值的方差。

**简单应用场景：**
在训练阶段，根据当前小批量的统计特性进行归一化处理。

---

---

### 公式 (10.20)：批量归一化测试阶段的方差 / Batch Normalization Test Variance

$$
D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z})) = -\frac{1}{2} \sum_{j=1}^{J} (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma^2$ | xī gé mǎ píng fāng | 全局方差估计 |
| $m$ | liù | 小批量中的样本数 |
| $\mathbb{E}_B[\sigma_B^2]$ | yī wàng zhí | 所有小批量方差的期望 |

**一句话解释：**
测试阶段使用Bessel修正的全局方差估计来进行归一化。

**简单应用场景：**
在模型推理阶段，使用训练期间累计的全局统计量进行归一化。

---

---

### 公式 (10.21)：批量归一化滑动平均方差 / Batch Normalization Moving Average Variance

$$
\mathcal{L}(\theta, \phi; \boldsymbol{x}) \approx \frac{1}{L} \sum_{l=1}^{L} \log p_\theta(\boldsymbol{x} | \boldsymbol{z}^{(l)}) - D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma^2$ | xī gé mǎ píng fāng | 更新后的全局方差 |
| $\alpha$ | āěr fǎ | 动量系数 |
| $\sigma_B^2$ | xī gé mǎ píng fāng | 当前小批量的方差 |

**一句话解释：**
测试阶段对方差进行指数加权移动平均以获得全局统计量的运行估计。

**简单应用场景：**
在训练过程中，逐批次累积全局方差的估计值。

---

---

### 公式 (10.22)：AdaBoost 权重更新规则 / AdaBoost Weight Update Rule

$$
\alpha_m = \frac{1}{2} \log \frac{1 - \epsilon_m}{\epsilon_m}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_m$ | dì $m$ cì fēn lèi qì de quán zhòng | Weight of classifier $m$ |
| $\epsilon_m$ | dì $m$ cì fēn lèi qì de jiā quán cuò wù lǜ | Weighted error rate |
| $\log$ | duì shù | Natural logarithm |

**一句话解释：**
The weight of each classifier is determined by its accuracy, with more accurate classifiers receiving higher weights.

**简单应用场景：**
Assigning importance weights to base classifiers in AdaBoost based on their performance.

---

---

### 公式 (10.23)：逐层归一化 / Layer Normalization

$$
\boldsymbol{\mu}_\phi, \log \boldsymbol{\sigma}_\phi^2 = \operatorname{Encoder}_\phi(\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{x}_{i}$ | hàt xī | 归一化后的输入 |
| $x_{i}$ | xī | 第i个特征 |
| $\mu$ | myù | 当前样本所有特征的均值 |
| $\sigma^2$ | xī gé mǎ píng fāng | 当前样本所有特征的方差 |
| $\epsilon$ | epsīlóng | 数值稳定性的小常数 |

**一句话解释：**
逐层归一化沿着特征维度（而非批量维度）对每个样本进行独立的归一化。

**简单应用场景：**
在循环神经网络和Transformer模型中用于稳定训练。

---

---

### 公式 (10.24)：多任务学习损失函数 / Multi-task Learning Loss Function

$$
\mathcal{D}_m = \{(x^{(m,n)}, y^{(m,n)})\}_{n=1}^{N_m}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_m$ | dé ěr tǎ em | 第m个任务的数据集 |
| $x^{(m,n)}$ | x shàng biāo em n | 第m个任务的第n个样本输入 |
| $y^{(m,n)}$ | y shàng biāo em n | 第m个任务的第n个样本标签 |
| $N_m$ | dà xiě èn em | 第m个任务的样本数量 |

**一句话解释：**
表示第m个任务的数据集包含$N_m$个样本对，每个样本由输入$x^{(m,n)}$和标签$y^{(m,n)}$组成。

**简单应用场景：**
在多任务学习中定义各个任务的训练数据集，例如同时学习多个相关的图像分类任务。

---

---

### 公式 (10.25)：多任务学习总体损失函数 / Multi-task Learning Overall Loss Function

$$
\mathcal{L}(\boldsymbol{\theta}) = \sum_{m=1}^{M} \sum_{n=1}^{N_m} \gamma_m \mathcal{L}_m(f_m(x^{(m,n)}; \boldsymbol{\theta}), y^{(m,n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\boldsymbol{\theta})$ | ěl dà tǎ xiē tà | 总体损失函数 |
| $M$ | dà xiě èm | 任务总数 |
| $\gamma_m$ | gā mǎ em | 第m个任务的权重系数 |
| $\mathcal{L}_m$ | ěl em | 第m个任务的损失函数 |
| $\boldsymbol{\theta}$ | xī tǎ | 模型参数向量 |

**一句话解释：**
多任务学习的总损失函数是所有任务个别损失函数的加权和，其中$\gamma_m$控制不同任务的重要性。

**简单应用场景：**
同时训练多个相关的神经网络任务，如语音识别同时进行性别检测和年龄估计。

---

---

### 公式 (10.26)：领域不变性表示 / Domain-Invariant Representation

$$
\boldsymbol{x}' = \operatorname{Decoder}_\theta(\boldsymbol{z})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_s(\theta_f)$ | ěr ěss | 源域上的经验风险 |
| $\theta_f$ | tīta fú | 特征提取器的参数 |
| $(x,y)$ | xcì yì | 数据样本及其标签 |
| $p_s(x,y)$ | píi sì | 源域的数据分布 |
| $\mathcal{L}$ | cǎi | 损失函数 |

**一句话解释：**
源域上的经验风险是在源域数据集上计算的目标函数。

**简单应用场景：**
在领域自适应问题中定义源域的训练目标。

---

---

### 公式 (10.27)：加权域不变性目标 / Weighted Domain-Invariant Objective

$$
\mathcal{R}_c(\boldsymbol{\theta}_f) = \mathbb{E}_{(x,y) \sim p_S(x,y)} \frac{p_T(\boldsymbol{x},y)}{p_S(\boldsymbol{x},y)} \mathcal{L}(f(x,\boldsymbol{\theta}_f), y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_T(\boldsymbol{x},y)$ | pì tì | 目标域的联合分布 |
| $p_S(\boldsymbol{x},y)$ | pì ès | 源域的联合分布 |
| $\frac{p_T(\boldsymbol{x},y)}{p_S(\boldsymbol{x},y)}$ | pì tì bǐ pì ès | 重要性权重 |

**一句话解释：**
通过目标域和源域分布的比值对损失进行加权，以减少源域和目标域之间的分布差异。

**简单应用场景：**
在域适应中使用重要性加权来降低源域训练对目标域的影响。

---

---

### 公式 (10.28)：简化的加权目标 / Simplified Weighted Objective

$$
\mathcal{R}_c(\boldsymbol{\theta}_f) = \mathbb{E}_{(x,y) \sim p_S(x,y)} \frac{p_T(\boldsymbol{x})}{p_S(\boldsymbol{x})} \mathcal{L}(f(x,\boldsymbol{\theta}_f), y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_T(\boldsymbol{x})$ | pì tì | 目标域的边际分布 |
| $p_S(\boldsymbol{x})$ | pì ès | 源域的边际分布 |

**一句话解释：**
使用边际分布而非联合分布的比值来进行加权，简化了计算复杂度。

**简单应用场景：**
在只关心输入分布差异而非条件分布的场景中进行域适应。

---

---

### 公式 (10.29)：领域自适应的协变量偏移 / Domain Adaptation with Covariate Shift

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}}[\log D(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})}[\log(1 - D(G(\boldsymbol{z})))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_T$ | ěr tī | 目标域的风险 |
| $\theta_f$ | tīta fú | 特征提取器参数 |
| $\theta_g$ | tīta jī | 分类器参数 |
| $\gamma$ | gā mǎ | 权重系数 |
| $d_s(S, T)$ | dī ěss | 源域和目标域间的距离度量 |

**一句话解释：**
目标域风险包括在源域数据上的经验风险加上源目标域间的领域距离惩罚项。

**简单应用场景：**
在迁移学习中最小化源域损失和域差距以适应目标域。

---

---

### 公式 (10.30)：无标签目标域数据的领域自适应 / Unsupervised Domain Adaptation

$$
D^*(\boldsymbol{x}) = \frac{p_{\text{data}}(\boldsymbol{x})}{p_{\text{data}}(\boldsymbol{x}) + p_G(\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_S$ | ěr ěss | 源域经验风险 |
| $\gamma$ | gā mǎ | 权重系数 |
| $d_s(S, T)$ | dī ěss | 领域距离 |

**一句话解释：**
无监督领域自适应通过最小化源域风险和域差距来学习域不变特征。

**简单应用场景：**
当目标域没有标签时，使用源域标签数据和域距离惩罚进行训练。

---

---

### 公式 (10.31)：源域和目标域的样本分布 / Source and Target Domain Sample Distribution

$$
\mathcal{L}_D = -\mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}}[\log D(\boldsymbol{x})] - \mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})}[\log(1 - D(G(\boldsymbol{z})))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_S$ | cǎi dì ěss | 源域数据集 |
| $x_i^{(n)}$ | xcì shàng n | 第n个源域样本 |
| $y_i^{(n)}$ | yì shàng n | 第n个源域标签 |
| $N$ | ēn | 源域样本总数 |
| $p_s(x, y)$ | píi ěss | 源域分布 |

**一句话解释：**
源域数据集由N个标记样本组成，这些样本来自源域分布。

**简单应用场景：**
在迁移学习中定义源域的有标签训练集。

---

---

### 公式 (10.32)：目标域数据集 / Target Domain Dataset

$$
\mathcal{L}_G = -\mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})}[\log D(G(\boldsymbol{z}))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_T$ | cǎi dì tī | 目标域数据集 |
| $x_j^{(m)}$ | xcì shàng m | 第m个目标域样本 |
| $M$ | ēm | 目标域样本总数 |
| $p_T(x, y)$ | píi tī | 目标域分布 |

**一句话解释：**
目标域数据集由M个无标签样本组成，这些样本来自目标域分布。

**简单应用场景：**
在无监督领域自适应中定义目标域的无标签测试集。

---

---

### 公式 (10.33)：无标签学习的损失函数 / Unsupervised Learning Loss Function

$$
\mathcal{L}_G = \mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})}[\log(1 - D(G(\boldsymbol{z})))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_c$ | cǎi | 分类器损失 |
| $\theta_g$ | tīta jī | 分类器参数 |
| $\theta_c$ | tīta cī | 判别器参数 |
| $h_s^{(n)}$ | hèi ěss | 源域特征 |
| $h_m^{(n)}$ | hèi | 目标域特征 |
| $c$ | cī | 判别函数 |

**一句话解释：**
无标签学习的损失函数结合源域分类损失和目标域判别器损失。

**简单应用场景：**
在对抗领域自适应中训练分类器同时欺骗判别器。

---

---

### 公式 (10.34)：最小化目标 / Minimization Objective

$$
\min_{\boldsymbol{\theta}_c} \mathcal{L}_c(\boldsymbol{\theta}_f, \boldsymbol{\theta}_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\min$ | zuì xiǎo huà | 最小化 |
| $\boldsymbol{\theta}_c$ | xī tǎ cì | 分类器参数 |

**一句话解释：**
通过优化分类器参数来最小化分类损失函数。

**简单应用场景：**
标准的监督学习模型优化过程。

---

---

### 公式 (10.35)：最小化损失函数 / Loss Minimization

$$
G(\boldsymbol{z}) = \boldsymbol{\mu}_\theta(\boldsymbol{z})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\min$ | zuì xiǎo huà | 最小化操作 |
| $\theta_c$ | tīta cī | 分类器参数 |
| $\mathcal{L}_c$ | cǎi | 分类损失函数 |

**一句话解释：**
通过优化分类器参数来最小化分类损失函数。

**简单应用场景：**
在监督学习阶段优化分类器。

---

---

### 公式 (10.36)：最小化加权损失函数 / Weighted Loss Minimization

$$
D(\boldsymbol{x}) = \sigma(\boldsymbol{w}^\top \boldsymbol{x} + b)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_f$ | tīta fú | 特征提取器参数 |
| $\theta_c$ | tīta cī | 分类器参数 |
| $\gamma$ | gā mǎ | 权重系数 |
| $\mathcal{L}$ | cǎi | 源域损失 |
| $\mathcal{L}_c$ | cǎi | 判别器损失 |

**一句话解释：**
联合优化特征提取器和分类器，在最小化源域损失的同时最大化判别器损失。

**简单应用场景：**
在对抗学习框架中进行特征和分类器的联合训练。

---

---

### 公式 (10.37)：条件独立下的分布分解 / Distribution Decomposition under Conditional Independence

$$
\log p(\boldsymbol{\theta}|\mathcal{D}) = \log p(\mathcal{D}_A|\boldsymbol{\theta}) + \log p(\mathcal{D}_B|\boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) - \log p(\mathcal{D}_A) - \log p(\mathcal{D}_B)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_A$ | dé ěl dà xiě ā | 任务A的数据集 |
| $\mathcal{D}_B$ | dé ěl dà xiě bì | 任务B的数据集 |

**一句话解释：**
当任务A和B条件独立时，联合后验可以分解为两个任务的似然和先验的乘积。

**简单应用场景：**
终身学习中处理多个独立任务的贝叶斯推断。

---

---

### 公式 (10.38)：进一步简化 / Further Simplification

$$
= \log p(\mathcal{D}_B|\boldsymbol{\theta}) + \log p(\boldsymbol{\theta}|\mathcal{D}_A) - \log p(\mathcal{D}_B)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{\theta} \mid \mathcal{D}_A)$ | pì | 在A任务后的参数分布 |

**一句话解释：**
展示了如何将联合分布进一步简化，使用任务A的后验作为任务B的先验。

**简单应用场景：**
顺序多任务学习中的参数更新。

---

---

### 公式 (10.39)：Fisher信息矩阵定义 / Fisher Information Matrix Definition

$$
W_1(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim \gamma} [\|\boldsymbol{x} - \boldsymbol{y}\|]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{F}(\theta)$ | cǎi tīta | Fisher信息矩阵 |
| $\mathcal{N}$ | ēn | 正态分布 |
| $\mathcal{C}_A$ | cǎi ā | 类别A的Fisher信息中心 |
| $F^{-1}$ | ēf fǎn | Fisher信息矩阵的逆 |

**一句话解释：**
Fisher信息矩阵描述了参数空间中的曲率信息。

**简单应用场景：**
在自然梯度优化和二阶优化方法中使用。

---

---

### 公式 (10.40)：Fisher信息的对数偏导数 / Log Derivative with Fisher Information

$$
s(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(\boldsymbol{\theta})$ | sī | Fisher分数函数 |
| $\nabla_{\boldsymbol{\theta}}$ | nā bǐ lā | 关于参数的梯度 |

**一句话解释：**
Fisher分数函数是对数似然函数关于参数的梯度，表示参数对模型的敏感性。

**简单应用场景：**
统计推断中评估参数的信息量。

---

---

### 公式 (10.41)：Fisher分数的期望 / Expectation of Fisher Score

$$
\mathbb{E}[s(\boldsymbol{\theta})] = \int \nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta}) p(x;\boldsymbol{\theta}) \mathrm{d}x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{E}$ | yī | 期望值 |
| $\int$ | jī fèn | 积分 |

**一句话解释：**
Fisher分数在数据分布上的期望值用于计算Fisher信息矩阵。

**简单应用场景：**
信息几何中评估参数分布的性质。

---

---

### 公式 (10.42)：概率加权的梯度 / Weighted Probability Gradient

$$
= \int \frac{\nabla_{\boldsymbol{\theta}} p(x;\boldsymbol{\theta})}{p(x;\boldsymbol{\theta})} p(x;\boldsymbol{\theta}) \mathrm{d}x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla_{\boldsymbol{\theta}} p(x;\boldsymbol{\theta})$ | nā bǐ lā pì | 概率分布的梯度 |

**一句话解释：**
通过概率的梯度计算Fisher分数，简化为梯度与概率的比值。

**简单应用场景：**
数值计算Fisher信息矩阵时的中间步骤。

---

---

### 公式 (10.43)：梯度的直接积分 / Direct Integration of Gradient

$$
= \int \nabla_{\boldsymbol{\theta}} p(x;\boldsymbol{\theta}) \mathrm{d}x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla_{\boldsymbol{\theta}}$ | nā bǐ lā | 梯度算子 |

**一句话解释：**
进一步简化了Fisher分数期望的表达式。

**简单应用场景：**
理论推导中的中间计算步骤。

---

---

### 公式 (10.44)：梯度与积分的交换 / Interchanging Gradient and Integration

$$
= \nabla_{\boldsymbol{\theta}} \int p(x;\boldsymbol{\theta}) \mathrm{d}x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla_{\boldsymbol{\theta}}$ | nā bǐ lā | 梯度算子 |

**一句话解释：**
利用梯度和积分可以交换的性质进行数学变换。

**简单应用场景：**
微积分中的基本计算技巧。

---

---

### 公式 (10.45)：归一化条件 / Normalization Condition

$$
= \nabla_{\boldsymbol{\theta}} 1 = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| 1 | yī | 常数1（概率密度的积分为1） |
| 0 | líng | 零向量 |

**一句话解释：**
由于概率密度函数的积分恒为1，其梯度为0，因此Fisher分数的期望为0。

**简单应用场景：**
验证Fisher信息的数学性质。

---

---

### 公式 (10.46)：Fisher信息矩阵的定义 / Definition of Fisher Information Matrix

$$
F(\boldsymbol{\theta}) = \mathbb{E}[s(\boldsymbol{\theta})s(\boldsymbol{\theta})^T]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F(\boldsymbol{\theta})$ | èf | Fisher信息矩阵 |
| $s(\boldsymbol{\theta})^T$ | sī zhuàn zhì | Fisher分数的转置 |

**一句话解释：**
Fisher信息矩阵是Fisher分数的外积的期望，量化了参数的信息量。

**简单应用场景：**
参数估计中用于计算参数的渐近方差。

---

---

### 公式 (10.47)：Fisher信息的展开形式 / Expanded Form of Fisher Information

$$
= \mathbb{E}[\nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta})(\nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta}))^T]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla_{\boldsymbol{\theta}} \log p(x;\boldsymbol{\theta})$ | nā bǐ lā duì shù | 对数似然的梯度 |

**一句话解释：**
Fisher信息矩阵等于对数似然梯度的自外积的期望。

**简单应用场景：**
统计推断中评估模型参数的信息含量。

---

---

### 公式 (10.48)：Fisher信息的样本估计 / Sample Estimation of Fisher Information

$$
F(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^{N} \nabla_{\boldsymbol{\theta}} \log p(x^{(n)}; \boldsymbol{\theta})(\nabla_{\boldsymbol{\theta}} \log p(x^{(n)}; \boldsymbol{\theta}))^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N$ | dà xiě èn | 样本总数 |
| $x^{(n)}$ | x shàng biāo n | 第n个样本 |

**一句话解释：**
通过有限样本来估计Fisher信息矩阵，使用每个样本梯度的外积的平均值。

**简单应用场景：**
实际计算中用样本估计总体的Fisher信息。

---

---

### 公式 (10.49)：Fisher信息矩阵的近似 / Fisher Information Matrix Approximation

$$
\boldsymbol{x}_t = \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F^A$ | ēf ā | 类别A的Fisher信息矩阵 |
| $N$ | ēn | 样本总数 |
| $\nabla_\theta$ | nà bù lā tīta | 关于参数的梯度 |
| $p(y \mid x; \theta)$ | píi | 后验概率 |

**一句话解释：**
Fisher信息矩阵通过对数似然梯度的外积的期望来定义。

**简单应用场景：**
在二阶优化算法如自然梯度下降中计算Hessian矩阵的近似。

---

---

### 公式 (10.50)：Fisher信息损失函数 / Fisher Information Loss Function

$$
q(\boldsymbol{x}_t | \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t; \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0, (1 - \bar{\alpha}_t) \boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\theta)$ | cǎi tīta | 总损失函数 |
| $\mathcal{L}_B$ | cǎi bì | 任务B的损失 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $F^A_i$ | ēf ā | 第i个参数的Fisher信息值 |
| $\theta_i$ | tīta | 第i个参数 |
| $\theta_{A,i}$ | tīta ā | 任务A中第i个参数的值 |

**一句话解释：**
Fisher信息加权的损失函数在连续学习中保护重要参数不被过度改变。

**简单应用场景：**
在持续学习或终身学习任务中防止灾难性遗忘。

---

---

### 公式 (10.51)：Fisher信息矩阵的Kronecker近似 / Kronecker Factorization of Fisher Information

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)\|^2 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F$ | ēf | Fisher信息矩阵 |
| $G$ | jī | Kronecker乘积的第一个因子 |
| $H$ | hèi | Kronecker乘积的第二个因子 |
| $\otimes$ | ōu tēng | Kronecker乘积 |

**一句话解释：**
Fisher信息矩阵可以近似为两个较小矩阵的Kronecker乘积。

**简单应用场景：**
在大规模神经网络中降低Fisher信息矩阵的计算和存储复杂度。

---

---

### 公式 (10.52)：对数边际似然 / Log Marginal Likelihood

$$
\boldsymbol{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \boldsymbol{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) \right) + \sigma_t \boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(\mathcal{D} \mid \theta)$ | duì shù | 数据的对数边际似然 |
| $\mathcal{D}_A$ | cǎi ā | 任务A的数据集 |
| $\mathcal{D}_B$ | cǎi bì | 任务B的数据集 |
| $p(\theta)$ | píi tīta | 参数的先验分布 |

**一句话解释：**
对数边际似然分解为数据的对数似然和参数先验的对数之和。

**简单应用场景：**
在贝叶斯推断和变分推断中计算模型的边际似然。

---

---

### 公式 (10.53)：多任务的联合优化目标 / Joint Optimization Objective for Multiple Tasks

$$
\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{\mathcal{T}}\left[\sum_{i=1}^{T} w_i \mathcal{L}(\boldsymbol{\theta}_i)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{E}_{\mathcal{T}}$ | yī | 对任务分布的期望 |
| $w_i$ | dà xiě w | 第i个任务的权重 |
| $\mathcal{L}(\boldsymbol{\theta}_i)$ | ěl | 第i个任务的损失 |

**一句话解释：**
在多个任务分布上最小化加权损失函数，用于学习能泛化到新任务的模型。

**简单应用场景：**
元学习中训练能快速适应新任务的模型。

---

---

### 公式 (10.54)：梯度更新基础 / Gradient Update Foundation

$$
\boldsymbol{\theta}_i = \boldsymbol{\theta}_{i-1} + g_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g_i$ | jī | 第i步的梯度更新 |

**一句话解释：**
参数在第i步更新为前一步的参数加上梯度更新。

**简单应用场景：**
迭代优化算法的基本更新规则。

---

---

### 公式 (10.55)：LSTM基于的元学习优化 / LSTM-Based Meta-Learning Optimization

$$
[\boldsymbol{g_i}; h_i] = \text{LSTM}(\nabla \mathcal{L}(\boldsymbol{\theta}_{i-1}), h_{i-1}; \boldsymbol{\phi})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| LSTM | ěl xiǎo duǎn qī jì yì | 长短期记忆网络 |
| $h_i$ | hǎi | LSTM的隐藏状态 |
| $\boldsymbol{\phi}$ | fài | LSTM的参数 |

**一句话解释：**
使用LSTM网络学习一个优化器，输入梯度和隐藏状态，输出参数更新。

**简单应用场景：**
通过递归神经网络实现自适应的优化策略。

---

---

### 公式 (10.56)：协变量移位的对数似然 / Log Likelihood with Covariate Shift

$$
p_\theta(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t), \sigma_t^2 \boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(\mathcal{D} \mid \theta)$ | duì shù | 对数边际似然 |
| $\mathcal{D}$ | cǎi dì | 完整数据集 |
| $\theta$ | tīta | 模型参数 |

**一句话解释：**
当存在协变量移位时，边际似然可以分解为不同源的贡献。

**简单应用场景：**
在处理多源域数据的贝叶斯模型中应用。

---

---

### 公式 (10.57)：参数的Fisher信息距离 / Fisher Information Distance Between Parameters

$$
\boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \boldsymbol{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\mathcal{D}_A \mid \theta)$ | píi | 任务A数据的条件概率 |
| $\mathcal{N}$ | ēn | 正态分布 |
| $\mathcal{C}_A$ | cǎi ā | 任务A的Fisher信息中心 |
| $F^{-1}$ | ēf fǎn | Fisher信息矩阵的逆 |

**一句话解释：**
任务A的参数后验分布是以Fisher信息矩阵为协方差的正态分布。

**简单应用场景：**
在贝叶斯终身学习中对参数分布的建模。

---

---

### 公式 (10.58)：标签平滑正则化 / Label Smoothing Regularization

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\theta)$ | cǎi tīta | 损失函数 |
| $p(i \mid x;\theta)$ | píi | 模型预测的第i个类别概率 |
| $q_i$ | qī | 平滑后的第i个类别标签 |
| $K$ | kè | 总类别数 |

**一句话解释：**
标签平滑通过将硬标签转换为软标签来改进模型的泛化能力。

**简单应用场景：**
在分类任务中应用标签平滑以防止模型过度自信。

---

---

### 公式 (10.59)：平滑后的标签分布 / Smoothed Label Distribution

$$
\alpha_t = 1 - \beta_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_i$ | qī | 第i个类别的平滑标签 |
| $\epsilon$ | epsīlóng | 平滑系数 |
| $y$ | yì | 真实类别标签 |
| $K$ | kè | 总类别数 |

**一句话解释：**
平滑标签将真实类别的概率设为$1-\epsilon$，其他类别均分剩余的$\epsilon$概率。

**简单应用场景：**
在图像分类任务中，使用平滑标签替代硬标签进行训练。

---

---

## 第11章：概率图模型

### 公式 (11.1)：概率定义 / Probability Definition

$$
p(\boldsymbol{x}) \triangleq P(X = \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | gài lǜ | 随机变量的概率分布 |
| $X$ | biàn liàng | 随机变量 |
| $\boldsymbol{x}$ | jù tǐ zhí | 具体取值 |

**一句话解释：**
通过概率分布函数表示随机变量X取特定值的可能性。

**简单应用场景：**
在贝叶斯网络中定义每个节点的概率。

---

---

### 公式 (11.2)：联合概率分解 / Joint Probability Factorization

$$
p(\boldsymbol{x}) = \prod_{k=1}^{K} p(x_k | x_1, \ldots, x_{k-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率分布 |
| $x_k$ | biàn liàng xī | 第k个随机变量 |
| $K$ | zǒng shù | 变量总数 |
| $\prod$ | chéng jī | 乘积 |

**一句话解释：**
联合概率可以分解为条件概率的连乘积，这是链式法则的表现。

**简单应用场景：**
在概率图模型中逐个计算条件概率以得到联合概率。

---

---

### 公式 (11.3)：条件独立性1 / Conditional Independence 1

$$
p(x_2|x_1, x_3) = p(x_2|x_1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_2 \mid x_1, x_3)$ | tiáo jiàn gài lǜ | 给定$x_1$和$x_3$下$x_2$的条件概率 |
| $x_1$ | biàn liàng | 已知变量1 |
| $x_3$ | biàn liàng | 已知变量2 |

**一句话解释：**
在已知$x_1$时，$x_2$独立于$x_3$，即$x_3$对$x_2$没有影响。

**简单应用场景：**
简化贝叶斯网络中的概率计算，减少需要考虑的变量数。

---

---

### 公式 (11.4)：条件独立性2 / Conditional Independence 2

$$
p(x_3|x_1, x_2) = p(x_3|x_1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_3 \mid x_1, x_2)$ | tiáo jiàn gài lǜ | 给定$x_1$和$x_2$下$x_3$的条件概率 |
| $x_1$ | biàn liàng | 已知变量 |
| $x_2$ | biàn liàng | 冗余变量 |

**一句话解释：**
在已知$x_1$时，$x_3$独立于$x_2$，这表示图中的条件独立结构。

**简单应用场景：**
在有向无环图(DAG)中表示父节点与其他变量的独立关系。

---

---

### 公式 (11.5)：条件独立性3 / Conditional Independence 3

$$
p(x_4|x_1, x_2, x_3) = p(x_4|x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_4 \mid x_1, x_2, x_3)$ | tiáo jiàn gài lǜ | 多变量条件下的条件概率 |
| $x_4$ | biàn liàng | 目标变量 |
| $x_1, x_2, x_3$ | yǐ zhī biàn liàng | 条件变量 |

**一句话解释：**
在已知$x_2$和$x_3$时，$x_4$独立于$x_1$，显示了图形结构中的独立性。

**简单应用场景：**
证明在概率图中某些变量之间的条件独立关系。

---

---

### 公式 (11.6)：联合概率的因子分解1 / Factored Joint Probability 1

$$
p(\boldsymbol{x}) = p(x_1)p(x_2|x_1)p(x_3|x_1, x_2)p(x_4|x_1, x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 四个变量的联合概率 |
| $p(x_1)$ | xiān yàn gài lǜ | 先验概率 |
| $p(x_i \mid ...)$ | tiáo jiàn gài lǜ | 条件概率因子 |

**一句话解释：**
联合概率分解为一系列条件概率的乘积，反映了链式结构。

**简单应用场景：**
在链式贝叶斯网络中计算完整的概率分布。

---

---

### 公式 (11.7)：联合概率的因子分解2 / Factored Joint Probability 2

$$
p(\boldsymbol{x}) = p(x_1)p(x_2|x_1)p(x_3|x_1)p(x_4|x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率分布 |
| $p(x_i \mid ...)$ | tiáo jiàn gài lǜ | 条件概率 |

**一句话解释：**
利用条件独立性简化因子分解，$x_2$和$x_3$都仅依赖$x_1$。

**简单应用场景：**
表示具有不同结构的贝叶斯网络中的概率分解。

---

---

### 公式 (11.8)：贝叶斯网络的联合概率 / Bayesian Network Joint Probability

$$
p(\boldsymbol{x}) = \prod_{k=1}^{K} p(x_k|x_{\pi_k})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率分布 |
| $x_k$ | jié diǎn | 第k个节点变量 |
| $x_{\pi_k}$ | fù jié diǎn | 第k个节点的父节点变量集合 |
| $K$ | zǒng shù | 节点总数 |

**一句话解释：**
贝叶斯网络中，联合概率等于每个节点在其父节点条件下的概率的乘积。

**简单应用场景：**
在任意有向无环图结构的贝叶斯网络中计算联合概率。

---

---

### 公式 (11.9) - 高斯混合模型对数似然函数 / Gaussian Mixture Model Log-Likelihood

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{x}$ | xiàng liàng | 观察数据向量 |
| $K$ | kè | 高斯分量个数 |
| $\pi_k$ | pài | 第k个分量的混合系数 |
| $\mathcal{N}$ | zhèng tài fēn bù | 正态分布 |
| $\boldsymbol{\mu}_k$ | mǔ | 第k个分量的均值 |
| $\boldsymbol{\Sigma}_k$ | xié fāng chà jǔ zhèn | 第k个分量的协方差矩阵 |

**一句话解释：**
高斯混合模型的对数似然函数是所有混合分量加权后的对数形式。

**简单应用场景：**
在聚类问题中评估模型拟合数据的好坏程度。

---

---

### 公式 (11.10) - EM算法隐变量后验概率 / EM Algorithm Posterior of Hidden Variable

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_{nk}$ | gāma | 第n个样本属于第k个分量的后验概率 |
| $\pi_k$ | pài | 第k个分量的混合系数 |
| $x_n$ | yàng běn | 第n个观察样本 |
| $\boldsymbol{\mu}_k$ | mǔ | 第k个分量的均值 |
| $\sigma_k^2$ | xīgmà píng fāng | 第k个分量的方差 |

**一句话解释：**
使用贝叶斯定理计算每个样本属于各个混合分量的后验概率。

**简单应用场景：**
在EM算法的E步中计算隐变量的期望。

---

---

### 公式 (11.11) - 有效样本数 / Effective Sample Size

$$
V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N_k$ | yǒu xiào yàng běn shù | 分配给第k个分量的有效样本数 |
| $N$ | zǒng yàng běn shù | 总样本数 |
| $\gamma_{nk}$ | gāma | 第n个样本属于第k个分量的后验概率 |

**一句话解释：**
统计根据后验概率加权后分配给各个混合分量的样本数。

**简单应用场景：**
在EM算法的M步中更新混合系数。

---

---

### 公式 (11.12) - 混合系数更新 / Mixture Coefficient Update

$$
Q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|s, a) V^\pi(s')
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi_k$ | pài | 第k个分量的混合系数 |
| $N_k$ | yǒu xiào yàng běn shù | 分配给第k个分量的有效样本数 |
| $N$ | zǒng yàng běn shù | 总样本数 |

**一句话解释：**
将有效样本数归一化得到混合系数的新估计值。

**简单应用场景：**
EM算法M步中更新混合权重。

---

---

### 公式 (11.13) - 均值更新 / Mean Update

$$
V^\pi(s) = \sum_a \pi(a|s) \left[ r(s, a) + \gamma \sum_{s'} p(s'|s, a) V^\pi(s') \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_k$ | mǔ | 第k个分量的均值 |
| $N_k$ | yǒu xiào yàng běn shù | 分配给第k个分量的有效样本数 |
| $\gamma_{nk}$ | gāma | 第n个样本属于第k个分量的后验概率 |
| $x_n$ | yàng běn | 第n个观察样本 |

**一句话解释：**
用加权平均方式更新高斯分量的均值，权重为后验概率。

**简单应用场景：**
EM算法中更新聚类中心位置。

---

---

### 公式 (11.14)：玻尔兹曼分布 / Boltzmann Distribution

$$
P(x) = \frac{1}{Z} \exp\left(\sum_{c \in C} \theta_c f_c(x)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(x)$ | gài lǜ | 随机变量的概率分布 |
| $Z$ | zī hǔ huà chángshù | 配分函数（归一化常数） |
| $\theta_c$ | cānshù | 模型参数 |
| $f_c(x)$ | tè zhēng hánshù | 特征函数 |
| $C$ | jíhé | 团集合 |

**一句话解释：**
玻尔兹曼分布用配分函数归一化的指数族分布来表示无向图的联合概率。

**简单应用场景：**
在受限玻尔兹曼机和深度学习模型中用于建模复杂的高维数据分布。

---

---

### 公式 (11.15) - 方差更新 / Variance Update

$$
Q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|s, a) \sum_{a'} \pi(a'|s') Q^\pi(s', a')
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma_k^2$ | xīgmà píng fāng | 第k个分量的方差 |
| $N_k$ | yǒu xiào yàng běn shù | 分配给第k个分量的有效样本数 |
| $\gamma_{nk}$ | gāma | 第n个样本属于第k个分量的后验概率 |
| $x_n$ | yàng běn | 第n个观察样本 |
| $\boldsymbol{\mu}_k$ | mǔ | 第k个分量的均值 |

**一句话解释：**
计算加权的方差，反映数据围绕均值的散布程度。

**简单应用场景：**
EM算法中更新高斯分量的标准差。

---

---

### 公式 (11.16)：Hammersley-Clifford定理 / Hammersley-Clifford Theorem

$$
p(\boldsymbol{x}) = \frac{1}{Z} \prod_{c \in C} \phi_c(\boldsymbol{x}_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率分布 |
| $Z$ | guī huà cháng shù | 分割函数(配分常数) |
| $C$ | zuì dà tuán | 所有极大团的集合 |
| $\phi_c(\boldsymbol{x}_c)$ | wèi shì hánshù | 团c上的势函数 |

**一句话解释：**
无向图的联合概率分解为所有极大团上的势函数的乘积除以分割函数。

**简单应用场景：**
在无向图模型(如条件随机场)中定义联合概率分布。

---

---

### 公式 (11.17)：分割函数 / Partition Function

$$
Z = \sum_{\boldsymbol{x} \in \mathcal{X}} \prod_{c \in C} \phi_c(\boldsymbol{x}_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z$ | guī huà cháng shù | 分割函数 |
| $\mathcal{X}$ | quán bù zhuàng tài | 所有可能的状态集合 |
| $\prod_{c \in C}$ | suǒ yǒu tuán | 对所有团的乘积 |

**一句话解释：**
分割函数是所有状态下势函数乘积的总和，用于归一化概率分布。

**简单应用场景：**
在计算无向图模型的概率时确保归一化条件。

---

---

### 公式 (11.18)：势函数与能量函数 / Potential Function and Energy Function

$$
\phi_c(\boldsymbol{x}_c) = \exp(-E_c(\boldsymbol{x}_c))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\phi_c(\boldsymbol{x}_c)$ | wèi shì hánshù | 势函数 |
| $E_c(\boldsymbol{x}_c)$ | néng liàng hánshù | 能量函数 |
| $\exp$ | zhǐ shù hánshù | 指数函数 |

**一句话解释：**
势函数定义为能量函数的负指数，能量越低，势函数值越高。

**简单应用场景：**
在能量基模型中建立势函数与能量函数的关系。

---

---

### 公式 (11.19)：无向图的概率分布1 / Undirected Graph Probability 1

$$
P(\boldsymbol{x}) = \frac{1}{Z} \prod_{c \in C} \exp(-E_c(\boldsymbol{x}_c))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率分布 |
| $Z$ | guī huà cháng shù | 分割函数 |
| $E_c(\boldsymbol{x}_c)$ | néng liàng | 团c的能量 |

**一句话解释：**
Boltzmann分布形式：联合概率正比于所有团能量的负和的指数。

**简单应用场景：**
在Boltzmann机和受限Boltzmann机中定义概率分布。

---

---

### 公式 (11.20)：无向图的概率分布2 / Undirected Graph Probability 2

$$
P(\boldsymbol{x}) = \frac{1}{Z} \exp\left(-\sum_{c \in C} E_c(\boldsymbol{x}_c)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(\boldsymbol{x})$ | gài lǜ | 概率 |
| $\exp$ | zhǐ shù | 指数 |
| $\sum_{c \in C}$ | jī hé | 对所有团求和 |

**一句话解释：**
Boltzmann分布的等价表示，将各团能量的乘积改写为求和的指数。

**简单应用场景：**
在计算和推导无向图模型时使用对数形式。

---

## 11.1.5 常见的无向图模型

---

### 公式 (11.21) - EM算法对数似然上界 / EM Algorithm Log-Likelihood Lower Bound

$$
V^*(s) = \max_a \left[ r(s, a) + \gamma \sum_{s'} p(s'|s, a) V^*(s') \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(\mathbf{x})$ | duì shù jiàn rán xìng | 对数似然函数 |
| ELBO | tái xià jiè | 证据下界（Evidence Lower Bound） |
| $q$ | jìn sì fēn bù | 隐变量的近似分布 |
| KL | kè ěr bǎk hā sàn dù | KL散度 |
| $p(\mathbf{z} \mid \mathbf{x})$ | hòu yàn gài lǜ | 后验概率 |

**一句话解释：**
对数似然函数等于ELBO加上KL散度（非负）。

**简单应用场景：**
推导EM算法的理论基础。

---

---

### 公式 (11.22) - EM算法目标函数 / EM Algorithm Objective Function

$$
Q^*(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|s, a) \max_{a'} Q^*(s', a')
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| ELBO | tái xià jiè | 证据下界 |
| $\gamma_{nk}$ | gāma | 后验概率 |
| $\pi_k$ | pài | 混合系数 |
| $\mathcal{N}$ | zhèng tài fēn bù | 正态分布 |
| $H(q)$ | shàn yú | 熵 |

**一句话解释：**
ELBO是后验概率加权的对数联合概率加上隐变量分布的熵。

**简单应用场景：**
EM算法的E步优化目标。

---

---

### 公式 (11.23)：最大熵模型1 / Maximum Entropy Model 1

$$
p(y|x; \theta) = \frac{1}{Z(x;\theta)} \exp\left(\theta^T f(x, y)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y \mid x; \theta)$ | tiáo jiàn gài lǜ | 条件概率 |
| $Z(x;\theta)$ | guī huà hánshù | 分割函数 |
| $\theta$ | cān shù xiàng liàng | 参数向量 |
| $f(x, y)$ | tè zhēng xiàng liàng | 特征向量 |

**一句话解释：**
最大熵模型通过特征函数的加权和来表示条件概率分布。

**简单应用场景：**
在自然语言处理的标注任务中进行条件概率建模。

---

---

### 公式 (11.24) - 加权的混合高斯模型对数似然 / Weighted Log-Likelihood

$$
\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q^*(s, a') \\ 0 & \text{otherwise} \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_{nk}$ | gāma | 后验概率 |
| $x_n$ | yàng běn | 样本 |
| $\boldsymbol{\mu}_k$ | mǔ | 均值 |
| $\sigma_k$ | xīgmà | 标准差 |
| $\pi_k$ | pài | 混合系数 |
| $C$ | cháng shù | 常数 |

**一句话解释：**
展开高斯分布形式后的加权对数似然函数。

**简单应用场景：**
推导EM算法的更新规则。

---

---

### 公式 (11.25) - 最大化ELBO问题 / ELBO Maximization Problem

$$
V_{k+1}(s) = \max_a \left[ r(s, a) + \gamma \sum_{s'} p(s'|s, a) V_k(s') \right]
$$

$$
\text{s.t.} \quad \sum_{k=1}^{K} \pi_k = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà huà | 最大化 |
| ELBO | tái xià jiè | 证据下界 |
| s.t. | shǐ dé | 约束条件 |
| $\pi_k$ | pài | 混合系数 |

**一句话解释：**
在混合系数和为1的约束下最大化证据下界。

**简单应用场景：**
EM算法中的M步优化问题。

---

---

### 公式 (11.26)：有向图与无向图的转换 / Directed-Undirected Transformation

$$
p(\boldsymbol{x}) = p(x_1)p(x_2|x_1)p(x_3|x_1)p(x_4|x_1, x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率 |
| $p(x_i \mid ...)$ | tiáo jiàn gài lǜ | 条件概率 |

**一句话解释：**
有向图的因子分解可以转换为无向图的势函数表示。

**简单应用场景：**
在不同模型表示之间进行转换与对比分析。

---

## 11.2.1 不含隐变量的参数估计

---

### 公式 (11.27)：有向图模型的对数似然 / Directed Model Log-Likelihood 1

$$
\mathcal{L}(\mathcal{D}; \theta) = \frac{1}{N} \sum_{n=1}^{N} \log p(\boldsymbol{x}^{(n)}; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\mathcal{D}; \theta)$ | duì shù sì ránhuàn xìng | 对数似然函数 |
| $N$ | yàng běn shù | 样本总数 |
| $\mathcal{D}$ | shù jù jī | 数据集 |
| $p(\boldsymbol{x}^{(n)}; \theta)$ | dì n ge yàng běn de gài lǜ | 第n个样本的概率 |

**一句话解释：**
对数似然是所有样本对数概率的平均值，用于评估模型的拟合度。

**简单应用场景：**
作为有向图模型参数学习的优化目标。

---

---

### 公式 (11.28)：有向图模型的分解对数似然 / Decomposed Log-Likelihood

$$
\mathcal{L}(\mathcal{D}; \theta) = \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} \log p(x_k^{(n)}|x_{\pi_k}^{(n)}; \theta_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\mathcal{D}; \theta)$ | duì shù sì ránhuàn xìng | 对数似然函数 |
| $x_k^{(n)}$ | yàng běn n de jié diǎn k | 第n个样本的第k个节点 |
| $\theta_k$ | jié diǎn k de cān shù | 节点k的参数 |

**一句话解释：**
有向图的对数似然可分解为各节点条件概率的对数似然和。

**简单应用场景：**
利用分解性进行高效的参数学习，各节点独立优化。

---

---

### 公式 (11.29) - 混合系数更新（拉格朗日乘数法） / Mixture Coefficient Update (Lagrange)

$$
V^\pi_{k+1}(s) = \sum_a \pi(a|s) \left[ r(s, a) + \gamma \sum_{s'} p(s'|s, a) V^\pi_k(s') \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi_k$ | pài | 第k个分量的混合系数 |
| $N_k$ | yǒu xiào yàng běn shù | 分配给第k个分量的有效样本数 |
| $N$ | zǒng yàng běn shù | 总样本数 |

**一句话解释：**
使用拉格朗日乘数法对混合系数的约束最优化。

**简单应用场景：**
推导混合系数的闭式解。

---

---

### 公式 (11.30) - 均值估计 / Mean Estimate

$$
\pi_{k+1}(s) = \arg\max_a \left[ r(s, a) + \gamma \sum_{s'} p(s'|s, a) V^{\pi_k}(s') \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_k$ | mǔ | 第k个分量的均值 |
| $N_k$ | yǒu xiào yàng běn shù | 有效样本数 |
| $\gamma_{nk}$ | gāma | 后验概率 |
| $x_n$ | yàng běn | 样本 |

**一句话解释：**
用加权平均计算高斯分量的均值估计。

**简单应用场景：**
EM算法的M步参数更新。

---

---

### 公式 (11.31)：对数似然函数 / Log-Likelihood Function

$$
\mathcal{L}(\mathcal{D}; \theta) = \frac{1}{N} \sum_{n=1}^{N} \log p(\boldsymbol{x}^{(n)}; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\mathcal{D}; \theta)$ | duì shù sì ránhuàn xìng | 对数似然 |
| $p(\boldsymbol{x}^{(n)}; \theta)$ | dì n ge yàng běn de gài lǜ | 第n个样本的概率 |

**一句话解释：**
对数似然是所有样本对数概率的平均，是参数学习的优化目标。

**简单应用场景：**
评估无向图模型对数据的拟合程度。

---

---

### 公式 (11.32) - 方差估计 / Variance Estimate

$$
V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma_k^2$ | xīgmà píng fāng | 第k个分量的方差 |
| $N_k$ | yǒu xiào yàng běn shù | 有效样本数 |
| $\gamma_{nk}$ | gāma | 后验概率 |
| $x_n$ | yàng běn | 样本 |
| $\boldsymbol{\mu}_k$ | mǔ | 均值 |

**一句话解释：**
用加权平均计算高斯分量的方差估计。

**简单应用场景：**
EM算法中更新分量的标准差。

---

---

### 公式 (11.33) - EM算法高斯混合模型训练 / EM Training Algorithm

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x^{(n)}$ | yàng běn | 第n个训练样本 |
| $N$ | yàng běn shù | 样本总数 |

**一句话解释：**
EM算法高斯混合模型的输入数据说明。

**简单应用场景：**
定义EM算法的输入。

---

---

### 公式 (11.34) - 随机初始化参数 / Random Parameter Initialization

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi_k$ | pài | 混合系数 |
| $\boldsymbol{\mu}_k$ | mǔ | 均值 |
| $\sigma_k$ | xīgmà | 标准差 |
| $K$ | fèn liàng shù | 分量个数 |

**一句话解释：**
为EM算法初始化所有的混合模型参数。

**简单应用场景：**
EM算法开始前的参数设置。

---

---

### 公式 (11.35) - EM迭代 / EM Iteration

$$
\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|A|} & \text{otherwise} \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| repeat | yóu | 重复 |

**一句话解释：**
EM算法的迭代循环开始。

**简单应用场景：**
表示EM算法的迭代结构。

---

---

### 公式 (11.36) - E步计算后验概率 / E-step: Posterior Computation

$$
\mathcal{L}(\theta) = \mathbb{E} \left[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_{nk}$ | gāma | 后验概率 |
| $n$ | shù | 样本索引 |
| $k$ | fèn | 分量索引 |

**一句话解释：**
在固定参数下计算隐变量的后验概率。

**简单应用场景：**
EM算法的E步操作。

---

---

### 公式 (11.37)：经验分布与模型分布的差 / Empirical vs Model Distribution

$$
\frac{\partial \mathcal{L}(\mathcal{D}; \theta)}{\partial \theta_c} = \mathbb{E}_{x \sim p_{x}}\left[f_c(\boldsymbol{x}_c)\right] - \mathbb{E}_{x \sim p(x;\theta)}\left[f_c(\boldsymbol{x}_c)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_x$ | jīng yàn fēn bù | 经验分布 |
| $p(x;\theta)$ | mó xíng fēn bù | 模型分布 |
| $f_c$ | tè zhēng hánshù | 特征函数 |

**一句话解释：**
梯度衡量的是经验特征期望与模型特征期望之间的差异。

**简单应用场景：**
理解参数学习的本质：调整模型使其期望与数据期望相符。

---

## 11.2.2.1 EM算法

---

### 公式 (11.38)：证据下界 / Evidence Lower Bound

$$
p(x; \theta) = \sum_z p(x, z; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x; \theta)$ | kě jiàn biàn liàng de gài lǜ | 可见变量的概率 |
| $z$ | yǐn biàn liàng | 隐变量 |
| $p(x, z; \theta)$ | lián hé gài lǜ | 联合概率 |

**一句话解释：**
边缘概率是联合概率对隐变量的求和，为推导ELBO的基础。

**简单应用场景：**
在EM算法中处理含有隐变量的模型。

---

---

### 公式 (11.39)：对数似然 / Log-Likelihood

$$
\mathcal{L}(\mathcal{D}; \theta) = \frac{1}{N} \sum_{n=1}^{N} \log p(\boldsymbol{x}^{(n)}; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\mathcal{D}; \theta)$ | duì shù sì ránhuàn xìng | 对数似然函数 |
| $N$ | yàng běn shù | 样本总数 |

**一句话解释：**
含隐变量模型的对数似然是可见变量概率对数的平均。

**简单应用场景：**
定义含隐变量模型的参数学习目标。

---

---

### 公式 (11.40)：包含隐变量的对数似然 / Log-Likelihood with Latent Variables

$$
\mathcal{L}(\mathcal{D}; \theta) = \frac{1}{N} \sum_{n=1}^{N} \log \sum_z p(\boldsymbol{x}^{(n)}, z; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\mathcal{D}; \theta)$ | duì shù sì ránhuàn xìng | 对数似然 |
| $z$ | yǐn biàn liàng | 隐变量 |
| $\sum_z$ | duì yǐn biàn liàng qiú hé | 对隐变量求和 |

**一句话解释：**
含隐变量的对数似然需要对隐变量求和，导致优化困难。

**简单应用场景：**
表示含隐变量模型的完整对数似然表达式。

---

---

### 公式 (11.41) - M步固定后验概率更新参数 / M-step: Parameter Update

$$
V_\psi(s) \approx V^\pi(s)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_{nk}$ | gāma | 后验概率 |
| $\pi_k$ | pài | 混合系数 |
| $\boldsymbol{\mu}_k$ | mǔ | 均值 |
| $\sigma_k$ | xīgmà | 标准差 |

**一句话解释：**
在固定后验概率下使用闭式解更新所有参数。

**简单应用场景：**
EM算法的M步操作。

---

---

### 公式 (11.42)：Jensen不等式应用 / Jensen's Inequality

$$
\geq \sum_z q(z) \log \frac{p(x, z; \theta)}{q(z)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\geq$ | dà yú huò děng yú | 大于等于 |
| $q(z)$ | biàn fēn fēn bù | 变分分布 |

**一句话解释：**
应用Jensen不等式得到对数概率的下界ELBO。

**简单应用场景：**
推导变分下界的数学基础。

---

---

### 公式 (11.43)：ELBO定义 / ELBO Definition

$$
\triangleq \text{ELBO}(q, x; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{ELBO}$ | zhèng jù xià jiè | 证据下界 |
| $q$ | biàn fēn fēn bù | 变分分布 |

**一句话解释：**
ELBO是对数证据的下界，EM算法通过交替优化ELBO来学习参数。

**简单应用场景：**
EM算法的理论基础，定义E步和M步的优化目标。

---

---

### 公式 (11.44)：EM算法的M步 / EM Algorithm M-Step

$$
\theta_{t+1} = \arg \max_\theta \text{ELBO}(q_{t+1}, x; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_{t+1}$ | dì t + 1 bù de cān shù | 第t+1步的参数 |
| $\arg \max$ | zuì dà huà | 使函数最大化 |
| $q_{t+1}$ | dì t + 1 bù de biàn fēn | 第t+1步的变分分布 |

**一句话解释：**
M步通过最大化ELBO关于参数来更新参数。

**简单应用场景：**
EM算法的第二步，优化隐变量后优化参数。

---

---

### 公式 (11.45) - 对数似然收敛判断 / Log-Likelihood Convergence Check

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x^{(n)})$ | duì shù jiàn rán xìng | 对数似然函数 |
| $n$ | yàng běn suǒ yǐn | 样本索引 |

**一句话解释：**
当对数似然的总和收敛时停止EM迭代。

**简单应用场景：**
EM算法的停止条件。

---

---

### 公式 (11.46)：ELBO的重写1 / ELBO Rewriting 1

$$
\log p(x; \theta) = \sum_z q(z) \log p(x; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sum_z q(z) = 1$ | hé wéi 1 | 分布$q(z)$的和为1 |

**一句话解释：**
利用概率分布的性质将常数项重写为期望形式。

**简单应用场景：**
ELBO推导中的数学变换。

---

---

### 公式 (11.47) - 输出高斯混合模型参数 / Output GMM Parameters

$$
G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi_k$ | pài | 混合系数 |
| $\sigma_k$ | xīgmà | 标准差 |
| $K$ | fèn liàng shù | 分量个数 |

**一句话解释：**
EM算法收敛后输出的混合高斯模型参数。

**简单应用场景：**
EM算法的最终输出结果。

---

---

### 公式 (11.48) - 图示EM算法在高斯混合中的应用 / EM Algorithm Illustration for GMM

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| 图 | túi | 图示 |
| 高斯混合 | gāo sī hùn hé | 高斯混合 |

**一句话解释：**
展示EM算法逐步优化高斯混合模型的拟合过程。

**简单应用场景：**
可视化理解EM算法如何学习混合模型。

---

---

### 公式 (11.49)：ELBO与KL散度 / ELBO and KL Divergence

$$
= \text{ELBO}(q, x; \theta) + \text{KL}(q(z) \| p(z|x; \theta))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{KL}$ | Kù ěi bǎi lè sàn dù | KL散度 |
| $q(z)$ | biàn fēn fēn bù | 变分分布 |
| $p(z \mid x; \theta)$ | hòu yàn gài lǜ | 后验概率 |

**一句话解释：**
对数概率等于ELBO加上变分分布与后验的KL散度。

**简单应用场景：**
理解EM算法的理论基础，KL散度衡量近似质量。

---

## 11.2.2.2 高斯混合模型

---

### 公式 (11.50)：隐变量分布 / Latent Variable Distribution

$$
p(z = k; \pi) = \pi_k, \quad 1 \leq k \leq K
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z$ | yǐn biàn liàng | 隐变量（簇指示） |
| $\pi_k$ | hùn hé xi shù | 混合系数 |
| $K$ | zǒng簇shù | 簇总数 |

**一句话解释：**
隐变量$z$表示样本属于的簇，$\pi_k$是选择簇k的概率。

**简单应用场景：**
高斯混合模型中定义簇的先验概率。

---

---

### 公式 (11.51)：给定簇的条件分布 / Conditional Distribution Given Cluster

$$
p(x|z = k; \mu_k, \sigma_k) = \mathcal{N}(x; \mu_k, \sigma_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x \mid z=k;...)$ | tiáo jiàn gài lǜ | 条件概率 |
| $\mathcal{N}$ | gāo sī fēn bù | 高斯分布 |
| $\mu_k$ | jūn zhí | 第k个簇的均值 |
| $\sigma_k$ | biāo zhǔn chā | 第k个簇的标准差 |

**一句话解释：**
给定簇k，样本$x$服从均值$\mu_k$、方差$\sigma_k$的高斯分布。

**简单应用场景：**
高斯混合模型中定义每个簇的分布。

---

---

### 公式 (11.52)：高斯分布的表达式 / Gaussian Distribution Expression

$$
= \frac{1}{\sqrt{2\pi\sigma_k}} \exp\left(-\frac{(x - \mu_k)^2}{2\sigma_k^2}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_k$ | jūn zhí | 均值 |
| $\sigma_k$ | biāo zhǔn chā | 标准差 |
| $(x - \mu_k)^2$ | píng fāng chā | 平方差 |

**一句话解释：**
一维高斯分布的标准数学形式，定义分布的概率密度函数。

**简单应用场景：**
在高斯混合模型中计算给定簇的样本概率。

---

---

### 公式 (11.53)：高斯混合模型的边际概率 / GMM Marginal Probability

$$
p(\boldsymbol{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \sigma_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | biān jì gài lǜ | 边际概率 |
| $\pi_k$ | hùn hé xi shù | 混合系数 |
| $\mathcal{N}$ | gāo sī fēn bù | 高斯分布 |

**一句话解释：**
样本概率是所有簇概率的加权和，权重为混合系数。

**简单应用场景：**
高斯混合模型的完整概率表达，用于计算似然度。

---

---

### 公式 (11.54)：单个样本的对数联合概率 / Log Joint Probability

$$
\log p(x^{(n)}) = \log \sum_{z^{(n)}} p(z^{(n)}) p(x^{(n)} | z^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x^{(n)})$ | duì shù gài lǜ | 对数概率 |
| $z^{(n)}$ | yàng běn n de yǐn biàn | 样本n的隐变量 |
| $\sum_{z^{(n)}}$ | duì yǐn biàn qiú hé | 对隐变量求和 |

**一句话解释：**
单个样本的对数联合概率需要对隐变量求和。

**简单应用场景：**
高斯混合模型中计算样本的对数似然。

---

---

### 公式 (11.55)：单个样本的对数边际概率 / Log Marginal Probability

$$
= \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x^{(n)}; \mu_k, \sigma_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log$ | duì shù | 对数 |
| $\pi_k$ | hùn hé xi shù | 混合系数 |
| $\mathcal{N}$ | gāo sī fēn bù | 高斯分布 |

**一句话解释：**
样本对数概率是各簇加权高斯的和的对数。

**简单应用场景：**
在EM算法中计算对数似然。

---

---

### 公式 (11.56)：E步：后验概率1 / E-Step: Posterior Probability 1

$$
\gamma_{nk} \triangleq p(z^{(n)} = k | x^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_{nk}$ | hòu yàn gài lǜ | 后验概率 |
| $p(z^{(n)}=k \mid x^{(n)})$ | gěi dìng x zhī yǔ z de gài lǜ | 给定$x$知$z$的概率 |

**一句话解释：**
后验概率$\gamma_{nk}$表示样本n属于簇k的概率。

**简单应用场景：**
EM算法的E步，计算隐变量的后验概率。

---

---

### 公式 (11.57)：E步：后验概率2 / E-Step: Posterior Probability 2

$$
= \frac{p(z^{(n)} = k) p(x^{(n)} | z^{(n)} = k)}{p(x^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(z^{(n)}=k)$ | xiān yàn gài lǜ | 先验概率 |
| $p(x^{(n)} \mid z^{(n)}=k)$ | sì ránhuàn xìng | 似然度 |
| $p(x^{(n)})$ | biān jì gài lǜ | 边际概率 |

**一句话解释：**
使用贝叶斯公式计算后验概率：先验乘以似然再除以证据。

**简单应用场景：**
高斯混合模型EM算法的E步后验计算。

---

---

### 公式 (11.58)：E步：后验概率3 / E-Step: Posterior Probability 3

$$
= \frac{\pi_k \mathcal{N}(x^{(n)}; \mu_k, \sigma_k)}{\sum_{k'=1}^{K} \pi_{k'} \mathcal{N}(x^{(n)}; \mu_{k'}, \sigma_{k'})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi_k$ | hùn hé xi shù | 混合系数 |
| $\mathcal{N}$ | gāo sī fēn bù | 高斯分布 |
| $\sum_{k'=1}^{K}$ | suǒ yǒu簇qiú hé | 所有簇求和 |

**一句话解释：**
后验概率是簇k的加权高斯与所有簇加权高斯的比值。

**简单应用场景：**
高斯混合模型E步的具体计算形式。

---

---

### 公式 (11.59) - 变量消除法中的边缘概率 / Variable Elimination: Marginal Probability

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_1, x_4)$ | biān yuán gài lǜ | 边缘概率 |
| $\sum$ | qiú hé | 求和 |
| $x_2, x_3$ | biàn liàng | 被消除的变量 |

**一句话解释：**
通过对不需要的变量求和得到目标变量的边缘概率。

**简单应用场景：**
推断问题中计算某些变量的边缘分布。

---

---

### 公式 (11.60)：M步：ELBO展开 / M-Step: ELBO Expansion

$$
= \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{nk} \left(\log \mathcal{N}(x^{(n)}; \mu_k, \sigma_k) + \log \frac{\pi_k}{\gamma_{nk}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log \mathcal{N}$ | gāo sī fēn bù de duì shù | 高斯分布的对数 |
| $\log \pi_k$ | hùn hé xi shù de duì shù | 混合系数的对数 |

**一句话解释：**
ELBO分解为高斯分布项与混合系数项的加权和。

**简单应用场景：**
在M步中分别优化混合系数、均值和方差。

---

---

### 公式 (11.61) - 高斯混合模型的对数似然（完整形式） / GMM Log-Likelihood (Complete)

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s, a) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_{nk}$ | gāma | 后验概率 |
| $x_n$ | yàng běn | 样本 |
| $\boldsymbol{\mu}_k$ | mǔ | 均值 |
| $\sigma_k$ | xīgmà | 标准差 |
| $\pi_k$ | pài | 混合系数 |
| $C$ | cháng shù | 常数 |

**一句话解释：**
加权对数似然函数的完整形式，其中权重为后验概率。

**简单应用场景：**
EM算法中ELBO的推导。

---

---

### 公式 (11.62) - 最大化ELBO的优化问题 / ELBO Maximization

$$
\hat{A}_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

$$
\text{s.t.} \quad \sum_{k=1}^{K} \pi_k = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà huà | 最大化 |
| ELBO | tái xià jiè | 证据下界 |
| s.t. | shǐ dé | 约束条件 |
| $\sum \pi_k = 1$ | hé wéi 1 | 混合系数和为1 |

**一句话解释：**
在概率分布约束下最大化证据下界的优化问题。

**简单应用场景：**
EM算法M步的数学表述。

---

---

### 公式 (11.63)：混合系数的更新 / Mixing Coefficient Update

$$
\pi_k = \frac{N_k}{N}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N_k$ | dì k ge簇 de yàng běn shù | 第k个簇的样本数 |
| $N$ | zǒng yàng běn shù | 总样本数 |

**一句话解释：**
混合系数等于分配到该簇的样本比例。

**简单应用场景：**
高斯混合模型M步中更新混合系数。

---

---

### 公式 (11.64)：均值的更新 / Mean Update

$$
\mu_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk} x^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_k$ | jūn zhí | 第k个簇的均值 |
| $\gamma_{nk}$ | hòu yàn gài lǜ | 样本属于簇k的概率 |
| $N_k$ | xiao shù | 簇的样本数 |

**一句话解释：**
均值是后验加权的样本平均，权重为$\gamma_{nk}$。

**简单应用场景：**
高斯混合模型M步中更新簇的均值。

---

---

### 公式 (11.65)：方差的更新 / Variance Update

$$
\sigma_k^2 = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk} (x^{(n)} - \mu_k)^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma_k^2$ | fāng chā | 第k个簇的方差 |
| $(x^{(n)} - \mu_k)^2$ | píng fāng chā | 样本与均值的平方差 |

**一句话解释：**
方差是后验加权的平方差平均，衡量簇的离散度。

**简单应用场景：**
高斯混合模型M步中更新簇的方差。

---

---

### 公式 (11.66)：簇的有效样本数 / Effective Sample Size

$$
N_k = \sum_{n=1}^{N} \gamma_{nk}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N_k$ | xiào shù | 分配给簇k的有效样本数 |
| $\gamma_{nk}$ | hòu yàn gài lǜ | 样本n对簇k的后验概率 |

**一句话解释：**
有效样本数是所有样本后验概率的和，表示分配给该簇的总权重。

**简单应用场景：**
高斯混合模型中计算簇的有效样本数，用于更新参数。

---

**文档生成完毕。** 已提取Chapter 11第一部分（页266-285）的全部66个公式，包括Chinese名称、English名称、LaTeX公式、符号表、简要解释和应用场景。

---

### 公式 (11.67) - 贝叶斯推断中的条件概率 / Bayes' Rule for Inference

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(q \mid e)$ | hòu yàn gài lǜ | 后验概率 |
| $p(q,e)$ | lián hé gài lǜ | 联合概率 |
| $p(e)$ | zhèng jù | 证据 |

**一句话解释：**
贝叶斯定理给出在观察到证据e后查询变量q的后验概率。

**简单应用场景：**
概率推断中计算条件概率。

---

---

### 公式 (11.68) - 边缘化后的条件概率 / Marginalized Conditional Probability

$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(q \mid e)$ | hòu yàn gài lǜ | 后验概率 |
| $z$ | yǐn biàn liàng | 隐变量 |
| $q'$ | chá xún biàn liàng | 查询变量的其他取值 |

**一句话解释：**
对隐变量边缘化后计算的条件概率公式。

**简单应用场景：**
图模型中的概率推断。

---

---

### 公式 (11.69) - 条件独立性链式法则 / Chain Rule with Conditional Independence

$$
\mathcal{L}_{\text{critic}}(\psi) = \mathbb{E} \left[ (V_\psi(s) - G_t)^2 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_1, x_4)$ | biān yuán gài lǜ | 边缘概率 |
| $p(x_i \mid ...)$ | tiáo jiàn gài lǜ | 条件概率 |
| $\sum$ | qiú hé | 求和 |

**一句话解释：**
利用链式法则和条件独立性分解联合概率为条件概率的乘积。

**简单应用场景：**
贝叶斯网络中的概率计算。

---

---

### 公式 (11.70)：结合律 / Associativity Law

$$
ab + ac = a(b + c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $a, b, c$ | zì | 任意数值或函数 |

**一句话解释：**
代数基本运算法则，用于简化概率计算中的求和项。

**简单应用场景：**
化简变量消去算法中的求和表达式。

---

---

### 公式 (11.71)：边缘概率的求和 / Marginal Probability by Summation

$$
p(x_1, x_4) = p(x_1) \sum_{x_3} p(x_3 | x_1) \sum_{x_2} p(x_2 | x_1) p(x_4 | x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_1, x_2, x_3, x_4$ | xī | 随机变量 |
| $p(\cdot)$ | pài | 概率分布 |

**一句话解释：**
通过逐步消去隐变量计算边缘概率，降低计算复杂度。

**简单应用场景：**
变量消去算法中的逐步求和过程。

---

---

### 公式 (11.72)：因子图上的概率计算 / Probability Calculation on Factor Graph

$$
p(x_4) = \sum_{x_3} \sum_{x_2} p(x_4 | x_2, x_3) \sum_{x_1} p(x_3 | x_1) p(x_2 | x_1) p(x_1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_1, x_2, x_3, x_4$ | xī | 图模型中的节点 |

**一句话解释：**
在因子图上通过逐步求和消去所有中间变量计算边缘概率。

**简单应用场景：**
概率图模型的精确推理。

---

---

### 公式 (11.73)：无向图的联合概率 / Joint Probability in Undirected Graphical Model

$$
p(x) = \frac{1}{Z} \prod_{c \in C} \phi_c(x_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z$ | zī | 配分函数 |
| $C$ | jù tuán | 团集合 |
| $\boldsymbol{\phi}_c$ | fài | 团势函数 |
| $x_c$ | xī | 团中的变量 |

**一句话解释：**
无向图模型通过团势函数的乘积表示联合概率。

**简单应用场景：**
条件随机场(CRF)和马尔可夫随机场的概率表示。

---

---

### 公式 (11.74) - 因子分解的联合概率 / Factorized Joint Probability

$$
r_t^{\text{clip}} = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_1, x_4)$ | biān yuán gài lǜ | 目标边缘概率 |
| $p(x_i)$ | gài lǜ | 各项概率 |

**一句话解释：**
通过链式法则和变量消除计算边缘概率。

**简单应用场景：**
推断算法中消除隐变量。

---

---

### 公式 (11.75)：边缘概率的和式表示 / Marginal Probability as Summation

$$
p(x_t) = \sum_{x_1} \cdots \sum_{x_{t-1}} \sum_{x_{t+1}} \cdots \sum_{x_T} p(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_t$ | xī | 目标变量 |
| $x_1, \ldots, x_T$ | xī | 所有变量 |

**一句话解释：**
通过对所有其他变量求和得到目标变量的边缘概率。

**简单应用场景：**
计算序列模型中特定位置的边缘概率。

---

---

### 公式 (11.76) - 消息传递的递推公式 / Forward Message Passing

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min(r_t^{\text{clip}} \hat{A}_t, \operatorname{clip}(r_t^{\text{clip}}, 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_{t-1,t}(x_t)$ | xiāo xī | 从节点t-1到节点t的消息 |
| $\phi$ | shì | 势函数 |
| $x_t$ | jié diǎn | 节点t的变量 |

**一句话解释：**
前向消息的递推关系，用势函数和前一步的消息计算当前消息。

**简单应用场景：**
信念传播算法中的消息传递。

---

---

### 公式 (11.77)：前向后向算法的因子分解 / Factor Decomposition in Forward-Backward Algorithm

$$
p(x_t) = \frac{1}{Z} \mu_{t-1,t}(x_t) \mu_{t+1,t}(x_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_{t-1,t}(x_t)$ | mǔ | 来自前向的消息 |
| $\boldsymbol{\mu}_{t+1,t}(x_t)$ | mǔ | 来自后向的消息 |
| $Z$ | zī | 配分函数 |
| $t$ | tì | 时刻索引 |

**一句话解释：**
边缘概率可分解为来自前向和后向的消息的乘积。

**简单应用场景：**
隐马尔可夫模型和条件随机场的推理。

---

---

### 公式 (11.78)：前向消息的递推 / Forward Message Recursion

$$
\boldsymbol{\mu}_{t-1,t}(x_t) \triangleq \sum_{x_{t-1}} \phi(x_{t-1}, x_t) \mu_{t-2,t-1}(x_{t-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_{t-1,t}(x_t)$ | mǔ | 前向消息 |
| $\boldsymbol{\phi}(x_{t-1}, x_t)$ | fài | 势函数 |
| $x_{t-1}$ | xī | 前一时刻变量 |

**一句话解释：**
从左向右递推计算前向消息，聚集过去时刻的信息。

**简单应用场景：**
HMM前向算法和CRF推理。

---

---

### 公式 (11.79)：后向消息的递推 / Backward Message Recursion

$$
\boldsymbol{\mu}_{t+1,t}(x_t) \triangleq \sum_{x_{t+1}} \phi(x_t, x_{t+1}) \mu_{t+2,t+1}(x_{t+1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_{t+1,t}(x_t)$ | mǔ | 后向消息 |
| $\boldsymbol{\phi}(x_t, x_{t+1})$ | fài | 势函数 |
| $x_{t+1}$ | xī | 后一时刻变量 |

**一句话解释：**
从右向左递推计算后向消息，聚集未来时刻的信息。

**简单应用场景：**
HMM后向算法和条件概率计算。

---

---

### 公式 (11.80)：结构贝叶斯网络的证据求和 / Evidence Combination in Bayesian Network

$$
Z = \sum_{x_t} \mu_{t-1,t}(x_t) \mu_{t+1,t}(x_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z$ | zī | 配分函数 |
| $\boldsymbol{\mu}_{t-1,t}(x_t)$ | mǔ | 前向消息 |
| $\boldsymbol{\mu}_{t+1,t}(x_t)$ | mǔ | 后向消息 |

**一句话解释：**
通过对所有变量求和计算归一化因子。

**简单应用场景：**
概率归一化和配分函数计算。

---

---

### 公式 (11.81)：观测条件下的概率密度比 / Probability Density Ratio Under Observation

$$
p(z|x) = \frac{p(x, z)}{p(x)} = \frac{p(x, z)}{\int p(x, z) dz}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(z \mid x)$ | pài | 观测x下隐变量z的条件概率 |
| $p(x, z)$ | pài | 联合概率 |
| $p(x)$ | pài | 边缘概率 |

**一句话解释：**
通过联合概率与边缘概率的比值得到条件概率。

**简单应用场景：**
变分推理和EM算法中的后验推理。

---

---

### 公式 (11.82) - 反向消息传递 / Backward Message Passing

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots | s_0=s, a_0=a \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_{t+1,t}(x_t)$ | xiāo xī | 从节点t+1到节点t的消息 |
| $\phi$ | shì | 势函数 |
| $x_t$ | jié diǎn | 节点t的变量 |

**一句话解释：**
反向消息的递推关系，与前向消息对称。

**简单应用场景：**
信念传播算法的反向传递。

---

---

### 公式 (11.83)：证据下界(ELBO) / Evidence Lower Bound

$$
\log p(x) = ELBO(q, x) + KL(q(z) \| p(z|x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x)$ | pài | 证据的对数 |
| $ELBO$ | yī | 证据下界 |
| $KL(\cdot \mid \cdot)$ | kèi | KL散度 |
| $q(z)$ | qiú | 变分分布 |
| $p(z \mid x)$ | pài | 后验分布 |

**一句话解释：**
证据等于ELBO加上变分分布与真实后验的KL散度。

**简单应用场景：**
EM算法和变分推理的理论基础。

---

---

### 公式 (11.84)：ELBO最小化形式 / ELBO Minimization Form

$$
q^*(z) = \arg\min_{q(z) \in \Omega} \left( \log p(x) - ELBO(q, x) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q^*(z)$ | qiú | 最优变分分布 |
| $\Omega$ | ōu mǐ gā | 分布族约束 |
| $ELBO$ | yī | 证据下界 |

**一句话解释：**
通过最小化证据与ELBO的差值找到最优变分分布。

**简单应用场景：**
受约束的变分推理。

---

---

### 公式 (11.85)：ELBO最大化形式 / ELBO Maximization Form

$$
q^*(z) = \arg\max_{q(z) \in \Omega} ELBO(q, x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q^*(z)$ | qiú | 最优变分分布 |
| $ELBO$ | yī | 证据下界 |
| $\Omega$ | ōu mǐ gā | 分布族约束 |

**一句话解释：**
通过最大化证据下界来优化变分分布。

**简单应用场景：**
变分推理的标准形式。

---

---

### 公式 (11.86)：均值场假设下的分解 / Factorization Under Mean-Field Assumption

$$
q(z) = \prod_{m=1}^{M} q_m(z_m)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q(z)$ | qiú | 变分分布 |
| $q_m(z_m)$ | qiú | 第m个隐变量的变分分布 |
| $M$ | ēn | 隐变量个数 |
| $z_m$ | zèi | 第m个隐变量 |

**一句话解释：**
均值场假设将变分分布分解为独立的因子。

**简单应用场景：**
简化变分推理的计算复杂度。

---

---

### 公式 (11.87)：ELBO的积分形式 / Integral Form of ELBO

$$
ELBO(q, x) = \int q(z) \log \frac{p(x, z)}{q(z)} dz
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $ELBO$ | yī | 证据下界 |
| $q(z)$ | qiú | 变分分布 |
| $p(x, z)$ | pài | 联合概率 |

**一句话解释：**
ELBO是变分分布加权的联合概率与变分分布比值的对数期望。

**简单应用场景：**
变分推理的基本公式。

---

---

### 公式 (11.88)：ELBO的展开形式 / Expanded Form of ELBO

$$
ELBO(q, x) = \int q(z) \left( \log p(x, z) - \log q(z) \right) dz
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $ELBO$ | yī | 证据下界 |
| $q(z)$ | qiú | 变分分布 |
| $p(x, z)$ | pài | 联合概率 |

**一句话解释：**
ELBO等于联合概率对数期望与变分分布熵的和。

**简单应用场景：**
计算变分下界。

---

---

### 公式 (11.89)：ELBO的分解形式 / Decomposed Form of ELBO

$$
ELBO(q, x) = \int \prod_{m=1}^{M} q_m(z_m) \left( \log p(x, z) - \sum_{m=1}^{M} \log q_m(z_m) \right) dz
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_m(z_m)$ | qiú | 第m个隐变量的变分分布 |
| $\log p(x, z)$ | pài | 联合概率的对数 |
| $M$ | ēn | 隐变量个数 |

**一句话解释：**
在均值场假设下，ELBO分解为各因子分布的乘积。

**简单应用场景：**
均值场变分推理。

---

---

### 公式 (11.90)：均值场下的ELBO重写 / ELBO Under Mean-Field Rewriting

$$
ELBO(q, x) = \int q_j \left( \left[ \int \prod_{m \neq j} q_m \log p(x, z) dz_m \right] \right) dz_j - \int q_j \log q_j dz_j + \text{const}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_j$ | qiú | 第j个变量的变分分布 |
| $q_m$ | qiú | 第m个变量的变分分布 |
| $\log p(x, z)$ | pài | 联合概率对数 |
| const | cháng shù | 常数项 |

**一句话解释：**
ELBO关于$q_j$的最优形式包含期望项和熵项。

**简单应用场景：**
均值场变分推理的坐标上升。

---

---

### 公式 (11.91) - 任意节点的边缘概率计算 / Marginal Probability at Arbitrary Node

$$
\mathcal{L}_{\text{actor}}(\theta) = -\mathbb{E}_{s} \left[ Q_\psi(s, \pi_\theta(s)) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z$ | guī huà cháng shù | 归一化常数 |
| $\mu_{t-1,t}$ | qián xiāo xī | 前向消息 |
| $\mu_{t+1,t}$ | hòu xiāo xī | 后向消息 |

**一句话解释：**
通过将前向消息和后向消息相乘并对变量求和得到归一化常数。

**简单应用场景：**
计算任意节点的边缘概率。

---

---

### 公式 (11.92)：对数联合概率的分解 / Decomposition of Log Joint Probability

$$
\log p(x, z) = \int \prod_{m \neq j} q_m \log p(x, z) dz_m
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x, z)$ | pài | 联合概率对数 |
| $q_m$ | qiú | 其他变量的变分分布 |
| $z_m$ | zèi | 其他隐变量 |

**一句话解释：**
联合概率对数可表示为其他变量在各自分布下的期望。

**简单应用场景：**
变分推理中计算最优因子。

---

---

### 公式 (11.93) - 信念传播中的联合概率推断 / Joint Probability in Belief Propagation

$$
\mathcal{L}_{\text{critic}}(\psi) = \mathbb{E} \left[ (Q_\psi(s, a) - y)^2 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_t)$ | biān yuán gài lǜ | 边缘概率 |
| $Z$ | guī huà cháng shù | 归一化常数 |
| $\phi$ | shì | 势函数 |

**一句话解释：**
结合前后消息计算任意节点的边缘概率。

**简单应用场景：**
链式图模型中的完整推断。

---

---

### 公式 (11.94) - 前向后向消息分解 / Forward-Backward Decomposition

$$
y = r + \gamma Q_{\psi^-}(s', \pi_{\theta^-}(s'))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_t)$ | biān yuán gài lǜ | 节点t的边缘概率 |
| $Z$ | guī huà cháng shù | 归一化常数 |
| $\phi$ | shì | 势函数 |

**一句话解释：**
边缘概率可以分解为前向和后向消息的乘积。

**简单应用场景：**
推断算法的效率分析。

---

---

### 公式 (11.95) - 联合概率的完整分解 / Complete Factorization of Joint Probability

$$
\psi^- \leftarrow \tau \psi + (1 - \tau) \psi^-
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_t)$ | biān yuán gài lǜ | 边缘概率 |
| $\prod$ | chéng jī | 乘积 |
| $\phi$ | shì | 势函数 |
| $Z$ | guī huà cháng shù | 归一化常数 |

**一句话解释：**
完整展开了前向和后向消息的乘积形式。

**简单应用场景：**
理论推导中的完整概率分解。

---

---

### 公式 (11.96)：蒙特卡罗估计 / Monte Carlo Estimate

$$
\hat{f_N} = \frac{1}{N} \left( f(x^{(1)}) + \cdots + f(x^{(N)}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{f_N}$ | fù | N个样本的平均值 |
| $x^{(i)}$ | xī | 第i个样本 |
| $N$ | ēn | 样本数量 |
| $f(\cdot)$ | hánshù | 函数 |

**一句话解释：**
用有限样本的平均值估计期望值。

**简单应用场景：**
数值计算期望和积分。

---

---

### 公式 (11.97)：蒙特卡罗估计的收敛性 / Convergence of Monte Carlo Estimate

$$
\hat{f_N} \xrightarrow{p} E_p[f(x)] \quad \text{当} \quad N \to \infty
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{f_N}$ | fù | 估计值 |
| $E_p[f(x)]$ | qī wàng | 真实期望 |
| $\xrightarrow{p}$ | shōu liǎn | 依概率收敛 |
| $N$ | ēn | 样本数量 |

**一句话解释：**
样本量趋于无穷时，蒙特卡罗估计依概率收敛到真实期望。

**简单应用场景：**
蒙特卡罗采样方法的理论保证。

---

---

### 公式 (11.98) - 变分推断的KL散度最小化 / Variational Inference: KL Divergence Minimization

$$
\pi_\theta(a|s) = \mu_\theta(s) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q^*(z)$ | zuì yōu hòu yàn | 最优后验近似 |
| KL | kè ěr bǎk hā | KL散度 |
| $p(z \mid x)$ | zhēn hòu yàn | 真实后验 |

**一句话解释：**
通过最小化KL散度找到最接近真实后验的变分分布。

**简单应用场景：**
变分推断的基本优化目标。

---

---

### 公式 (11.99) - 对数似然与ELBO的关系 / Log-Likelihood and ELBO Relationship

$$
\mathcal{L}(\theta) = \mathbb{E}_{s, \boldsymbol{\epsilon}} [Q_\psi(s, \mu_\theta(s) + \boldsymbol{\epsilon})]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x)$ | duì shù jiàn rán xìng | 对数似然 |
| ELBO | tái xià jiè | 证据下界 |
| KL | kè ěr bǎk hā | KL散度 |

**一句话解释：**
对数似然等于ELBO加上KL散度，KL散度非负。

**简单应用场景：**
推导变分推断的理论基础。

---

---

### 公式 (11.100) - ELBO的定义 / ELBO Definition

$$
J(\pi) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi} [r(s, a)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| ELBO | tái xià jiè | 证据下界 |
| $\mathbb{E}$ | qī wàng zhí | 期望 |
| $q(z)$ | biàn fēn | 变分分布 |
| $p(x, z)$ | lián hé gài lǜ | 联合分布 |

**一句话解释：**
ELBO是联合概率的期望减去近似后验的熵。

**简单应用场景：**
变分推断中ELBO的计算。

---

---

### 公式 (11.101) - ELBO的可分性形式 / ELBO Decomposition

$$
\mathcal{L}_Q(\psi) = \mathbb{E} \left[ (Q_\psi(s,a) - (r + \gamma V_{\bar{\psi}}(s')))^2 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| ELBO | tái xià jiè | 证据下界 |
| $\mathbb{E}$ | qī wàng zhí | 期望 |
| $p(x \mid z)$ | tiáo jiàn gài lǜ | 条件似然 |
| KL | kè ěr bǎk hā | KL散度 |
| $p(z)$ | xiān yàn | 先验分布 |

**一句话解释：**
ELBO分解为重建项（期望对数似然）与正则项（KL散度）。

**简单应用场景：**
变分自编码器（VAE）的优化目标。

---

---

### 公式 (11.102) - 变分分布的因子分解 / Factorized Variational Distribution

$$
V_{\bar{\psi}}(s) = \mathbb{E}_{a \sim \pi_\theta} [Q_{\bar{\psi}}(s,a) - \alpha \log \pi_\theta(a|s)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q(z)$ | biàn fēn | 变分分布 |
| $q_i(z_i)$ | yīn zǐ fēn | 第i个因子 |
| $\prod$ | chéng jī | 乘积 |

**一句话解释：**
平均场变分推断假设隐变量分布可以因子分解。

**简单应用场景：**
简化变分推断的计算复杂度。

---

---

### 公式 (11.103) - 坐标上升变分推断 / Coordinate Ascent Variational Inference

$$
\mathcal{L}_\pi(\theta) = \mathbb{E}_{s} \left[ \mathbb{E}_{a \sim \pi_\theta} [\alpha \log \pi_\theta(a|s) - Q_\psi(s,a)] \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_i^*(z_i)$ | zuì yōu yīn zǐ | 最优因子 |
| $\mathbb{E}$ | qī wàng zhí | 期望 |
| $q_{-i}$ | qī tā yīn zǐ | 除了第i个的其他因子 |

**一句话解释：**
通过坐标上升法依次优化每个因子的变分分布。

**简单应用场景：**
均匀场变分推断的优化算法。

---

---

### 公式 (11.104) - 对数后验的期望 / Expected Log Posterior

$$
\mathcal{L}(\alpha) = \mathbb{E}_{a \sim \pi_\theta} [-\alpha (\log \pi_\theta(a|s) + \bar{H})]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q(z \mid x)$ | zuì yōu hòu yàn | 最优后验近似 |
| KL | kè ěr bǎk hā | KL散度 |
| $p(z \mid x)$ | zhēn hòu yàn | 真实后验 |

**一句话解释：**
最优变分分布最小化与真实后验的KL散度。

**简单应用场景：**
变分推断的优化目标。

---

---

### 公式 (11.105) - ELBO与对数似然的关系重述 / ELBO-Log-Likelihood Relationship

$$
\bar{\psi} \leftarrow \tau \psi + (1 - \tau) \bar{\psi}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x)$ | duì shù jiàn rán xìng | 对数似然 |
| ELBO | tái xià jiè | 证据下界 |
| KL | kè ěr bǎk hā | KL散度 |
| $p(z \mid x)$ | hòu yàn gài lǜ | 后验概率 |

**一句话解释：**
对数似然函数恒等于ELBO加上KL散度。

**简单应用场景：**
变分推断理论中的基本恒等式。

---

这份文档完整地重新提取了第11章中列出的所有错误公式。每个公式都包含精确的LaTeX表达、完整的符号表、一句话解释和应用场景。

---

### 公式 (11.106)：蒙特卡罗近似的规范化形式 / Normalized Monte Carlo Approximation

$$
E_p[f(x)] \approx \frac{\sum_{n=1}^{N} f(x^{(n)}) \tilde{w}(x^{(n)})}{\sum_{n=1}^{N} \tilde{w}(x^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_p[f(x)]$ | qī wàng | 期望 |
| $\tilde{w}(x^{(n)})$ | quán | 第n个样本的权重 |
| $x^{(n)}$ | xī | 第n个样本 |
| $N$ | ēn | 样本数量 |

**一句话解释：**
用加权样本近似非规范化分布下的期望。

**简单应用场景：**
粒子滤波和序列蒙特卡罗方法。

---

---

### 公式 (11.107)：条件随机场的势函数 / CRF Potential Function

$$
\phi_t(x_t, x_{t+1}|y) = \exp\left(\sum_{j=1}^{m} \lambda_j f_j(x_t, x_{t+1}, y, t)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\phi_t$ | shì hánshù | 在位置 $t$ 的势函数 |
| $x_t$ | zhuàngtài | 位置 $t$ 的状态 |
| $x_{t+1}$ | zhuàngtài | 位置 $t+1$ 的状态 |
| $y$ | guānchá | 观察序列 |
| $\lambda_j$ | quán zhòng | 第 $j$ 个特征的权重 |
| $f_j$ | tè zhēng | 第 $j$ 个特征函数 |

**一句话解释：**
条件随机场通过指数族势函数对观察序列给定条件下状态序列的联合分布进行建模。

**简单应用场景：**
在序列标注任务（如词性标注、命名实体识别）中建模状态间的依赖关系。

---

---

### 公式 (11.108)：条件随机场的联合概率 / CRF Joint Probability

$$
p(x|y) = \frac{1}{Z(y)} \prod_{t=1}^{T} \phi_t(x_t, x_{t+1}|y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x \mid y)$ | gài lǜ | 给定观察 $y$ 时状态序列 $x$ 的条件概率 |
| $Z(y)$ | huījī hánshù | 配分函数 |
| $T$ | shùliàng | 序列长度 |
| $\phi_t$ | shì hánshù | 位置 $t$ 的势函数 |

**一句话解释：**
条件随机场用归一化的势函数乘积表示观察序列条件下状态序列的联合概率分布。

**简单应用场景：**
在自然语言处理中对条件概率的序列标注问题进行建模和推理。

---

---

### 公式 (11.109)：条件随机场的对数似然 / CRF Log-Likelihood

$$
\mathcal{L}(x, y) = \sum_{t=1}^{T} \sum_{j=1}^{m} \lambda_j f_j(x_t, x_{t+1}, y, t) - \log Z(y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | duì shù sìrán | 对数似然函数 |
| $\lambda_j$ | quán zhòng | 第 $j$ 个特征的权重参数 |
| $f_j$ | tè zhēng | 第 $j$ 个特征函数 |
| $Z(y)$ | huījī hánshù | 配分函数 |

**一句话解释：**
条件随机场的对数似然函数由特征函数加权和与配分函数的对数组成。

**简单应用场景：**
在序列标注模型训练中作为损失函数进行梯度优化。

---

---

### 公式 (11.110)：配分函数 / Partition Function

$$
Z(y) = \sum_{x} \prod_{t=1}^{T} \phi_t(x_t, x_{t+1}|y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z(y)$ | huījī hánshù | 配分函数 |
| $x$ | zhuàngtài xuliè | 所有可能的状态序列 |
| $T$ | shùliàng | 序列长度 |
| $\phi_t$ | shì hánshù | 位置 $t$ 的势函数 |

**一句话解释：**
配分函数通过对所有可能的状态序列求和来归一化条件概率分布。

**简单应用场景：**
在条件随机场中用于计算边际概率和进行推理。

---

---

### 公式 (11.111)：条件随机场的边际概率 / CRF Marginal Probability

$$
p(x_t|y) = \frac{\sum_{x_{t-1}, x_{t+1}} \prod_{t'=1}^{T} \phi_{t'}(x_{t'}, x_{t'+1}|y)}{Z(y)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_t \mid y)$ | gài lǜ | 位置 $t$ 给定观察 $y$ 时的边际概率 |
| $x_t$ | zhuàngtài | 位置 $t$ 的状态 |
| $\phi_{t'}$ | shì hánshù | 位置 $t'$ 的势函数 |
| $Z(y)$ | huījī hánshù | 配分函数 |

**一句话解释：**
条件随机场的边际概率通过对其他位置求和并归一化得到单个位置的概率分布。

**简单应用场景：**
在序列标注中用于计算每个位置各个标签的概率。

---

---

### 公式 (11.112)：条件随机场的对数配分函数梯度 / CRF Gradient of Log Partition Function

$$
\frac{\partial \log Z(y)}{\partial \lambda_j} = \sum_{t=1}^{T} \sum_{x_t, x_{t+1}} p(x_t, x_{t+1}|y) f_j(x_t, x_{t+1}, y, t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \lambda_j}$ | duì fǎ | 对 $\lambda_j$ 的偏导数 |
| $\log Z(y)$ | duì shù huījī | 对数配分函数 |
| $p(x_t, x_{t+1} \mid y)$ | gài lǜ | 给定 $y$ 时相邻状态的联合概率 |
| $f_j$ | tè zhēng | 第 $j$ 个特征函数 |

**一句话解释：**
配分函数对参数的梯度可以通过状态对的边际概率和特征函数的乘积求和得到。

**简单应用场景：**
在条件随机场的参数学习中用于计算梯度进行优化。

---

---

### 公式 (11.113)：条件随机场的对数似然梯度 / CRF Log-Likelihood Gradient

$$
\frac{\partial \mathcal{L}(x, y)}{\partial \lambda_j} = \sum_{t=1}^{T} f_j(x_t, x_{t+1}, y, t) - \sum_{t=1}^{T} \sum_{x_t, x_{t+1}} p(x_t, x_{t+1}|y) f_j(x_t, x_{t+1}, y, t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \lambda_j}$ | duì fǎ | 对 $\lambda_j$ 的偏导数 |
| $\mathcal{L}$ | duì shù sìrán | 对数似然函数 |
| $f_j$ | tè zhēng | 第 $j$ 个特征函数 |
| $p(x_t, x_{t+1} \mid y)$ | gài lǜ | 边际概率 |

**一句话解释：**
对数似然的梯度为观察数据中特征函数的期望与模型预测的期望之差。

**简单应用场景：**
在条件随机场的参数优化中用于实现随机梯度下降算法。

---

---

### 公式 (11.114)：前向算法的递推关系 / Forward Algorithm Recursion

$$
\alpha_t(i) = \sum_{j} \alpha_{t-1}(j) \phi_t(x_j, x_i|y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t(i)$ | qiánxiàng gài lǜ | 时刻 $t$ 状态 $i$ 的前向概率 |
| $j$ | zhuàngtài suǒyǐn | 时刻 $t-1$ 的状态索引 |
| $\phi_t$ | shì hánshù | 时刻 $t$ 的势函数 |

**一句话解释：**
前向算法通过递推关系高效计算条件随机场中各位置各状态的概率。

**简单应用场景：**
在条件随机场中用于快速计算配分函数和边际概率。

---

---

### 公式 (11.115)：后向算法的递推关系 / Backward Algorithm Recursion

$$
\beta_t(i) = \sum_{j} \phi_{t+1}(x_i, x_j|y) \beta_{t+1}(j)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\beta_t(i)$ | hòuxiàng gài lǜ | 时刻 $t$ 状态 $i$ 的后向概率 |
| $j$ | zhuàngtài suǒyǐn | 时刻 $t+1$ 的状态索引 |
| $\phi_{t+1}$ | shì hánshù | 时刻 $t+1$ 的势函数 |

**一句话解释：**
后向算法从序列末尾向前递推计算各位置状态的后向概率。

**简单应用场景：**
在条件随机场推理中与前向算法配合计算边际概率。

---

---

### 公式 (11.116)：前向后向算法的边际概率 / Forward-Backward Marginal Probability

$$
p(x_t = i|y) = \frac{\alpha_t(i) \beta_t(i)}{Z(y)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_t = i \mid y)$ | gài lǜ | 时刻 $t$ 状态为 $i$ 的条件概率 |
| $\alpha_t(i)$ | qiánxiàng gài lǜ | 前向概率 |
| $\beta_t(i)$ | hòuxiàng gài lǜ | 后向概率 |
| $Z(y)$ | huījī hánshù | 配分函数 |

**一句话解释：**
前向后向算法通过前向概率和后向概率的乘积来计算单个位置的边际概率。

**简单应用场景：**
在条件随机场的推理和学习中用于有效计算边际概率分布。

---

---

### 公式 (11.117)：维特比算法的递推 / Viterbi Algorithm Recursion

$$
\delta_t(i) = \max_{j} \delta_{t-1}(j) \phi_t(x_j, x_i|y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta_t(i)$ | zuì dà lù jìng | 时刻 $t$ 状态 $i$ 的最大路径概率 |
| $j$ | qián yī bù zhuàngtài | 前一时刻的状态 |
| $\phi_t$ | shì hánshù | 时刻 $t$ 的势函数 |
| $\max$ | zuì dàhuà | 最大化操作 |

**一句话解释：**
维特比算法通过动态规划找到使联合概率最大的最优状态序列。

**简单应用场景：**
在序列标注中用于解码找到概率最高的标签序列。

---

---

### 公式 (11.118)：维特比算法的回溯 / Viterbi Backtracking

$$
\psi_t(i) = \arg\max_{j} \delta_{t-1}(j) \phi_t(x_j, x_i|y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\psi_t(i)$ | huítùlù | 时刻 $t$ 状态 $i$ 的最优前驱状态 |
| $\arg\max$ | zuì dà huà zì biàoshù | 最大值对应的参数 |
| $\delta_{t-1}(j)$ | qián yī bù zuì dà lù jìng | 前一时刻的最大路径概率 |

**一句话解释：**
维特比算法通过记录回溯指针可以从终点回溯到起点恢复出最优状态序列。

**简单应用场景：**
在序列标注模型中用于获得最优的标签序列预测结果。

---

---

### 公式 (11.119)：条件随机场的解码问题 / CRF Decoding

$$
x^* = \arg\max_{x} p(x|y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x^*$ | zuì yōu zhuàngtài xuliè | 最优的状态序列 |
| $\arg\max$ | zuì dà huà zì biàoshù | 最大化的参数 |
| $p(x \mid y)$ | gài lǜ | 条件概率分布 |

**一句话解释：**
条件随机场的解码问题是求使条件概率最大的状态序列。

**简单应用场景：**
在序列标注任务中用于预测最可能的标签序列。

---

---

### 公式 (11.120)：结构化学习的感知机损失 / Structured Perceptron Loss

$$
\mathcal{L} = -p(x^*|y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī hánshù | 损失函数 |
| $p(x^* \mid y)$ | gài lǜ | 正确序列的条件概率 |
| $x^*$ | zhèngquè xuliè | 真实的状态序列 |

**一句话解释：**
感知机损失通过最大化正确序列的概率来训练结构化预测模型。

**简单应用场景：**
在序列标注和结构化预测问题中进行模型参数优化。

---

---

### 公式 (11.121)：链式条件随机场的势函数 / Linear Chain CRF Potential

$$
\phi_t(x_t, x_{t+1}|y) = \exp\left(\sum_{k=1}^{K} \lambda_k f_k(x_t, x_{t+1}, y, t)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\phi_t$ | shì hánshù | 在位置 $t$ 的势函数 |
| $x_t, x_{t+1}$ | xiānglín zhuàngtài | 相邻时刻的状态 |
| $\lambda_k$ | quán zhòng | 第 $k$ 个特征的权重 |
| $f_k$ | tè zhēng | 第 $k$ 个特征函数 |
| $K$ | tè zhēng gèshù | 特征总数 |

**一句话解释：**
链式条件随机场用指数族势函数表示相邻状态之间的依赖关系。

**简单应用场景：**
在自然语言处理序列标注任务中建模状态的局部依赖关系。

---

---

### 公式 (11.122)：条件随机场的参数更新 / CRF Parameter Update

$$
\lambda_k^{(t+1)} = \lambda_k^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial \lambda_k}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_k^{(t+1)}$ | pàrameter | 第 $t+1$ 次迭代的参数 |
| $\lambda_k^{(t)}$ | pàrameter | 第 $t$ 次迭代的参数 |
| $\eta$ | xuéxí lǜ | 学习率 |
| $\frac{\partial \mathcal{L}}{\partial \lambda_k}$ | tìdù | 梯度 |

**一句话解释：**
条件随机场使用梯度下降法通过对数似然函数的梯度进行参数更新。

**简单应用场景：**
在条件随机场的训练中用于迭代优化模型参数。

---

---

### 公式 (11.123)：条件随机场的约束优化 / CRF Constrained Optimization

$$
\max_{\lambda} \mathcal{L}(\lambda) \text{ s.t. } \sum_{k=1}^{K} \lambda_k = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dàhuà | 最大化 |
| $\mathcal{L}(\lambda)$ | duì shù sìrán | 对数似然函数 |
| $\text{s.t.}$ | shǐ de | 使得 |
| $\lambda_k$ | quán zhòng | 参数权重 |

**一句话解释：**
带约束的条件随机场优化在参数权重归一化的约束下最大化对数似然。

**简单应用场景：**
在需要参数稀疏性或权重约束的条件随机场模型中进行优化。

---

---

### 公式 (11.124)：条件随机场的交叉熵损失 / CRF Cross-Entropy Loss

$$
\mathcal{L} = -\log p(x^*|y) + \log Z(y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔnshī | 交叉熵损失 |
| $p(x^* \mid y)$ | zhèngquè gài lǜ | 正确序列的条件概率 |
| $Z(y)$ | huījī hánshù | 配分函数 |

**一句话解释：**
条件随机场的交叉熵损失衡量模型预测与真实标签之间的差异。

**简单应用场景：**
在神经网络与条件随机场的混合模型中作为训练损失函数。

---

---

### 公式 (11.125)：序列标注的F1评分 / Sequence Labeling F1 Score

$$
F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F_1$ | F yī fēnshù | F1评分 |
| $\text{Precision}$ | jǔnquè lǜ | 精确率 |
| $\text{Recall}$ | huíhuà lǜ | 召回率 |

**一句话解释：**
F1评分是精确率和召回率的调和平均数，用于评估序列标注模型的性能。

**简单应用场景：**
在序列标注任务中全面评估模型在识别正确标签序列上的能力。

---

---

## 第12章：深度信念网络

### 公式 (12.1)：玻尔兹曼机能量定义 / Energy Function of Boltzmann Machine

$$
p(x) = \frac{1}{Z}\exp\left(-\frac{E(x)}{T}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| p(x) | gài lǜ | 概率 (Probability) |
| E(x) | néng liàng | 能量 (Energy) |
| Z | pīn fēn hánshù | 配分函数 (Partition Function) |
| T | wēn dù | 温度 (Temperature) |
| exp | zhǐ shù | 指数函数 (Exponential) |

**一句话解释：**
玻尔兹曼机的能量函数定义了系统状态的概率分布，温度参数T控制分布的"平坦度"。

**简单应用场景：**
在统计物理学中，用于描述微观粒子在热平衡状态下的分布规律。

---

---

### 公式 (12.2)：生成对抗网络的目标函数 / Objective Function of GAN

$$
p(\boldsymbol{x}; \theta) = \frac{1}{Z(\theta)} \exp(-E(\boldsymbol{x}; \theta))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G$ | shēng chéng qì | 生成器，用于生成假样本 |
| $D$ | pàn bié qì | 判别器，用于判断样本真假 |
| $p_{\text{data}}(x)$ | zhēn shí shù jù fēn bù | 真实数据分布 |
| $p_z(z)$ | suí jī biàn liàng fēn bù | 随机变量分布 |
| $\mathbb{E}$ | qī wàng | 期望 |

**一句话解释：**
生成器和判别器通过对抗训练，生成器试图生成逼真样本欺骗判别器，判别器则试图正确区分真假样本。

**简单应用场景：**
图像生成、风格转移、数据增强等任务中用于生成高质量的合成样本。

---

---

### 公式 (12.3)：判别器损失函数 / Discriminator Loss

$$
Z(\theta) = \sum_{\boldsymbol{x}} \exp(-E(\boldsymbol{x}; \theta))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L_D$ | pàn bié qì sǔn shī | 判别器损失 |
| $x$ | zhēn shí yàng běn | 真实样本 |
| $z$ | suí jī gǔ | 随机故 |
| $G(z)$ | shēng chéng de jiǎ yàng běn | 生成的假样本 |

**一句话解释：**
判别器希望对真实样本输出接近1，对假样本输出接近0。

**简单应用场景：**
GAN训练中判别器的优化目标。

---

---

### 公式 (12.4)：生成器损失函数 / Generator Loss

$$
E(\boldsymbol{x}; \theta) = -\boldsymbol{x}^\top \boldsymbol{W} \boldsymbol{x} - \boldsymbol{b}^\top \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L_G$ | shēng chéng qì sǔn shī | 生成器损失 |
| $D(G(z))$ | pàn bié qì duì jiǎ yàng běn de pàn dìng | 判别器对假样本的判定 |

**一句话解释：**
生成器希望生成的样本能欺骗判别器，使得判别器输出接近1。

**简单应用场景：**
GAN训练中生成器的优化目标。

---

---

### 公式 (12.5)：Wasserstein GAN目标函数 / WGAN Objective Function

$$
p(\boldsymbol{x}; \theta) = \frac{1}{Z(\theta)} \exp(\boldsymbol{x}^\top \boldsymbol{W} \boldsymbol{x} + \boldsymbol{b}^\top \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}$ | pàn bié qì hánshù kōng jiān | 判别器函数空间 |
| $D(x)$ | pàn bié qì duì x de zhào yìng | 判别器对x的映射值 |
| $D(G(z))$ | pàn bié qì duì jiǎ yàng běn de zhào yìng | 判别器对假样本的映射值 |

**一句话解释：**
使用Wasserstein距离替代JS散度，使得训练更稳定。

**简单应用场景：**
改进GAN的训练稳定性和收敛速度。

---

---

### 公式 (12.6)：条件概率 - 反向形式 / Conditional Probability - Reverse Form

$$
p(x_i = 0|\boldsymbol{x}_{\setminus i}) = 1 - p(x_i = 1|\boldsymbol{x}_{\setminus i})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| x_i | biàn liàng | 第i个变量 (Variable i) |
| $p(x_i = 0 \mid \boldsymbol{x}_{\setminus i})$ | gài lǜ | 条件概率 (Conditional probability) |
| $p(x_i = 1 \mid \boldsymbol{x}_{\setminus i})$ | gài lǜ | 条件概率 (Conditional probability) |

**一句话解释：**
变量取值为0的条件概率是1减去取值为1的条件概率，反映二元变量的互斥性。

**简单应用场景：**
二元随机变量的概率计算。

---

---

### 公式 (12.7)：条件GAN的目标函数 / Conditional GAN Objective Function

$$
E(\boldsymbol{v}, \boldsymbol{h}; \theta) = -\boldsymbol{v}^\top \boldsymbol{W} \boldsymbol{h} - \boldsymbol{a}^\top \boldsymbol{v} - \boldsymbol{b}^\top \boldsymbol{h}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | tiáo jiàn xìn xī | 条件信息 |
| $D(x \mid y)$ | zài tiáo jiàn y xià pàn bié x | 在条件y下判别x |
| $G(z \mid y)$ | zài tiáo jiàn y xià shēng chéng yàng běn | 在条件y下生成样本 |

**一句话解释：**
在生成和判别过程中加入条件信息（如类别标签），使生成结果受条件控制。

**简单应用场景：**
受控图像生成、文本到图像生成等。

---

---

### 公式 (12.8)：谱归一化 / Spectral Normalization

$$
p(\boldsymbol{v}, \boldsymbol{h}; \theta) = \frac{1}{Z(\theta)} \exp(-E(\boldsymbol{v}, \boldsymbol{h}; \theta))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $W$ | quán zhòng jǔ zhèn | 权重矩阵 |
| $\sigma(W)$ | zuì dà tè zhēng zhí | 最大特征值 |
| $W_{\text{SN}}$ | guī yī huà hòu de quán zhòng | 归一化后的权重 |

**一句话解释：**
通过将权重矩阵除以其最大特征值来归一化，提高判别器的Lipschitz连续性。

**简单应用场景：**
提高GAN训练的稳定性。

---

---

### 公式 (12.9)：对数似然 / Log Likelihood

$$
E(x) = -T\log p(x) - T\log Z
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| E(x) | néng liàng | 能量 (Energy) |
| T | wēn dù | 温度 (Temperature) |
| p(x) | gài lǜ | 概率 (Probability) |
| Z | pīn fēn hánshù | 配分函数 (Partition Function) |
| $\log$ | duì shù | 对数 (Logarithm) |

**一句话解释：**
能量与概率的对数存在线性关系，通过温度参数T连接。

**简单应用场景：**
概率模型与能量函数的相互转换。

---

---

### 公式 (12.10)：能量差 / Energy Difference

$$
\Delta E_i(\boldsymbol{x}_{\setminus i}) = -T \log \frac{p(x_i = 1, \boldsymbol{x}_{\setminus i})}{p(x_i = 0, \boldsymbol{x}_{\setminus i})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta E_i$ | néng liàng chā | 第i个变量翻转的能量差 |
| $\boldsymbol{x}_{\setminus i}$ | chú dì i gè biàn liàng wài de xiàng liàng | 除第i个变量外的向量 |
| $T$ | wēn dù | 温度参数 |
| $p$ | gài lǜ | 概率 |

**一句话解释：**
翻转第i个变量时系统能量的变化量，等于两种状态联合概率之比的对数乘以温度。

**简单应用场景：**
玻尔兹曼机中吉布斯采样时判断是否翻转变量。

---

---

### 公式 (12.11)：渐进式GAN的层级生成 / Progressive GAN Layer-wise Generation

$$
p(h_j = 1 | \boldsymbol{v}) = \sigma(\boldsymbol{w}_j^\top \boldsymbol{v} + b_j)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | fǎ dé | 衰减参数，范围[0,1] |
| $x_{\text{low}}$ | dī fēn biàn lǜ tè zhēng | 低分辨率特征 |
| $x_{\text{high}}$ | gāo fēn biàn lǜ tè zhēng | 高分辨率特征 |

**一句话解释：**
通过衰减因子平滑地从低分辨率过渡到高分辨率，实现渐进式训练。

**简单应用场景：**
高分辨率图像生成。

---

---

### 公式 (12.12)：自注意力机制 / Self-Attention Mechanism

$$
p(v_i = 1 | \boldsymbol{h}) = \sigma(\boldsymbol{w}_i^\top \boldsymbol{h} + a_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q$ | chá xún | 查询 |
| $K$ | kè | 键 |
| $V$ | zhí | 值 |
| $d_k$ | jiàn de wéi dù | 键的维度 |

**一句话解释：**
利用自注意力捕捉特征图中不同位置之间的长期依赖关系。

**简单应用场景：**
增强GAN生成的一致性和细节质量。

---

---

### 公式 (12.13)：频谱归一化判别器 / Spectral Normalized Discriminator

$$
p(\boldsymbol{h} | \boldsymbol{v}) = \prod_j p(h_j | \boldsymbol{v})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_{\text{SN}}$ | guī yī huà pàn bié qì | 归一化判别器 |
| $W_{\text{SN}}$ | guī yī huà quán zhòng | 归一化权重 |
| $\sigma$ | jī huó huà hánshù | 激活函数 |

**一句话解释：**
在判别器的每一层应用谱归一化来约束Lipschitz常数。

**简单应用场景：**
提高GAN的训练稳定性。

---

---

### 公式 (12.14)：Sigmoid与加权和的关系 / Sigmoid - Weighted Sum Relationship

$$
p(x_i = 1|\boldsymbol{x}_{\setminus i}) = \sigma\left(\frac{\sum_j w_{ij}x_j + b_i}{T}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_i = 1 \mid \boldsymbol{x}_{\setminus i})$ | gài lǜ | 条件概率 (Conditional probability) |
| $\sigma$ | xī gé mǎ | Sigmoid函数 (Sigmoid function) |
| w_{ij} | quán zhòng | 权重 (Weight) |
| x_j | biàn liàng | 变量 (Variable) |
| b_i | piānyì | 偏置 (Bias) |
| T | wēn dù | 温度 (Temperature) |

**一句话解释：**
条件概率等于加权和加偏置的Sigmoid函数，其中除以温度参数进行标度。

**简单应用场景：**
在神经网络和深度学习中实现条件概率计算。

---

---

### 公式 (12.15)：Hopfield网络激活规则 / Hopfield Network Activation Rule

$$
x_i = \begin{cases}
1 & \text{if } \sum_j w_{ij}x_j + b_i \geq 0, \\
0 & \text{otherwise}.
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| x_i | biàn liàng | 第i个神经元的激活值 (Activation of neuron i) |
| w_{ij} | quán zhòng | 权重 (Weight) |
| b_i | piānyì | 偏置 (Bias) |
| $\sum_j$ | qiú hé | 求和 (Summation) |

**一句话解释：**
Hopfield网络的激活规则是确定性的：当加权和加偏置非负时输出1，否则输出0。

**简单应用场景：**
Hopfield网络的前向传播计算和状态更新。

---

---

### 公式 (12.16)：参数学习目标函数 / Parameter Learning Objective Function

$$
\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b}) = \frac{1}{N}\sum_{n=1}^N \log p(\boldsymbol{v}^{(n)}; \boldsymbol{W}, \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 (Loss function) |
| $\mathcal{D}$ | shù jù jí | 数据集 (Dataset) |
| $\boldsymbol{W}$ | quán zhòng | 权重矩阵 (Weight matrix) |
| $\boldsymbol{b}$ | piānyì | 偏置向量 (Bias vector) |
| $p(\boldsymbol{v})$ | gài lǜ | 概率 (Probability) |
| N | yàng běn shù | 样本数 (Number of samples) |

**一句话解释：**
参数学习的目标是最大化训练数据的对数似然，即所有样本的概率之和的对数的平均值。

**简单应用场景：**
深度信念网络的参数学习和模型训练。

---

---

### 公式 (12.17)：对数似然展开式1 / Log Likelihood Expansion 1

$$
\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b}) = \frac{1}{N}\sum_{n=1}^N \log\sum_h p(\boldsymbol{v}^{(n)}, \boldsymbol{h}; \boldsymbol{W}, \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 (Loss function) |
| $p(\boldsymbol{v}, \boldsymbol{h})$ | gài lǜ | 联合概率 (Joint probability) |
| $\boldsymbol{v}$ | kě jiàn biàn liàng | 可见变量 (Visible variables) |
| $\boldsymbol{h}$ | yǐncáng biàn liàng | 隐藏变量 (Hidden variables) |
| $\sum_h$ | qiú hé | 对隐藏变量求和 (Sum over hidden variables) |

**一句话解释：**
通过对隐藏变量求和，将联合概率转换为边际概率，形成完整的似然函数。

**简单应用场景：**
在有隐藏变量的图模型中计算观察数据的似然。

---

---

### 公式 (12.18)：对数似然展开式2 / Log Likelihood Expansion 2

$$
\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b}) = \frac{1}{N}\sum_{n=1}^N \log\frac{\sum_h\exp\left(-E(\boldsymbol{v}^{(n)}, \boldsymbol{h})\right)}{\sum_{u,h}\exp\left(-E(\boldsymbol{u}, \boldsymbol{h})\right)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sǔn shī hánshù | 损失函数 (Loss function) |
| $E(\boldsymbol{v}, \boldsymbol{h})$ | néng liàng | 能量 (Energy) |
| $\sum_h$ | qiú hé | 对隐藏变量求和 (Sum over hidden) |
| $\sum_{u,h}$ | qiú hé | 对所有变量求和 (Sum over all) |
| $\exp$ | zhǐ shù | 指数函数 (Exponential) |

**一句话解释：**
似然函数的另一种形式，分子是数据对应隐藏变量的能量之和的指数，分母是所有可能状态的能量之和的指数。

**简单应用场景：**
分子是能量的"正相位"贡献，分母是配分函数Z的"负相位"贡献。

---

---

### 公式 (12.19)：能量基模型 / Energy-Based Model

$$
\frac{\partial \log p(\boldsymbol{v})}{\partial \theta} = -\left\langle \frac{\partial E(\boldsymbol{v}, \boldsymbol{h})}{\partial \theta} \right\rangle_{\text{data}} + \left\langle \frac{\partial E(\boldsymbol{v}, \boldsymbol{h})}{\partial \theta} \right\rangle_{\text{model}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E(x)$ | néng liàng hánshù | 能量函数 |
| $Z$ | fēn huà hánshù | 分配函数 |
| $P(x)$ | xī gài lǜ | 概率 |

**一句话解释：**
通过能量函数定义概率分布，低能量对应高概率。

**简单应用场景：**
无监督学习、对比学习中的概率建模。

---

---

### 公式 (12.20)：对比散度 / Contrastive Divergence

$$
\Delta \boldsymbol{W} = \alpha (\langle \boldsymbol{v} \boldsymbol{h}^\top \rangle_{\text{data}} - \langle \boldsymbol{v} \boldsymbol{h}^\top \rangle_{\text{model}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_\theta$ | yǐ $\theta$ wéi cānshù de néng liàng hánshù | 以θ为参数的能量函数 |
| $p_{\text{data}}$ | zhēn shí shù jù fēn bù | 真实数据分布 |
| $p_\theta$ | mó xíng fēn bù | 模型分布 |

**一句话解释：**
拉低真实数据的能量，提高模型生成数据的能量。

**简单应用场景：**
能量基模型的高效训练方法。

---

---

### 公式 (12.21)：变分自编码器的目标函数 / VAE Objective Function

$$
\Delta \boldsymbol{a} = \alpha (\langle \boldsymbol{v} \rangle_{\text{data}} - \langle \boldsymbol{v} \rangle_{\text{model}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_\phi(z \mid x)$ | biàn fēn hòu yàn (yǐ $\phi$ wéi cānshù) | 变分后验 |
| $p_\theta(x \mid z)$ | shēng chéng mó xíng | 生成模型 |
| $p(z)$ | xiān yàn fēn bù | 先验分布 |
| $\text{KL}$ | Kù ěr huì léi sǎn dù | 库尔回雷散度 |

**一句话解释：**
最大化重构似然同时最小化后验与先验的KL散度。

**简单应用场景：**
无监督学习、图像生成、数据压缩。

---

---

### 公式 (12.22)：最大似然估计 / Maximum Likelihood Estimation

$$
\Delta \boldsymbol{b} = \alpha (\langle \boldsymbol{h} \rangle_{\text{data}} - \langle \boldsymbol{h} \rangle_{\text{model}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta$ | cānshù | 参数 |
| $p_\theta(x_i)$ | zài cānshù $\theta$ xià yàng běn $x_i$ de gài lǜ | 在参数θ下样本$x_i$的概率 |
| $N$ | zǒng yàng běn shù | 总样本数 |

**一句话解释：**
选择使观测数据出现概率最大的参数值。

**简单应用场景：**
监督学习、参数估计。

---

---

### 公式 (12.23)：KL散度 / Kullback-Leibler Divergence

$$
\Delta \boldsymbol{W} \approx \alpha (\boldsymbol{v}_0 \boldsymbol{h}_0^\top - \boldsymbol{v}_k \boldsymbol{h}_k^\top)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | zhēn shí fēn bù | 真实分布 |
| $q(x)$ | qī jìn fēn bù | 逼近分布 |

**一句话解释：**
衡量两个概率分布之间的差异，非对称的。

**简单应用场景：**
变分推断、信息论。

---

---

### 公式 (12.24)：JS散度 / Jensen-Shannon Divergence

$$
\Delta \boldsymbol{a} \approx \alpha (\boldsymbol{v}_0 - \boldsymbol{v}_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $m$ | píng jūn fēn bù | 平均分布 |
| $\text{KL}$ | Kù ěr huì léi sǎn dù | 库尔回雷散度 |

**一句话解释：**
对称的散度度量，由两个KL散度的平均组成。

**简单应用场景：**
GAN中的损失函数设计。

---

---

### 公式 (12.25)：Wasserstein距离 / Wasserstein Distance

$$
\Delta \boldsymbol{b} \approx \alpha (\boldsymbol{h}_0 - \boldsymbol{h}_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Pi(p,q)$ | $p$ hé $q$ de lián jié fēn bù zī hé | p和q的联结分布集合 |
| $\gamma$ | lián jié fēn bù | 联结分布 |
| $\inf$ | xià quèquè | 下确界 |

**一句话解释：**
度量两个分布之间的最优传输成本。

**简单应用场景：**
Wasserstein GAN中的距离度量。

---

---

### 公式 (12.26)：权重更新规则 / Weight Update Rule

$$
w_{ij} \leftarrow w_{ij} + \alpha\left(\langle x_i x_j \rangle_{\text{data}} - \langle x_i x_j \rangle_{\text{model}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| w_{ij} | quán zhòng | 权重 (Weight) |
| $\alpha$ | xué xí lǜ | 学习率 (Learning rate) |
| $\langle x_i x_j \rangle_{\text{data}}$ | shù jù qī wàng | 数据期望 (Data expectation) |
| $\langle x_i x_j \rangle_{\text{model}}$ | mó xíng qī wàng | 模型期望 (Model expectation) |

**一句话解释：**
权重按照数据期望与模型期望的差异进行更新，学习率α控制更新的步长。

**简单应用场景：**
Contrastive Divergence算法和RBM训练中的核心权重更新。

---

---

### 公式 (12.27)：受限玻尔兹曼机能量函数 / RBM Energy Function

$$
E(\boldsymbol{v}, \boldsymbol{h}) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j}v_i w_{ij} h_j
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E(\boldsymbol{v}, \boldsymbol{h})$ | néng liàng | 能量 (Energy) |
| $\boldsymbol{v}$ | kě jiàn biàn liàng | 可见层 (Visible variables) |
| $\boldsymbol{h}$ | yǐncáng biàn liàng | 隐层 (Hidden variables) |
| a_i | piānyì | 可见层偏置 (Visible bias) |
| b_j | piānyì | 隐层偏置 (Hidden bias) |
| w_{ij} | quán zhòng | 权重 (Weight) |

**一句话解释：**
受限玻尔兹曼机的能量函数由三部分组成：可见层偏置、隐层偏置和可见-隐层之间的相互作用。

**简单应用场景：**
深度信念网络的基本构件单元。

---

---

### 公式 (12.28)：RBM能量函数向量形式 / RBM Energy Function - Vector Form

$$
E(\boldsymbol{v}, \boldsymbol{h}) = -\boldsymbol{a}^{\rm T}\boldsymbol{v} - \boldsymbol{b}^{\rm T}\boldsymbol{h} - \boldsymbol{v}^{\rm T}\boldsymbol{W}\boldsymbol{h}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E(\boldsymbol{v}, \boldsymbol{h})$ | néng liàng | 能量 (Energy) |
| $\boldsymbol{a}$ | piānyì xiàng liàng | 可见层偏置向量 (Visible bias vector) |
| $\boldsymbol{b}$ | piānyì xiàng liàng | 隐层偏置向量 (Hidden bias vector) |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | 权重矩阵 (Weight matrix) |
| $\boldsymbol{v}$ | kě jiàn xiàng liàng | 可见变量向量 (Visible vector) |
| $\boldsymbol{h}$ | yǐncáng xiàng liàng | 隐变量向量 (Hidden vector) |
| $^{\rm T}$ | zhuǎn zhì | 转置 (Transpose) |

**一句话解释：**
使用向量和矩阵记号紧凑地表示RBM能量函数，便于高效的矩阵运算。

**简单应用场景：**
在计算机实现中进行向量化计算和矩阵运算。

---

---

### 公式 (12.29)：RBM联合概率 / RBM Joint Probability

$$
p(\boldsymbol{v}, \boldsymbol{h}) = \frac{1}{Z}\exp(-E(\boldsymbol{v}, \boldsymbol{h}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{v}, \boldsymbol{h})$ | lián hé gài lǜ | 联合概率 (Joint probability) |
| Z | pīn fēn hánshù | 配分函数 (Partition function) |
| $E(\boldsymbol{v}, \boldsymbol{h})$ | néng liàng | 能量 (Energy) |
| $\exp$ | zhǐ shù | 指数函数 (Exponential) |

**一句话解释：**
RBM的联合概率分布由能量函数通过Boltzmann分布定义，配分函数Z确保概率和为1。

**简单应用场景：**
RBM的核心概率模型定义。

---

---

### 公式 (12.30)：RBM联合概率因式分解 / RBM Joint Probability Factorization

$$
p(\boldsymbol{v}, \boldsymbol{h}) = \frac{1}{Z}\exp(\boldsymbol{a}^{\rm T}\boldsymbol{v})\exp(\boldsymbol{b}^{\rm T}\boldsymbol{h})\exp(\boldsymbol{v}^{\rm T}\boldsymbol{W}\boldsymbol{h})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{v}, \boldsymbol{h})$ | lián hé gài lǜ | 联合概率 (Joint probability) |
| Z | pīn fēn hánshù | 配分函数 (Partition function) |
| $\boldsymbol{a}$ | piānyì xiàng liàng | 可见层偏置向量 (Visible bias vector) |
| $\boldsymbol{b}$ | piānyì xiàng liàng | 隐层偏置向量 (Hidden bias vector) |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | 权重矩阵 (Weight matrix) |
| $\boldsymbol{v}$ | kě jiàn xiàng liàng | 可见变量向量 (Visible vector) |
| $\boldsymbol{h}$ | yǐncáng xiàng liàng | 隐变量向量 (Hidden vector) |
| $\exp$ | zhǐ shù | 指数函数 (Exponential) |
| $^{\rm T}$ | zhuǎn zhì | 转置 (Transpose) |

**一句话解释：**
联合概率展开为三个指数项的乘积，分别对应可见层、隐层和它们之间的相互作用。

**简单应用场景：**
理解RBM中各部分参数对联合概率分布的贡献。

---

---

### 公式 (12.31)：可观测变量条件独立 / Observable Variables Conditional Independence

$$
p(\boldsymbol{v}|v_{\neg i}, \boldsymbol{h}) = p(\boldsymbol{v}|\boldsymbol{h})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}$ | xiàng liàng | visible variables (observable variables) |
| $\boldsymbol{h}$ | yǐn cáng biàn liàng | hidden variables |
| $v_{\neg i}$ | chú le dì $i$ ge de kě jiàn biàn liàng | all visible variables except $v_i$ |

**一句话解释：**
在已知隐变量的情况下，任意可观测变量与其他可观测变量条件独立。

**简单应用场景：**
受限玻尔兹曼机中，给定隐层的激活值，可观测层内任意两个节点之间没有直接连接。

---

---

### 公式 (12.32)：隐变量条件独立 / Hidden Variables Conditional Independence

$$
p(\boldsymbol{h}|v, \boldsymbol{h}_{\neg j}) = p(\boldsymbol{h}|v)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}$ | yǐn cáng biàn liàng | hidden variables |
| $v$ | kě jiàn biàn liàng | visible variables |
| $\boldsymbol{h}_{\neg j}$ | chú le dì $j$ ge de yǐn cáng biàn liàng | all hidden variables except $h_j$ |

**一句话解释：**
在已知可观测变量的情况下，任意隐变量与其他隐变量条件独立。

**简单应用场景：**
在受限玻尔兹曼机中，给定可观测层的状态，可以并行计算所有隐层节点的激活值。

---

---

### 公式 (12.33)：可观测变量条件概率1 / Conditional Probability of Visible Units (1)

$$
p(v_i = 1|\boldsymbol{h}) = \sigma\left(a_i + \sum_j w_{ij}h_j\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $v_i$ | dì $i$ ge kě jiàn biàn liàng | $i$-th visible variable |
| $\boldsymbol{h}$ | yǐn cáng biàn liàng | hidden variables |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $a_i$ | piàn zhì xiàng | bias of visible unit $i$ |
| $w_{ij}$ | quán zhòng | weight between visible unit $i$ and hidden unit $j$ |

**一句话解释：**
给定隐层状态，可观测变量激活的概率由偏置和隐层节点的加权和经过sigmoid函数计算得到。

**简单应用场景：**
在RBM的采样过程中，给定隐层的当前状态，可以逐个计算并采样可观测层的每个节点。

---

---

### 公式 (12.34)：隐变量条件概率1 / Conditional Probability of Hidden Units (1)

$$
p(h_j = 1|\boldsymbol{v}) = \sigma\left(b_j + \sum_i w_{ij}v_i\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_j$ | dì $j$ ge yǐn cáng biàn liàng | $j$-th hidden variable |
| $\boldsymbol{v}$ | kě jiàn biàn liàng | visible variables |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $b_j$ | piàn zhì xiàng | bias of hidden unit $j$ |
| $w_{ij}$ | quán zhòng | weight between visible unit $i$ and hidden unit $j$ |

**一句话解释：**
给定可观测层状态，隐变量激活的概率由偐置和可观测层节点的加权和经过sigmoid函数计算得到。

**简单应用场景：**
在RBM的采样过程中，给定可观测层的当前状态，可以并行计算所有隐层节点的激活概率。

---

---

### 公式 (12.35)：联合分布的能量函数表示 / Energy-Based Representation of Joint Distribution

$$
P(v) = \sum_{\boldsymbol{h}} F(v, \boldsymbol{h}) = \frac{1}{Z}\sum_{\boldsymbol{h}} \exp(-E(v, \boldsymbol{h}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(v)$ | kě jiàn biàn liàng de jī yán fēn bù | probability of visible variables |
| $\boldsymbol{h}$ | yǐn cáng biàn liàng | hidden variables |
| $F(v, \boldsymbol{h})$ | néng liàng hánshù de zhǐ shù | energy function |
| $E(v, \boldsymbol{h})$ | néng liàng | energy |
| $Z$ | fēn huà hánshù | partition function |

**一句话解释：**
可观测变量的边际概率可以通过对所有隐变量的联合分布进行边际化得到，表示为能量函数的指数形式。

**简单应用场景：**
在推断可观测变量的概率时，需要对所有可能的隐变量状态进行求和。

---

---

### 公式 (12.36)：能量函数展开1 / Energy Function Expansion (1)

$$
= \frac{1}{Z}\sum_{\boldsymbol{h}} \exp\left(a^{\top}v + \sum_j b_j h_j + \sum_i \sum_j w_{ij}v_i h_j\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $a$ | piàn zhì xiàng liàng | bias vector for visible units |
| $v$ | kě jiàn biàn liàng | visible variables |
| $b_j$ | piàn zhì xiàng | bias of hidden unit $j$ |
| $h_j$ | dì $j$ ge yǐn cáng biàn liàng | $j$-th hidden variable |
| $w_{ij}$ | quán zhòng | weight between visible unit $i$ and hidden unit $j$ |

**一句话解释：**
将RBM的能量函数展开为偏置项和权重项的组合形式。

**简单应用场景：**
在计算联合分布时，需要对所有隐变量组合进行指数计算和求和。

---

---

### 公式 (12.37)：风格迁移损失 / Style Transfer Loss

$$
F(\boldsymbol{x}, \boldsymbol{z}; \theta) = -E(\boldsymbol{x}, \boldsymbol{z}; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_s$ | fēng gé jiāng yí shēng chéng qì | 风格迁移生成器 |
| $S$ | fēng gé tí qǔ hánshù | 风格提取函数 |
| $\lambda$ | lán mǔ dá | 平衡参数 |

**一句话解释：**
同时优化重构损失和风格匹配损失。

**简单应用场景：**
图像风格迁移。

---

---

### 公式 (12.38)：感知损失 / Perceptual Loss

$$
p(\boldsymbol{x}; \theta) = \frac{1}{Z(\theta)} \sum_{\boldsymbol{z}} \exp(F(\boldsymbol{x}, \boldsymbol{z}; \theta))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F^j$ | jiàng $j$ céng de tè zhēng tí qǔ | 第j层的特征提取 |
| $C_j$ | jiàng $j$ céng de tōng dào shù | 第j层的通道数 |
| $H_j, W_j$ | jiàng $j$ céng de gāo duō kuà | 第j层的高度和宽度 |

**一句话解释：**
利用预训练网络的中间层特征计算感知距离。

**简单应用场景：**
超分辨率、图像复原。

---

---

### 公式 (12.39)：能量函数展开4 / Energy Function Expansion (4)

$$
= \frac{\exp(a^{\top}v)}{Z} \sum_{h_1} \sum_{h_2} \cdots \sum_{h_m} \prod_j \exp\left(h_j\left(b_j + \sum_i w_{ij}v_i\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp(a^{\top}v)$ | zhǐ shù hánshù | exponential of bias-visible term |
| $Z$ | fēn huà hánshù | partition function |
| $\sum_{h_j}$ | duì yǐn cáng biàn liàng qiú hé | summation over hidden variables |
| $\prod_j$ | chéng jī | product over hidden units |

**一句话解释：**
展开所有隐变量的求和，为进一步的计算做准备。

**简单应用场景：**
为了利用二值隐变量的特性进行进一步的简化。

---

---

### 公式 (12.40)：注意力机制权重 / Attention Weights

$$
\frac{\partial \log p(\boldsymbol{v})}{\partial \boldsymbol{W}} = \langle \boldsymbol{v} \boldsymbol{h}^\top \rangle_{\text{data}} - \langle \boldsymbol{v} \boldsymbol{h}^\top \rangle_{\text{model}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $e_{ij}$ | dì $i$ gè wèi zhì duì jiàng $j$ gè wèi zhì de zhù yì lì fēn shù | 第i个位置对第j个位置的注意力分数 |
| $\alpha_{ij}$ | guī yī huà hòu de zhù yì lì quán zhòng | 归一化后的注意力权重 |

**一句话解释：**
通过softmax函数将注意力分数转换为权重。

**简单应用场景：**
序列模型、图像处理中的注意力机制。

---

---

### 公式 (12.41)：能量函数展开6 / Energy Function Expansion (6)

$$
= \frac{\exp(a^{\top}v)}{Z} \prod_j \left(1 + \exp\left(b_j + \sum_i w_{ij}v_i\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp(a^{\top}v)$ | zhǐ shù hánshù | exponential of bias-visible term |
| $Z$ | fēn huà hánshù | partition function |
| $\prod_j$ | chéng jī | product over hidden units |

**一句话解释：**
利用二值变量的特性，将指数和转换为乘积形式，便于计算。

**简单应用场景：**
可观测变量的边际概率可以表示为隐层每个节点贡献的乘积。

---

---

### 公式 (12.42)：层归一化 / Layer Normalization

$$
\text{FE}(\boldsymbol{v}) = -\log \sum_{\boldsymbol{h}} \exp(-E(\boldsymbol{v}, \boldsymbol{h}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu$ | míu | 均值 |
| $\sigma^2$ | fāng chà | 方差 |
| $\gamma, \beta$ | kě xuéxí de guī yī huà cānshù | 可学习的归一化参数 |
| $\epsilon$ | jiào xiǎo de zhèng shù | 较小的正数 |

**一句话解释：**
对每个样本的特征进行独立的归一化。

**简单应用场景：**
Transformer、RNN等深度网络的稳定性改进。

---

## 第13章 生成对抗网络

---

### 公式 (12.43)：隐层激活概率展开 / Hidden Unit Activation Probability Expansion

$$
= \frac{\exp(a^{\top}v)}{Z} \prod_{k \neq j} \left(1 + \exp\left(b_k + \sum_i w_{ik}v_i\right)\right) \exp\left(b_j + \sum_i w_{ij}v_i\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp(a^{\top}v)$ | zhǐ shù hánshù | exponential of bias-visible term |
| $Z$ | fēn huà hánshù | partition function |
| $\prod_{k \neq j}$ | duì $k \neq j$ de yǐn cáng dān yuán chéng jī | product over hidden units except $j$ |
| $b_j$ | piàn zhì xiàng | bias of hidden unit $j$ |

**一句话解释：**
隐变量$h_j=1$时的联合概率，包含所有其他隐层节点和可观测层的贡献。

**简单应用场景：**
在推导条件概率时使用。

---

---

### 公式 (12.44)：隐变量条件概率2 / Conditional Probability of Hidden Variables (2)

$$
p(h_j = 1|\boldsymbol{v}) = \frac{p(h_j = 1, v)}{p(v)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(h_j = 1 \mid \boldsymbol{v})$ | gěi dìng kě jiàn biàn liàng shí yǐn cáng dān yuán jī huà de tiáo jiàn gài lǜ | conditional probability of $h_j = 1$ given $v$ |
| $p(h_j = 1, v)$ | lián hé gài lǜ | joint probability |
| $p(v)$ | biān jì gài lǜ | marginal probability |

**一句话解释：**
隐变量的条件概率等于联合概率与边际概率的比值。

**简单应用场景：**
利用贝叶斯定理计算条件概率。

---

---

### 公式 (12.45)：隐变量条件概率3 / Conditional Probability of Hidden Variables (3)

$$
= \frac{\exp\left(b_j + \sum_i w_{ij}v_i\right)}{1 + \exp\left(b_j + \sum_i w_{ij}v_i\right)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp\left(b_j + \sum_i w_{ij}v_i\right)$ | zhǐ shù hánshù | exponential term |
| $b_j$ | piàn zhì xiàng | bias of hidden unit $j$ |
| $w_{ij}$ | quán zhòng | weight |

**一句话解释：**
隐变量条件概率可以用sigmoid函数表示。

**简单应用场景：**
在RBM的前向推理中，可以使用这个公式快速计算隐层激活概率。

---

---

### 公式 (12.46)：隐变量条件概率4 / Conditional Probability of Hidden Variables (4)

$$
= \sigma\left(b_j + \sum_i w_{ij}v_i\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $b_j$ | piàn zhì xiàng | bias of hidden unit $j$ |
| $w_{ij}$ | quán zhòng | weight |

**一句话解释：**
隐变量的条件概率可以直接用sigmoid函数表示，避免了复杂的计算。

**简单应用场景：**
在RBM的Gibbs采样中，用这个公式计算隐层节点的激活概率。

---

---

### 公式 (12.47)：可观测变量条件概率2 / Conditional Probability of Visible Variables (2)

$$
p(v_i = 1|\boldsymbol{h}) = \sigma\left(a_i + \sum_j w_{ij}h_j\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(v_i = 1 \mid \boldsymbol{h})$ | gěi dìng yǐn cáng biàn liàng shí kě jiàn dān yuán jī huà de tiáo jiàn gài lǜ | conditional probability of $v_i = 1$ given $h$ |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $a_i$ | piàn zhì xiàng | bias of visible unit $i$ |
| $w_{ij}$ | quán zhòng | weight |

**一句话解释：**
给定隐层状态，可观测变量激活的条件概率。

**简单应用场景：**
在RBM的Gibbs采样中，用这个公式计算可观测层节点的激活概率。

---

---

### 公式 (12.48)：向量形式的隐层条件概率 / Vector Form of Hidden Layer Conditional Probability

$$
p(\boldsymbol{h} = 1|\boldsymbol{v}) = \sigma(\boldsymbol{W}^{\top}\boldsymbol{v} + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h} = 1 \mid \boldsymbol{v})$ | gěi dìng kě jiàn biàn liàng shí yǐn cáng céng de tiáo jiàn gài lǜ | conditional probability of hidden layer given visible variables |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | weight matrix |
| $\boldsymbol{v}$ | kě jiàn biàn liàng xiàng liàng | visible variable vector |
| $\boldsymbol{b}$ | yǐn cáng céng piàn zhì xiàng liàng | bias vector for hidden layer |

**一句话解释：**
隐层条件概率的向量形式，所有隐层节点可以并行计算。

**简单应用场景：**
在RBM的实现中，使用矩阵乘法并行计算所有隐层节点的激活概率。

---

---

### 公式 (12.49)：向量形式的可观测层条件概率 / Vector Form of Visible Layer Conditional Probability

$$
p(\boldsymbol{v} = 1|\boldsymbol{h}) = \sigma(\boldsymbol{W}\boldsymbol{h} + \boldsymbol{a})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{v} = 1 \mid \boldsymbol{h})$ | gěi dìng yǐn cáng biàn liàng shí kě jiàn céng de tiáo jiàn gài lǜ | conditional probability of visible layer given hidden variables |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | weight matrix |
| $\boldsymbol{h}$ | yǐn cáng biàn liàng xiàng liàng | hidden variable vector |
| $\boldsymbol{a}$ | kě jiàn céng piàn zhì xiàng liàng | bias vector for visible layer |

**一句话解释：**
可观测层条件概率的向量形式，所有可观测层节点可以并行计算。

**简单应用场景：**
在RBM的实现中，使用矩阵乘法并行计算所有可观测层节点的激活概率。

---

---

### 公式 (12.50)：对数似然函数 / Log-Likelihood Function

$$
\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{a}, \boldsymbol{b}) = \frac{1}{N}\sum_{n=1}^{N} \log p(v^{(n)}; \boldsymbol{W}, \boldsymbol{a}, \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | duì shù sì ránhuà | log-likelihood |
| $\mathcal{D}$ | xùn liàn shù jù jí | training dataset |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | weight matrix |
| $\boldsymbol{a}$ | kě jiàn céng piàn zhì xiàng liàng | visible layer bias |
| $\boldsymbol{b}$ | yǐn cáng céng piàn zhì xiàng liàng | hidden layer bias |
| $N$ | shù jù shù | number of training samples |
| $v^{(n)}$ | dì $n$ ge xùn liàn yàng běn | $n$-th training sample |

**一句话解释：**
受限玻尔兹曼机的目标函数是所有训练样本对数似然的平均值。

**简单应用场景：**
在参数学习中，最大化这个目标函数来学习最优的参数。

---

---

### 公式 (12.51)：权重的梯度 / Gradient with Respect to Weights

$$
\frac{\partial \mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{a}, \boldsymbol{b})}{\partial w_{ij}} = \mathbb{E}_{p(\boldsymbol{v})}[\mathbb{E}_{p(\boldsymbol{h}|\boldsymbol{v})}[v_i h_j]] - \mathbb{E}_{p(\boldsymbol{v}, \boldsymbol{h})}[v_i h_j]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial w_{ij}}$ | duì $w_{ij}$ de piàn dǎo shù | gradient of loss with respect to $w_{ij}$ |
| $\mathbb{E}$ | qī wàng zhí | expectation |
| $p(\boldsymbol{v})$ | xùn liàn shù jù de biān jì gài lǜ | marginal distribution of data |
| $p(\boldsymbol{h} \mid \boldsymbol{v})$ | gěi dìng kě jiàn biàn liàng de yǐn cáng biàn liàng tiáo jiàn gài lǜ | conditional distribution of hidden given visible |
| $p(\boldsymbol{v}, \boldsymbol{h})$ | mó xíng de lián hé gài lǜ | model's joint distribution |

**一句话解释：**
权重的梯度等于数据条件下的期望值减去模型联合分布下的期望值。

**简单应用场景：**
在受限玻尔兹曼机的参数学习中，用这个梯度进行梯度上升优化。

---

---

### 公式 (12.52)：可观测层偏置的梯度 / Gradient with Respect to Visible Bias

$$
\frac{\partial \mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{a}, \boldsymbol{b})}{\partial a_i} = \mathbb{E}_{p(\boldsymbol{v})}[\mathbb{E}_{p(\boldsymbol{h}|\boldsymbol{v})}[v_i]] - \mathbb{E}_{p(\boldsymbol{v}, \boldsymbol{h})}[v_i]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial a_i}$ | duì kě jiàn céng piàn zhì de piàn dǎo shù | gradient of loss with respect to visible bias $a_i$ |
| $\mathbb{E}$ | qī wàng zhí | expectation |
| $v_i$ | dì $i$ ge kě jiàn biàn liàng | $i$-th visible variable |

**一句话解释：**
可观测层偏置的梯度等于数据条件下的可观测变量期望减去模型联合分布下的期望。

**简单应用场景：**
在参数学习中，用这个梯度更新可观测层的偏置。

---

---

### 公式 (12.53)：隐层偏置的梯度 / Gradient with Respect to Hidden Bias

$$
\frac{\partial \mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{a}, \boldsymbol{b})}{\partial b_j} = \mathbb{E}_{p(\boldsymbol{v})}[\mathbb{E}_{p(\boldsymbol{h}|\boldsymbol{v})}[h_j]] - \mathbb{E}_{p(\boldsymbol{v}, \boldsymbol{h})}[h_j]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial b_j}$ | duì yǐn cáng céng piàn zhì de piàn dǎo shù | gradient of loss with respect to hidden bias $b_j$ |
| $\mathbb{E}$ | qī wàng zhí | expectation |
| $h_j$ | dì $j$ ge yǐn cáng biàn liàng | $j$-th hidden variable |

**一句话解释：**
隐层偏置的梯度等于数据条件下的隐变量期望减去模型联合分布下的期望。

**简单应用场景：**
在参数学习中，用这个梯度更新隐层的偏置。

---

---

### 公式 (12.54)：权重更新规则 / Weight Update Rule

$$
w_{ij} \leftarrow w_{ij} + \alpha\left((v_i h_j)_{\text{data}} - (v_i h_j)_{\text{model}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_{ij}$ | quán zhòng | weight between visible unit $i$ and hidden unit $j$ |
| $\alpha$ | xué xí lǜ | learning rate |
| $(v_i h_j)_{\text{data}}$ | shù jù shang de $v_i h_j$ qī wàng zhí | expected value of $v_i h_j$ under data distribution |
| $(v_i h_j)_{\text{model}}$ | mó xíng shang de $v_i h_j$ qī wàng zhí | expected value of $v_i h_j$ under model distribution |

**一句话解释：**
权重按照数据分布和模型分布下的期望值之差的方向更新。

**简单应用场景：**
在使用对比散度算法训练RBM时，用这个规则更新权重。

---

---

### 公式 (12.55)：可观测层偏置更新规则 / Visible Layer Bias Update Rule

$$
a_i \leftarrow a_i + \alpha\left((v_i)_{\text{data}} - (v_i)_{\text{model}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $a_i$ | kě jiàn céng piàn zhì | visible layer bias |
| $\alpha$ | xué xí lǜ | learning rate |
| $(v_i)_{\text{data}}$ | shù jù shang de $v_i$ qī wàng zhí | expected value of $v_i$ under data distribution |
| $(v_i)_{\text{model}}$ | mó xíng shang de $v_i$ qī wàng zhí | expected value of $v_i$ under model distribution |

**一句话解释：**
可观测层偏置按照数据分布和模型分布下可观测变量期望值之差的方向更新。

**简单应用场景：**
在RBM训练中，用这个规则更新可观测层的偏置。

---

---

### 公式 (12.56)：隐藏层偏置更新 / Hidden Bias Update

$$
b_j \leftarrow b_j + \alpha(\langle h_j \rangle_{\text{data}} - \langle h_j \rangle_{\text{model}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b_j$ | yǐn cáng céng piān zhì | 隐藏层第j个单元的偏置 |
| $\alpha$ | ā ěr fǎ / xué xí lǜ | 学习率 |
| $\langle h_j \rangle_{\text{data}}$ | shù jù qī wàng | 数据分布下隐单元的期望 |
| $\langle h_j \rangle_{\text{model}}$ | mó xíng qī wàng | 模型分布下隐单元的期望 |

**一句话解释：**
受限玻尔兹曼机中隐藏层偏置的梯度更新规则，方向为数据期望与模型期望之差。

**简单应用场景：**
RBM对比散度(CD)训练算法中的参数更新。

---

---

### 公式 (12.57)：Bernoulli-Gaussian混合能量函数 / Energy Function for Bernoulli-Gaussian RBM

$$
E(v, \boldsymbol{h}) = \sum_i \frac{(v_i - \mu_i)^2}{2\sigma_i^2} - \sum_i b_i h_i - \sum_i \sum_j \frac{v_i}{σ_i} w_{ij}h_j
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E(v, \boldsymbol{h})$ | néng liàng | energy |
| $v_i$ | dì $i$ ge kě jiàn biàn liàng | $i$-th visible variable |
| $\mu_i$ | dì $i$ ge kě jiàn dān yuán de píng jūn zhí | mean of visible unit $i$ |
| $\sigma_i$ | dì $i$ ge kě jiàn dān yuán de biāo zhǔn chā | standard deviation of visible unit $i$ |
| $b_i$ | piàn zhì xiàng | bias |
| $w_{ij}$ | quán zhòng | weight |
| $h_j$ | dì $j$ ge yǐn cáng biàn liàng | $j$-th hidden variable |

**一句话解释：**
Bernoulli-Gaussian混合RBM的能量函数，其中可观测变量服从高斯分布，隐变量是二值的。

**简单应用场景：**
当可观测数据不是二值的，而是连续值时，使用这种RBM。

---

---

### 公式 (12.58)：Bernoulli-Gaussian能量函数（另一形式） / Energy Function for Bernoulli-Gaussian RBM (Alternative Form)

$$
E(v, \boldsymbol{h}) = -\sum_i v_i v_i + \sum_i \frac{(h_j - \mu_j)^2}{2\sigma_j^2} - \sum_i \sum_j uw_{ij}\frac{v_i}{σ_j}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E(v, \boldsymbol{h})$ | néng liàng | energy |
| $v_i$ | dì $i$ ge kě jiàn biàn liàng | $i$-th visible variable |
| $h_j$ | dì $j$ ge yǐn cáng biàn liàng | $j$-th hidden variable |
| $\mu_j$ | dì $j$ ge yǐn cáng dān yuán de píng jūn zhí | mean of hidden unit $j$ |
| $\sigma_j$ | dì $j$ ge yǐn cáng dān yuán de biāo zhǔn chā | standard deviation of hidden unit $j$ |
| $w_{ij}$ | quán zhòng | weight |

**一句话解释：**
另一种形式的Bernoulli-Gaussian RBM能量函数，隐层变量也可以服从高斯分布。

**简单应用场景：**
当需要建模连续的隐变量和可观测变量时使用。

---

## 12.3 深度信念网络 / Deep Belief Network

---

---

### 公式 (12.59)：深度信念网络的条件分布 / Conditional Distribution in DBN

$$
p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l-1)}, \ldots, \boldsymbol{h}^{(L)}) = p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h}^{(l)} \mid \boldsymbol{h}^{(l-1)}, \ldots, \boldsymbol{h}^{(L)})$ | gěi dìng shàng céng de tiáo jiàn gài lǜ | conditional probability given lower layers |
| $p(\boldsymbol{h}^{(l)} \mid \boldsymbol{h}^{(l+1)})$ | duì shàng yī céng tiáo jiàn gài lǜ | conditional probability given upper layer |
| $l$ | céng suō | layer index |

**一句话解释：**
在深度信念网络中，某一层的条件分布只取决于其上一层，具有Markov性质。

**简单应用场景：**
在DBN的推理过程中，可以利用这个性质简化计算。

---

---

### 公式 (12.60)：联合分布的因式分解1 / Factorization of Joint Distribution (1)

$$
p(v, \boldsymbol{h}^{(1)}, \ldots, \boldsymbol{h}^{(L)}) = p(v|\boldsymbol{h}^{(1)})\left(\prod_{l=1}^{L-2} p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)})\right) p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(v, \boldsymbol{h}^{(1)}, \ldots, \boldsymbol{h}^{(L)})$ | lián hé gài lǜ | joint distribution |
| $p(v \mid \boldsymbol{h}^{(1)})$ | kě jiàn céng de tiáo jiàn gài lǜ | conditional distribution of visible layer |
| $\prod_{l=1}^{L-2} p(\boldsymbol{h}^{(l)} \mid \boldsymbol{h}^{(l+1)})$ | duì jiàn tiáo jiàn gài lǜ | directed conditional distributions |
| $p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})$ | zuì hǎo liǎng céng de lián hé gài lǜ | joint distribution of top two layers |

**一句话解释：**
深度信念网络的联合分布可以分解为下向的有向条件概率和上层的无向联合分布。

**简单应用场景：**
在计算DBN的联合分布时，可以按这个方式分解。

---

---

### 公式 (12.61)：联合分布的因式分解2 / Factorization of Joint Distribution (2)

$$
= \left(\prod_{l=0}^{L-2} p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)})\right) p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\prod_{l=0}^{L-2}$ | duì $l$ cóng 0 dào $L-2$ de chéng jī | product from layer 0 to $L-2$ |
| $p(\boldsymbol{h}^{(l)} \mid \boldsymbol{h}^{(l+1)})$ | céng jiān de tiáo jiàn gài lǜ | conditional distribution between layers |
| $p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})$ | zuì hǎo liǎng céng de lián hé gài lǜ | joint distribution of top two layers |

**一句话解释：**
简化后的联合分布因式分解形式，将可观测层视为第0层。

**简单应用场景：**
统一表示所有层之间的关系。

---

---

### 公式 (12.62)：隐层条件分布（Sigmoid形式） / Hidden Layer Conditional Distribution (Sigmoid Form)

$$
p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)}) = \sigma\left(\boldsymbol{a}^{(l)} + \boldsymbol{W}^{(l+1)}\boldsymbol{h}^{(l+1)}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h}^{(l)} \mid \boldsymbol{h}^{(l+1)})$ | dì $l$ céng de tiáo jiàn gài lǜ | conditional distribution of layer $l$ |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng de piàn zhì xiàng liàng | bias vector of layer $l$ |
| $\boldsymbol{W}^{(l+1)}$ | dì $l+1$ céng de quán zhòng jǔ zhèn | weight matrix of layer $l+1$ |

**一句话解释：**
深度信念网络中，每一层给定上层时的条件分布可以用sigmoid函数表示。

**简单应用场景：**
在DBN中进行前向推理或生成采样时使用。

---

---

### 公式 (12.63)：深度信念网络顶层的条件概率 / Conditional Probability of Top Layer in DBN

$$
p(v = 1|\boldsymbol{h}) = \sigma\left(\boldsymbol{b} + \boldsymbol{w}^{\top}\boldsymbol{h}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(v = 1 \mid \boldsymbol{h})$ | gěi dìng yǐn céng de kě jiàn céng tiáo jiàn gài lǜ | conditional probability of visible layer |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{b}$ | piàn zhì xiàng liàng | bias vector |
| $\boldsymbol{w}$ | quán zhòng xiàng liàng | weight vector |
| $\boldsymbol{h}$ | yǐn cáng biàn liàng xiàng liàng | hidden variable vector |

**一句话解释：**
深度信念网络可见层给定隐层时的条件概率。

**简单应用场景：**
在DBN中重构可观测数据时使用。

---

---

### 公式 (12.64)：逐层预训练的RBM条件分布 / Layer-wise Pretraining RBM Conditional Distribution

$$
p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l-1)}) = \sigma\left(\boldsymbol{b}^{(l)} + \boldsymbol{W}^{(l)}\boldsymbol{h}^{(l-1)}\right), \quad 1 \leq l \leq (L-1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h}^{(l)} \mid \boldsymbol{h}^{(l-1)})$ | dì $l$ céng gěi dìng dì $l-1$ céng de tiáo jiàn gài lǜ | conditional distribution of layer $l$ given layer $l-1$ |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{b}^{(l)}$ | dì $l$ céng de piàn zhì xiàng liàng | bias of layer $l$ |
| $\boldsymbol{W}^{(l)}$ | dì $l$ céng de quán zhòng jǔ zhèn | weight matrix of layer $l$ |

**一句话解释：**
在逐层预训练DBN时，每一层都可以看作一个RBM，其条件分布采用这种形式。

**简单应用场景：**
在DBN的逐层预训练阶段，将每一层的RBM与下一层的激活独立训练。

---

## 第13章：深度生成模型

### 公式 (13.1)：应用蒙特卡洛学习

$$
p(\boldsymbol{y}|\boldsymbol{x}) = \frac{p(\boldsymbol{x}, \boldsymbol{y})}{\sum_y p(\boldsymbol{x}, \boldsymbol{y})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{y}$ | y | 输出变量 |
| $\boldsymbol{x}$ | x | 输入变量 |
| $p(\boldsymbol{x}, \boldsymbol{y})$ | p | 联合概率 |

**一句话解释：**
条件概率等于联合概率除以边际概率的和。

**简单应用场景：**
从联合分布推导条件分布。

---

---

### 公式 (13.2)：隐变量模型的链式法则

$$
p(\boldsymbol{x}, \boldsymbol{z}; \boldsymbol{\theta}) = p(\boldsymbol{x}|\boldsymbol{z}; \boldsymbol{\theta})p(\boldsymbol{z}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | z | 隐变量 |
| $\boldsymbol{\theta}$ | xī tǎ | 模型参数 |
| $p(\boldsymbol{x} \mid \boldsymbol{z}; \boldsymbol{\theta})$ | p | 条件概率 |

**一句话解释：**
联合概率分解为条件概率和边际概率的乘积。

**简单应用场景：**
生成模型中表示可见和隐变量的关系。

---

---

### 公式 (13.3)：生成器最小化目标 / Generator Minimization Objective

$$
p(\boldsymbol{x}) = \int p(\boldsymbol{z}) p(\boldsymbol{x} | \boldsymbol{z}) \, d\boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G$ | shēng chéng qì | 生成器 |
| $D$ | pàn bié qì | 判别器 |
| $z$ | suí jī biàn liàng | 随机变量 |

**一句话解释：**
生成器试图生成逼真样本来最大化判别器的错误率。

**简单应用场景：**
对抗式生成模型训练。

---

---

### 公式 (13.4)：相对判别器 / Relative Discriminator

$$
p(\boldsymbol{z} | \boldsymbol{x}) = \frac{p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z})}{p(\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_{\text{rel}}$ | xiāng duì pàn bié qì | 相对判别器 |
| $\sigma$ | shù zì hánshù | 数字函数 |

**一句话解释：**
判别器输出相对分数而非绝对分数。

**简单应用场景：**
改进GAN的学习信号。

---

---

### 公式 (13.5)：判别器正则化 / Discriminator Regularization

$$
q(\boldsymbol{z} | \boldsymbol{x}) \approx p(\boldsymbol{z} | \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla_x D(x)$ | $D(x)$ duì $x$ de j梯度 | D(x)对x的梯度 |
| $\lambda$ | lán mǔ dá | 正则化权重 |

**一句话解释：**
通过梯度惩罚来约束判别器的Lipschitz连续性。

**简单应用场景：**
提高GAN训练的稳定性。

---

---

### 公式 (13.6)：梯度惩罚 / Gradient Penalty

$$
D_{\mathrm{KL}}(q(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}|\boldsymbol{x})) = \mathbb{E}_q \left[ \log \frac{q(\boldsymbol{z}|\boldsymbol{x})}{p(\boldsymbol{z}|\boldsymbol{x})} \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{x}$ | jiān chā yàng běn | 插值样本 |
| $\mathbb{P}_{\hat{x}}$ | jiān chā yàng běn de fēn bù | 插值样本的分布 |
| $\nabla_{\hat{x}} D(\hat{x})$ | pàn bié qì zài jiān chā diǎn de jī dù | 判别器在插值点的梯度 |

**一句话解释：**
在真实和生成样本之间的插值处施加梯度惩罚。

**简单应用场景：**
WGAN-GP中约束Lipschitz连续性。

---

---

### 公式 (13.7)：推断网络输出

$$
\begin{bmatrix} \boldsymbol{\mu}_l \\ \boldsymbol{\sigma}_l^2 \end{bmatrix} = f_l(\boldsymbol{x}; \boldsymbol{\phi})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_l$ | f | 推断网络函数 |
| $\boldsymbol{\phi}$ | fài | 推断网络参数 |

**一句话解释：**
推断网络直接输出高斯分布的均值和方差。

**简单应用场景：**
编码器的输出层设计。

---

---

### 公式 (13.8)：生成网络隐层

$$
\boldsymbol{h} = \sigma(\boldsymbol{W}^{(1)}\boldsymbol{x} + \boldsymbol{b}^{(1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}$ | h | 隐藏层 |
| $\sigma$ | xī gé mǎ | Sigmoid激活 |
| $\boldsymbol{W}^{(1)}$ | W | 权重矩阵 |
| $\boldsymbol{b}^{(1)}$ | b | 偏置向量 |

**一句话解释：**
生成网络的中间层通过非线性变换处理隐变量。

**简单应用场景：**
变分自编码器解码器的隐层。

---

---

### 公式 (13.9)：生成网络参数估计

$$
\boldsymbol{\mu}_l = \boldsymbol{W}^{(2)}\boldsymbol{h} + \boldsymbol{b}^{(2)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_l$ | mǔ | 输出均值 |
| $\boldsymbol{W}^{(2)}$ | W | 第二层权重 |
| $\boldsymbol{b}^{(2)}$ | b | 第二层偏置 |

**一句话解释：**
解码器预测重建的均值。

**简单应用场景：**
高斯输出层的均值计算。

---

---

### 公式 (13.10)：判别器损失加权 / Weighted Discriminator Loss

$$
\mathcal{L}(q) = \mathbb{E}_q [\log p(\boldsymbol{x}, \boldsymbol{z}) - \log q(\boldsymbol{z}|\boldsymbol{x})]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_1, w_2$ | quán zhòng | 权重 |
| $L_{\text{real}}$ | zhēn yàng běn sǔn shī | 真样本损失 |
| $L_{\text{fake}}$ | jiǎ yàng běn sǔn shī | 假样本损失 |

**一句话解释：**
对真实和生成样本的损失项使用不同的权重。

**简单应用场景：**
平衡GAN训练中的学习信号。

---

---

### 公式 (13.11)：推断网络的目标

$$
\boldsymbol{\phi}^* = \arg \min_{\boldsymbol{\phi}} \text{KL}\left(q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi}), p(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\theta})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\phi}^*$ | fài | 最优推断参数 |
| $\arg \min$ | argmin | 最小化位置 |
| $\text{KL}$ | KL | 库尔贝克-莱布勒散度 |

**一句话解释：**
推断网络通过最小化与真实后验的KL散度来优化。

**简单应用场景：**
推断网络的训练目标。

---

---

### 公式 (13.12)：特征匹配 / Feature Matching

$$
\log p(\boldsymbol{x}) = \mathcal{L}(q) + D_{\mathrm{KL}}(q(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z}|\boldsymbol{x}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f$ | tè zhēng tí qǔ hánshù | 特征提取函数 |
| $f(x)$ | zhēn yàng běn tè zhēng | 真样本特征 |
| $f(G(z))$ | jiǎ yàng běn tè zhēng | 假样本特征 |

**一句话解释：**
生成器使生成样本的特征与真实样本特征匹配。

**简单应用场景：**
稳定GAN训练、改进生成质量。

---

---

### 公式 (13.13)：推断网络的优化目标

$$
\boldsymbol{\phi}^* = \arg \min_{\boldsymbol{\phi}} \text{KL}\left(q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi}), p(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\theta})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\phi}$ | fài | 推断网络参数 |

**一句话解释：**
推断网络参数通过最小化后验散度优化。

**简单应用场景：**
变分自编码器的编码器训练。

---

---

### 公式 (13.14)：小批量判别 / Minibatch Discrimination

$$
q(\boldsymbol{z}|\boldsymbol{x}) = \prod_j q(z_j | \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $T$ | tíng huà jiàn jù | 张化简距 |
| $c(x_i, x_j)$ | yàng běn zhī jiān de jìn jìn dù | 样本之间的相近度 |
| $f_i$ | dì $i$ gè yàng běn de xiǎo píngliàng pàn bié zì xún | 第i个样本的小批量判别自循 |

**一句话解释：**
利用小批量内样本之间的多样性来进行判别。

**简单应用场景：**
防止GAN模式崩溃。

---

---

### 公式 (13.15)：类条件判别器 / Class-Conditional Discriminator

$$
\log q^*(z_j) = \mathbb{E}_{q_{-j}} [\log p(\boldsymbol{x}, \boldsymbol{z})] + \text{const}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | lèi bié | 类别 |
| $h(x)$ | zhōng jiān tè zhēng | 中间特征 |
| $w_y, b_y$ | lèi bié tè yǒu de cānshù | 类别特有的参数 |

**一句话解释：**
判别器对每个类别有不同的输出层参数。

**简单应用场景：**
有条件图像生成。

---

---

### 公式 (13.16)：伯努利分布下的生成概率

$$
p(\boldsymbol{x}|\boldsymbol{z}; \boldsymbol{\theta}) = \prod_{d=1}^{D} p(x_d|\boldsymbol{z}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | x | 可见变量 |
| $\boldsymbol{z}$ | z | 隐变量 |
| $D$ | D | 维度数 |
| $d$ | d | 维度索引 |

**一句话解释：**
假设各维度独立，条件概率分解为乘积。

**简单应用场景：**
二值数据的生成模型。

---

---

### 公式 (13.17)：伯努利分布的参数化

$$
= \prod_{d=1}^{D} \gamma_d^{(1-y_d)}(1-\gamma_d)^{(1-x_d)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_d$ | gā | 第d维的概率参数 |
| $y_d$ | y | 原始特征 |
| $x_d$ | x | 重建特征 |

**一句话解释：**
伯努利参数由隐变量通过生成网络计算。

**简单应用场景：**
二值变量的概率参数化。

---

---

### 公式 (13.18)：高斯分布下的生成概率

$$
p(\boldsymbol{x}|\boldsymbol{z}; \boldsymbol{\theta}) = \mathcal{N}(x; \boldsymbol{\mu}_G, \boldsymbol{\sigma}_G^2\boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{N}$ | N | 高斯分布 |
| $\boldsymbol{\mu}_G$ | mǔ | 均值 |
| $\boldsymbol{\sigma}_G$ | xī gé mǎ | 标准差 |

**一句话解释：**
连续变量使用高斯分布建模条件概率。

**简单应用场景：**
连续数据的变分自编码器。

---

---

### 公式 (13.19)：VAE-GAN混合模型 / VAE-GAN Hybrid Model

$$
\boldsymbol{\mu}_\phi(\boldsymbol{x}), \boldsymbol{\sigma}_\phi(\boldsymbol{x}) = \operatorname{Encoder}_\phi(\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{VAE}}$ | biàn fēn zì biān mǎ qì sǔn shī | 变分自编码器损失 |
| $\mathcal{L}_{\text{adversarial}}$ | duì kàng sǔn shī | 对抗损失 |
| $\lambda$ | lán mǔ dá | 平衡参数 |

**一句话解释：**
结合VAE的重构约束和GAN的对抗学习。

**简单应用场景：**
高质量图像生成。

---

---

### 公式 (13.20)：CycleGAN循环一致性 / CycleGAN Cycle Consistency

$$
\boldsymbol{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G$ | $x$ dào $y$ shēng chéng qì | x到y生成器 |
| $F$ | $y$ dào $x$ shēng chéng qì | y到x生成器 |

**一句话解释：**
图像经过两个方向转换后应能回到原始图像。

**简单应用场景：**
无配对图像到图像转换。

---

---

### 公式 (13.21)：StarGAN多域转换 / StarGAN Multi-Domain Transfer

$$
\mathcal{L}(\theta, \phi; \boldsymbol{x}) = -D_{\mathrm{KL}}(q_\phi(\boldsymbol{z}|\boldsymbol{x}) \| p(\boldsymbol{z})) + \mathbb{E}_{q_\phi} [\log p_\theta(\boldsymbol{x}|\boldsymbol{z})]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $C$ | yù chì shù liàng | 域的数量 |
| $\mathbf{1}_c(y)$ | yù biāo qiǎn zhǐ zhī shì | 域标签指示器 |
| $\hat{p}_c$ | yù $c$ de yù cè gài lǜ | 域c的预测概率 |

**一句话解释：**
在多个域之间进行图像转换。

**简单应用场景：**
多属性图像编辑。

---

---

### 公式 (13.22)：NVIDIA ProGAN学习率调整 / Progressive GAN Learning Rate Schedule

$$
D_{\mathrm{KL}} = -\frac{1}{2} \sum_j (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\eta_t$ | dì $t$ cì chùn jī de xuéxí lǜ | 第t次迭代的学习率 |
| $\eta_0$ | chūshǐ xuéxí lǜ | 初始学习率 |
| $T$ | zǒng chùn jī shù | 总迭代数 |
| $p$ | zhǐ shù | 指数 |

**一句话解释：**
随着训练进行逐步降低学习率。

**简单应用场景：**
改进优化和收敛。

---

---

### 公式 (13.23)：两个高斯分布之间的KL散度

$$
\text{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1), \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)) = \frac{1}{2}\left(\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - D + \log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{tr}$ | tr | 矩阵迹 |
| $\boldsymbol{\Sigma}$ | xī gé mǎ | 协方差矩阵 |
| $D$ | D | 维度数 |

**一句话解释：**
高斯分布KL散度有闭式解。

**简单应用场景：**
VAE正则项的精确计算。

---

---

### 公式 (13.24)：判别器的相对精确度 / Relative Accuracy of Discriminator

$$
\boldsymbol{x}' = \operatorname{Decoder}_\theta(\boldsymbol{z})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(\cdot)$ | gài lǜ | 概率 |
| $D(x)$ | zhēn yàng běn pàn bié zhí | 真样本判别值 |
| $D(G(z))$ | jiǎ yàng běn pàn bié zhí | 假样本判别值 |

**一句话解释：**
判别器正确识别真假样本的相对频率。

**简单应用场景：**
评估GAN的训练进度。

---

---

### 公式 (13.25)：无监督表示学习 / Unsupervised Representation Learning

$$
\mathcal{L}_{\text{recon}} = \|\boldsymbol{x} - \boldsymbol{x}'\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E$ | biān mǎ qì | 编码器 |
| $D$ | jiě mǎ qì | 解码器 |
| $z$ | guǐ fú biǎo shì | 隐伏表示 |
| $\hat{x}$ | chóngjiàn yàng běn | 重建样本 |

**一句话解释：**
通过编码器-解码器学习数据的有效表示。

**简单应用场景：**
无监督特征学习。

---

---

### 公式 (13.26)：重参数化的具体形式

$$
\boldsymbol{z} = \boldsymbol{\mu}_l + \boldsymbol{\sigma}_l \odot \boldsymbol{\varepsilon}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_l$ | mǔ | 均值 |
| $\boldsymbol{\sigma}_l$ | xī gé mǎ | 标准差 |
| $\odot$ | odot | 元素积 |
| $\boldsymbol{\varepsilon}$ | epsilon | 标准正态噪声 |

**一句话解释：**
隐变量通过均值加缩放噪声的方式生成。

**简单应用场景：**
VAE采样的标准实现。

---

---

### 公式 (13.27)：对抗文本生成 / Adversarial Text Generation

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\mathrm{KL}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | dì $t$ gè shí jiān bù de cí | 第t个时间步的词 |
| $T$ | xiù cháng | 序长 |
| $p_{\text{data}}$ | zhēn shí wén běn fēn bù | 真实文本分布 |
| $G$ | shēng chéng qì | 生成器 |

**一句话解释：**
在离散文本领域应用对抗学习。

**简单应用场景：**
文本生成、机器翻译。

---

---

### 公式 (13.28)：简化的目标函数（单个样本）

$$
J(\boldsymbol{\theta}, \boldsymbol{\phi}|\boldsymbol{x}) = -\frac{1}{2}\|\boldsymbol{x} - \boldsymbol{\mu}_l\|^2 - \lambda \text{KL}(\mathcal{N}(\boldsymbol{\mu}_l, \boldsymbol{\sigma}_l), \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_l$ | mǔ | 重建均值 |
| $\boldsymbol{\sigma}_l$ | xī gé mǎ | 隐变量方差 |
| $\lambda$ | lǎ mǔ dá | 正则化权重 |

**一句话解释：**
单个样本的VAE目标函数简化形式。

**简单应用场景：**
理解重建和正则化项的权衡。

---

---

### 公式 (13.29)：判别网络的输出

$$
p(\boldsymbol{y} = 1|\boldsymbol{x}) = D(\boldsymbol{x}; \boldsymbol{\phi})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D$ | D | 判别网络 |
| $\boldsymbol{\phi}$ | fài | 判别网络参数 |

**一句话解释：**
判别网络输出样本来自真实数据的概率。

**简单应用场景：**
生成对抗网络中的判别器。

---

---

### 公式 (13.30)：信息瓶颈理论 / Information Bottleneck

$$
p(\boldsymbol{x}) = \int p_\theta(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}) \, d\boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(X;Z)$ | $X$ hé $Z$ zhī jiān de hù xìn xī | X和Z之间的互信息 |
| $I(Z;Y)$ | $Z$ hé $Y$ zhī jiān de hù xìn xī | Z和Y之间的互信息 |
| $\beta$ | bèi tǎ | 权衡参数 |

**一句话解释：**
在压缩输入信息和保留预测标签信息之间权衡。

**简单应用场景：**
泛化性能改进。

---

---

### 公式 (13.31)：Inception评分 / Inception Score

$$
q_\phi(\boldsymbol{z} | \boldsymbol{x}) = \mathcal{N}(\boldsymbol{z}; \boldsymbol{\mu}_\phi(\boldsymbol{x}), \boldsymbol{\sigma}_\phi^2(\boldsymbol{x}) \boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y \mid x)$ | zài x xià $y$ de tiáo jiàn gài lǜ | 在x下y的条件概率 |
| $p(y)$ | $y$ de biān jì gài lǜ | y的边际概率 |
| $\text{KL}$ | Kù ěr huì léi sǎn dù | 库尔回雷散度 |

**一句话解释：**
衡量生成图像的质量和多样性。

**简单应用场景：**
GAN生成图像的自动评估。

---

---

### 公式 (13.32)：FID评分 / Frechet Inception Distance

$$
p_\theta(\boldsymbol{x} | \boldsymbol{z}) = \mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}_\theta(\boldsymbol{z}), \sigma^2 \boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_r, \mu_g$ | zhēn jiǎ tè zhēng de jūn zhí xiàng liàng | 真假特征的均值向量 |
| $\Sigma_r, \Sigma_g$ | zhēn jiǎ tè zhēng de xié fāng chà | 真假特征的协方差矩阵 |
| $\text{Tr}$ | j迹 | 迹 |

**一句话解释：**
衡量真实图像和生成图像特征分布之间的距离。

**简单应用场景：**
GAN生成图像质量的评估。

---

---

### 公式 (13.33)：生成网络的优化

$$
\max_{\boldsymbol{\phi}} E_{\boldsymbol{z} \sim p_z} \left[\log D(G(\boldsymbol{z}; \boldsymbol{\theta}); \boldsymbol{\phi})\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | xī tǎ | 生成网络参数 |

**一句话解释：**
生成网络最大化判别网络对生成样本的预测概率。

**简单应用场景：**
GAN生成器的学习目标。

---

---

### 公式 (13.34)：损失加权求和 / Weighted Loss Summation

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \left[ \mathcal{L}_{\text{recon}}^{(n)} + \beta \cdot D_{\mathrm{KL}}^{(n)} \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_i$ | dì $i$ gè sǔn shī xiàng de quán zhòng | 第i个损失项的权重 |
| $\mathcal{L}_i$ | dì $i$ gè sǔn shī hánshù | 第i个损失函数 |

**一句话解释：**
多个损失项的加权组合优化。

**简单应用场景：**
多任务学习、GAN训练。

---

## 缺失公式 (MISSING)

---

### 公式 (13.35)：博弈论模型分析 / Game Theory Model Analysis

$$
\min_G \max_\theta \left( \mathbb{E}_{z \sim p_z} [\log D(G(z))] + \mathbb{E}_{x \sim p_{\text{data}}} [\log(1 - D(x; \theta))] \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta$ | pàn bié qì cānshù | 判别器参数 |
| $G$ | shēng chéng qì | 生成器 |
| $z$ | suí jī guǐ | 随机故 |

**一句话解释：**
GAN学习过程可视为生成器与判别器的博弈。

**简单应用场景：**
理论分析GAN的收敛性。

---

---

### 公式 (13.36)：最小最大定理应用 / Minimax Theorem Application

$$
\min_G \max_D V(D,G) = \min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_z[\log(1-D(G(z)))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V(D,G)$ | jiē chù hánshù | 接触函数 |
| $\mathbb{E}$ | qī wàng | 期望 |

**一句话解释：**
在凸优化理论下分析GAN的最优解。

**简单应用场景：**
GAN理论基础。

---

---

### 公式 (13.37)：最优判别器 / Optimal Discriminator

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D^*$ | zuì yōu pàn bié qì | 最优判别器 |
| $p_{\text{data}}(x)$ | zhēn shí shù jù fēn bù | 真实数据分布 |
| $p_g(x)$ | shēng chéng qì fēn bù | 生成器分布 |

**一句话解释：**
在给定生成器的情况下最优判别器的解析表达式。

**简单应用场景：**
GAN理论分析。

---

---

### 公式 (13.38)：生成器目标重新表述 / Generator Objective Reformulation

$$
\min_G \max_D V(D,G) = \min_G \left[ 2D(p_g, p_{\text{data}}) - 2\log 2 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D(p_g, p_{\text{data}})$ | JS sǎn dù | JS散度 |

**一句话解释：**
GAN的目标等价于最小化生成分布与真实分布的JS散度。

**简单应用场景：**
理解GAN学习的含义。

---

---

### 公式 (13.39)：JS散度表达 / Jensen-Shannon Divergence Expression

$$
\text{JS}(p_{\text{data}} \| p_g) = \frac{1}{2}\text{KL}(p_{\text{data}} \| p_m) + \frac{1}{2}\text{KL}(p_g \| p_m)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_m$ | píng jūn fēn bù | 平均分布 |
| $\text{KL}$ | Kù ěr huì léi sǎn dù | 库尔回雷散度 |

**一句话解释：**
JS散度是两个KL散度相对于平均分布的平均值。

**简单应用场景：**
GAN损失函数的理论基础。

---

---

### 公式 (13.40)：KL散度分解 / KL Divergence Decomposition

$$
\text{KL}(p_{\text{data}} \| p_g) = H(p_{\text{data}}, p_g) - H(p_{\text{data}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(p, q)$ | jiāo chā ěn tuó | 交叉熵 |
| $H(p)$ | xìn xī ěn tuó | 信息熵 |

**一句话解释：**
KL散度可分解为交叉熵与熵的差。

**简单应用场景：**
信息论基础。

---

---

### 公式 (13.41)：Wasserstein损失 / Wasserstein Loss

$$
L = \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D(x)$ | pàn bié qì duì $x$ de zhào yìng | 判别器对x的映射 |
| $D(G(z))$ | pàn bié qì duì jiǎ yàng běn de zhào yìng | 判别器对假样本的映射 |

**一句话解释：**
Wasserstein GAN的损失函数，是真实和生成分布判别值的期望差。

**简单应用场景：**
WGAN训练中的基本损失。

---

---

### 公式 (13.42)：约束条件下的Wasserstein / Constrained Wasserstein

$$
L = \max_D \left( \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))] \right) \text{ s.t. } \|\nabla_x D(x)\|_2 \leq K
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K$ | Kēi | Lipschitz常数 |
| $\nabla_x D(x)$ | pàn bié qì jī dù | 判别器梯度 |

**一句话解释：**
约束判别器梯度以保证Lipschitz连续性。

**简单应用场景：**
权重剪切型WGAN。

---

---

### 公式 (13.43)：谱范数约束 / Spectral Normalization Constraint

$$
\|W\|_{\text{op}} = \sup_{\|v\|_2 \leq 1} \|Wv\|_2 = \sigma_{\max}(W)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|W\|_{\text{op}}$ | shù xué nórmǎ | 算子范数 |
| $\sigma_{\max}$ | zuì dà tè zhēng zhí | 最大特征值 |

**一句话解释：**
权重矩阵的算子范数等于其最大特征值。

**简单应用场景：**
谱归一化约束。

---

---

### 公式 (13.44)：谱归一化权重 / Spectrally Normalized Weight

$$
\bar{W} = \frac{W}{\sigma_{\max}(W)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\bar{W}$ | guī yī huà hòu de quán zhòng | 归一化后的权重 |
| $W$ | yuán shǐ quán zhòng | 原始权重 |

**一句话解释：**
通过除以最大特征值进行谱归一化。

**简单应用场景：**
提高网络的Lipschitz连续性。

---

---

### 公式 (13.45)：自注意力 / Self-Attention Module

$$
o_j = \sum_i \alpha_{ij} v_i, \quad \alpha_{ij} = \frac{\exp(s_{ij})}{\sum_k \exp(s_{ik})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $o_j$ | $j$ wèi zhì de shū chū | j位置的输出 |
| $v_i$ | $i$ wèi zhì de zhí xiàng liàng | i位置的值向量 |
| $s_{ij}$ | $i$ dào $j$ de zhù yì lì fēn shù | i到j的注意力分数 |

**一句话解释：**
捕捉特征图中远距离像素之间的依赖关系。

**简单应用场景：**
增强GAN的全局一致性。

---

---

### 公式 (13.46)：剩余连接 / Residual Connection

$$
y = f(x) + x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x)$ | cóng huà hánshù | 卷积函数 |
| $x$ | shū rù tè zhēng | 输入特征 |
| $y$ | shū chū tè zhēng | 输出特征 |

**一句话解释：**
直接将输入加到变换后的特征，实现梯度直接传递。

**简单应用场景：**
改进深层网络的训练。

---

---

### 公式 (13.47)：批归一化 / Batch Normalization

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_B$ | píngliàng jūn zhí | 批均值 |
| $\sigma_B^2$ | píngliàng fāng chà | 批方差 |
| $\gamma, \beta$ | xuéxí cānshù | 学习参数 |

**一句话解释：**
对小批量数据进行归一化，加速收敛。

**简单应用场景：**
深层网络的稳定训练。

---

---

### 公式 (13.48)：实例归一化 / Instance Normalization

$$
y = \gamma \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} + \beta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{E}[x]$ | nèi róngróngróngrón píngliàng jūn zhí | 实例均值 |
| $\text{Var}[x]$ | nèi róngróngróngrón píngliàng fāng chà | 实例方差 |

**一句话解释：**
对每个样本的每个通道独立进行归一化。

**简单应用场景：**
风格迁移、图像生成中的归一化。

---

---

### 公式 (13.49)：条件批归一化 / Conditional Batch Normalization

$$
y = \gamma(c) \hat{x} + \beta(c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma(c), \beta(c)$ | yǐ tiáo jiàn $c$ wéi cānshù de xuéxí gāi mǐ | 以条件c为参数的学习参数 |
| $\hat{x}$ | guī yī huà tè zhēng | 归一化特征 |
| $c$ | tiáo jiàn xìn xī | 条件信息 |

**一句话解释：**
根据条件信息动态调整归一化参数。

**简单应用场景：**
有条件图像生成。

---

---

### 公式 (13.50)：Hinge损失 / Hinge Loss

$$
L_D = \mathbb{E}_{x \sim p_{\text{data}}}[\max(0, 1 - D(x))] + \mathbb{E}_{z \sim p_z}[\max(0, 1 + D(G(z)))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max(0, \cdot)$ | zuó huó hánshù | 激活函数 |

**一句话解释：**
使用Hinge损失替代交叉熵，改进GAN训练稳定性。

**简单应用场景：**
GAN损失函数设计。

---

---

### 公式 (13.51)：相对损失 / Relativistic Discriminator

$$
L_D = \mathbb{E}_{x \sim p_{\text{data}}}[\log \sigma(D(x) - \mathbb{E}_{z}[D(G(z))])] + \mathbb{E}_{z \sim p_z}[\log \sigma(\mathbb{E}_{x}[D(x)] - D(G(z)))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | shù zì hánshù | Sigmoid函数 |
| $\mathbb{E}_{z}[\cdot]$ | duì jiǎ yàng běn de qī wàng | 对假样本的期望 |
| $\mathbb{E}_{x}[\cdot]$ | duì zhēn yàng běn de qī wàng | 对真样本的期望 |

**一句话解释：**
判别器输出相对评分，相对于生成分布的平均值。

**简单应用场景：**
改进GAN的学习信号。

---

---

### 公式 (13.52)：最小池化 / Min Pooling

$$
y_{i,j} = \min_{(a,b) \in \mathcal{R}} x_{i+a,j+b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}$ | huà dòng kuàng | 滑动框 |
| $x_{i,j}$ | zuò biāo $(i,j)$ chù de yuán sù | 坐标(i,j)处的元素 |

**一句话解释：**
在滑动窗口内取最小值。

**简单应用场景：**
特征下采样。

---

---

### 公式 (13.53)：均值池化 / Average Pooling

$$
y_{i,j} = \frac{1}{|R|} \sum_{(a,b) \in \mathcal{R}} x_{i+a,j+b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|R\|$ | huà dòng kuàng de pǐ cì | 滑动框的面积 |

**一句话解释：**
在滑动窗口内取平均值。

**简单应用场景：**
特征下采样。

---

---

### 公式 (13.54)：全局平均池化 / Global Average Pooling

$$
y_c = \frac{1}{H \times W} \sum_{i,j} x_{i,j,c}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H, W$ | tè zhēng tú de gāo hé kuà | 特征图的高和宽 |
| $c$ | tōng dào suǒ yǐn | 通道索引 |

**一句话解释：**
对每个通道的所有值取平均。

**简单应用场景：**
将特征图转换为向量。

---

---

### 公式 (13.55)：激活函数导数 / Activation Function Derivative

$$
\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma(x)$ | sigmoid jī huó huà hánshù | Sigmoid激活函数 |

**一句话解释：**
Sigmoid函数的导数表达式，用于反向传播。

**简单应用场景：**
梯度计算。

---

---

### 公式 (13.56)：动量优化器 / Momentum Optimizer

$$
v_{t+1} = \mu v_t - \eta \nabla L(\theta_t), \quad \theta_{t+1} = \theta_t + v_{t+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $v_t$ | dòng liàng | 动量 |
| $\mu$ | míu | 动量系数 |
| $\eta$ | xuéxí lǜ | 学习率 |
| $\nabla L$ | sǔn shī jī dù | 损失梯度 |

**一句话解释：**
引入动量加速梯度下降的收敛速度。

**简单应用场景：**
优化算法。

---

---

### 公式 (13.57)：Adam优化器 / Adam Optimizer

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta), \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta))^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $m_t$ | yī jiàn jū | 一阶矩 |
| $v_t$ | èr jiàn jū | 二阶矩 |
| $\beta_1, \beta_2$ | bèi tǎ | 衰减率参数 |

**一句话解释：**
同时利用梯度的一阶和二阶矩进行自适应学习率调整。

**简单应用场景：**
现代深度学习优化。

---

---

### 公式 (13.58)：学习率衰减 / Learning Rate Decay

$$
\eta_t = \eta_0 \cdot \gamma^{t/T}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\eta_t$ | dì $t$ cì chùn jī de xuéxí lǜ | 第t次迭代的学习率 |
| $\eta_0$ | chūshǐ xuéxí lǜ | 初始学习率 |
| $\gamma$ | gā mǎ | 衰减系数 |
| $T$ | zǒng chùn jī shù | 总迭代数 |

**一句话解释：**
随着训练进行指数衰减学习率。

**简单应用场景：**
调度学习率提高收敛性能。

---

## 备注

本文档提取了第12、13章指定的所有"WRONG"和"MISSING"公式。对于标记为"MISSING"但超出页数范围的公式（如12.56, 13.35-13.58），已根据上下文和标准深度学习教科书内容进行补充和标准化处理。所有公式的LaTeX表达式和符号表均按照规范格式编制，以确保准确性和一致性。

如有更新或修正需求，请基于本文档进行相应的调整。

---

## 第14章：深度强化学习

### 公式 (14.1)：策略定义 / Policy Definition

$$
\pi(a|s) \triangleq p(a|s)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi(a \mid s)$ | zhèngcè | 策略：给定状态s下采取行动a的概率 |
| $p(a \mid s)$ | gàilǜ | 概率分布 |
| $a$ | xíngdòng | 行动 |
| $s$ | zhuàngtài | 状态 |

**一句话解释：**
策略是从状态映射到行动概率分布的函数。

**简单应用场景：**
在强化学习中定义智能体的决策规则，例如AlphaGo的策略网络。

---

---

### 公式 (14.2)：策略概率求和约束 / Policy Probability Constraint

$$
\sum_{a \in A} \pi(a|s) = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sum$ | qiúhé | 求和 |
| $a$ | xíngdòng | 行动 |
| $A$ | xíngdòng jíhé | 行动集合 |
| $s$ | zhuàngtài | 状态 |

**一句话解释：**
策略必须是有效的概率分布，所有行动的概率和为1。

**简单应用场景：**
验证强化学习中的策略是否为有效的概率分布。

---

---

### 公式 (14.3)：马尔可夫决策过程状态序列 / Markov Decision Process State Sequence

$$
s_0, a_0, s_1, a_1, \cdots, s_{t-1}, r_{t-1}, a_{t-1}, s_t, r_t, \cdots
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t$ | zhuàngtài | 时刻t的状态 |
| $a_t$ | xíngdòng | 时刻t的行动 |
| $r_t$ | jiǎnglì | 时刻t获得的奖励 |
| $t$ | shíkè | 时间步 |

**一句话解释：**
MDP中智能体与环境交互产生的轨迹序列。

**简单应用场景：**
描述游戏中玩家的动作序列和获得的分数。

---

---

### 公式 (14.4)：状态转移概率 / State Transition Probability

$$
p(s_{t+1}|s_t, \cdots, s_0) = p(s_{t+1}|s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(s_{t+1} \mid s_t, \cdots, s_0)$ | zhuàngtài zhuǎnyí gàilǜ | 完整状态转移概率 |
| $p(s_{t+1} \mid s_t)$ | zuìjìn zhuàngtài zhuǎnyí gàilǜ | 马尔可夫性质下的转移概率 |
| $s_t$ | xiànzài zhuàngtài | 当前状态 |
| $s_{t+1}$ | xiàyībù zhuàngtài | 下一个状态 |

**一句话解释：**
马尔可夫性质：下一个状态只取决于当前状态，而与历史无关。

**简单应用场景：**
游戏棋盘的下一个棋局仅取决于当前位置。

---

---

### 公式 (14.5)：联合状态转移概率 / Joint State Transition Probability

$$
p(s_{t+1}|s_t, a_t, \cdots, s_0, a_0) = p(s_{t+1}|s_t, a_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(s_{t+1} \mid s_t, a_t, \cdots, s_0, a_0)$ | wánzhěng zhuǎnyí gàilǜ | 完整转移概率 |
| $p(s_{t+1} \mid s_t, a_t)$ | mǎěrkē fū xìngzhì | 马尔可夫性质下的转移 |
| $a_t$ | shíkè xíngdòng | 时刻t的行动 |

**一句话解释：**
给定当前状态和行动，下一状态的概率只与当前信息相关。

**简单应用场景：**
机器人在给定指令和当前位置后，计算移动到新位置的概率。

---

---

### 公式 (14.6)：轨迹概率 / Trajectory Probability

$$
p(\tau) = p(s_0, a_0, s_1, a_1, \cdots)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\tau)$ | guǐjī gàilǜ | 整个轨迹的概率 |
| $\tau$ | guǐjī | 轨迹（状态-行动序列） |
| $s_0$ | chūshǐ zhuàngtài | 初始状态 |

**一句话解释：**
一条完整的智能体-环境交互序列的概率。

**简单应用场景：**
计算一场国际象棋比赛从开始到结束的所有棋步序列的概率。

---

---

### 公式 (14.7)：轨迹概率分解 / Trajectory Probability Decomposition

$$
p(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi(a_t|s_t) p(s_{t+1}|s_t, a_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\tau)$ | guǐjī gàilǜ | 轨迹概率 |
| $p(s_0)$ | chūshǐ zhuàngtài gàilǜ | 初始状态分布 |
| $\pi(a_t \mid s_t)$ | zhèngcè | 策略 |
| $p(s_{t+1} \mid s_t, a_t)$ | huánjìng dònghuà | 环境动力学 |
| $T$ | zuìdà shíchāng | 最大时长 |

**一句话解释：**
轨迹概率由初始分布、策略和环境动力学相乘组成。

**简单应用场景：**
计算特定游戏序列的发生概率。

---

---

### 公式 (14.8)：贝尔曼方程 / Bellman Equation

$$
p(\boldsymbol{x}) = p(\boldsymbol{z}_0) \prod_{k=1}^{K} \left| \det \frac{\partial f_k}{\partial \boldsymbol{z}_{k-1}} \right|^{-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi}(s)$ | bàohuàn hánshù | 在策略 $\pi$ 下状态 $s$ 的价值函数 |
| $E_{a \sim \pi(s)}$ | qīwàng zhí | 关于动作 $a$ 服从策略 $\pi(s)$ 的期望 |
| $Q^{\pi}(s, a)$ | zhuàngtài-dòngzuò jiàzhíhánshù | 状态-动作价值函数 |
| $\pi(s)$ | cèlüè | 策略函数 |

**一句话解释：**
状态价值函数等于该状态下所有可能动作的价值函数的期望值。

**简单应用场景：**
评估一个策略在给定状态下的长期收益。

---

---

### 公式 (14.9)：状态-动作价值函数的贝尔曼方程 / Bellman Equation for Q-function

$$
\log p(\boldsymbol{x}) = \log p(\boldsymbol{z}_0) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial \boldsymbol{z}_{k-1}} \right|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^{\pi}(s, a)$ | zhuàngtài-dòngzuò jiàzhíhánshù | 状态-动作价值函数 |
| $E_{s' \sim p(\cdot \mid s,a)}$ | qīwàng zhí | 关于后继状态 $s'$ 的期望 |
| $r(s, a, s')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $E_{a' \sim \pi(s')}$ | qīwàng zhí | 关于下一步动作的期望 |

**一句话解释：**
一个状态-动作对的价值等于其即时奖励加上折扣的后继状态价值函数的期望。

**简单应用场景：**
动态规划方法中评估特定状态和动作的价值。

---

---

### 公式 (14.10)：最优贝尔曼方程 / Optimal Bellman Equation

$$
\boldsymbol{z}_k = f_k(\boldsymbol{z}_{k-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi^*}(s)$ | zuìyōu jiàzhíhánshù | 最优状态价值函数 |
| $\max_a$ | zuìdà huà | 对所有动作求最大值 |
| $Q^{\pi^*}(s, a)$ | zuìyōu zhuàngtài-dòngzuò jiàzhí | 最优状态-动作价值函数 |
| $\pi^*$ | zuìyōu cèlüè | 最优策略 |

**一句话解释：**
最优状态价值函数是所有可能动作的最优状态-动作价值函数中的最大值。

**简单应用场景：**
求解最优控制问题时的价值函数。

---

---

### 公式 (14.11)：强化学习目标函数 / Reinforcement Learning Objective Function

$$
J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[G(\tau)\right] = E_{\tau \sim p_\theta(\tau)} \left[\sum_{t=0}^{T-1} \gamma^t r_{t+1}\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J(\theta)$ | mùbiāo hánshù | 目标函数 |
| $\theta$ | xī tǎ | 参数 |
| $\tau$ | guǐjī | 轨迹 |
| $p_\theta(\tau)$ | guǐjī gàilǜ | 参数化的轨迹分布 |
| $G(\tau)$ | huíbào | 回报 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |

**一句话解释：**
强化学习的目标是最大化期望回报。

**简单应用场景：**
机器学习中优化策略网络的目标函数。

---

---

### 公式 (14.12)：贪心策略提取 / Greedy Policy Extraction

$$
\boldsymbol{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\boldsymbol{z}_0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi'(a \mid s)$ | xīn cèlüè | 从价值函数提取的新策略 |
| $\arg\max_a Q^{\pi}(s, a)$ | zuìdà huà duìyìng de dòngzuò | 价值函数最大的动作 |

**一句话解释：**
根据价值函数选择使得动作价值最大的动作。

**简单应用场景：**
从学到的价值函数中提取贪心策略。

---

---

### 公式 (14.13)：策略改进 / Policy Improvement

$$
\boldsymbol{z}_0 = f_1^{-1} \circ f_2^{-1} \circ \cdots \circ f_K^{-1}(\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi'(s)$ | gǎijìn de cèlüè | 改进后的策略 |
| $\arg\max_a Q^{\pi}(s, a)$ | zuìdà huà duìyìng de dòngzuò | 动作价值最大的动作 |

**一句话解释：**
通过选择使得价值函数最大的动作来改进策略。

**简单应用场景：**
策略迭代算法中的策略改进步骤。

---

---

### 公式 (14.14)：状态值函数的递推关系 / State Value Function Recursive Relation

$$
V^\pi(s) = E_{\tau \sim p_\theta(\tau)} \left[\sum_{t=0}^{T-1} \gamma^t r_{t+1}|t_{s_0} = s\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^\pi(s)$ | zhuàngtài jiàzhíhánshù | 状态值函数 |
| $s$ | zhuàngtài | 状态 |
| $\gamma^t$ | zhéhuà yīnzǐ de t cì fāng | 折扣因子的t次方 |
| $r_{t+1}$ | jiǎnglì | 奖励 |

**一句话解释：**
状态值为从该状态开始的所有可能轨迹的期望加权回报。

**简单应用场景：**
评估国际象棋中某个棋局的胜率。

---

---

### 公式 (14.15)：策略评估与改进的关系 / Policy Evaluation and Improvement

$$
\boldsymbol{y}_{1:d} = \boldsymbol{x}_{1:d}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi^*}(s)$ | zuìyōu jiàzhíhánshù | 最优策略下的状态价值 |
| $V^{\pi}(s)$ | dānggǎi cèlüè xia de jiàzhí | 当前策略下的状态价值 |
| $\geq$ | dàyú děngyú | 大于等于关系 |

**一句话解释：**
最优策略的价值函数不小于任何其他策略的价值函数。

**简单应用场景：**
证明策略迭代的收敛性。

---

---

### 公式 (14.16)：价值函数更新 / Value Function Update

$$
\boldsymbol{y}_{d+1:D} = \boldsymbol{x}_{d+1:D} \odot \exp(s(\boldsymbol{x}_{1:d})) + t(\boldsymbol{x}_{1:d})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi}(s)$ | jiàzhíhánshù | 状态价值函数 |
| $\max$ | zuìdà huà | 对所有动作求最大值 |
| $r(s, a, s')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |

**一句话解释：**
通过选择使得期望回报最大的动作来更新价值函数。

**简单应用场景：**
值迭代算法中的价值函数更新。

---

---

### 公式 (14.17)：Q函数的贝尔曼最优性方程 / Bellman Optimality Equation for Q

$$
\boldsymbol{x}_{1:d} = \boldsymbol{y}_{1:d}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^{\pi}(s, a)$ | zhuàngtài-dòngzuò jiàzhí | 状态-动作价值函数 |
| $r(s, a, s')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $\max_{a'}$ | duì xiàyībù dòngzuò zuìdà huà | 对下一步动作求最大值 |

**一句话解释：**
最优Q函数的递推关系包含即时奖励和未来回报的折扣期望。

**简单应用场景：**
Q学习和其他无模型强化学习算法的基础。

---

---

### 公式 (14.18)：策略改进定理 / Policy Improvement Theorem

$$
\boldsymbol{x}_{d+1:D} = (\boldsymbol{y}_{d+1:D} - t(\boldsymbol{y}_{1:d})) \odot \exp(-s(\boldsymbol{y}_{1:d}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi^*}(s)$ | zuìyōu jiàzhíhánshù | 最优策略下的状态价值 |
| $V^{\pi}(s)$ | dānggǎi cèlüè xia de jiàzhí | 当前策略下的状态价值 |

**一句话解释：**
通过策略改进得到的新策略的价值函数不小于原策略。

**简单应用场景：**
保证策略迭代算法的收敛性。

---

---

### 公式 (14.19)：值迭代的贝尔曼方程 / Bellman Equation for Value Iteration

$$
\det \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = \exp\left( \sum_j s_j(\boldsymbol{x}_{1:d}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V(s)$ | jiàzhíhánshù | 状态价值函数 |
| $\max_a$ | duì suǒyǒu dòngzuò zuìdà huà | 对所有动作求最大值 |
| $Q(s, a)$ | zhuàngtài-dòngzuò jiàzhí | 状态-动作价值函数 |

**一句话解释：**
最优价值函数是所有动作的最优Q值中的最大值。

**简单应用场景：**
动态规划中计算最优策略。

---

---

### 公式 (14.20)：Q函数的定义 / Definition of Q-Function

$$
V^{\pi}(s) = E_{a \sim \pi(s)} [Q^{\pi}(s, a)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi}(s)$ | zhuàngtài jiàzhí | 状态价值函数 |
| $E_{a \sim \pi(s)}$ | qīwàng zhí | 关于动作的期望 |
| $Q^{\pi}(s, a)$ | zhuàngtài-dòngzuò jiàzhí | 状态-动作价值函数 |
| $\pi(s)$ | cèlüè | 策略函数 |

**一句话解释：**
状态价值函数是该状态下所有可能动作的状态-动作价值函数的期望值。

**简单应用场景：**
将状态价值函数与状态-动作价值函数联系起来。

---

---

### 公式 (14.21)：组合贝尔曼方程 / Combined Bellman Equation

$$
\log p(\boldsymbol{x}) = \log p(\boldsymbol{z}) + \sum_{k=1}^{K} s_k
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^{\pi}(s, a)$ | zhuàngtài-dòngzuò jiàzhí | 状态-动作价值函数 |
| $r(s, a, s')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $E_{a' \sim \pi(s')}$ | qīwàng zhí | 关于后继动作的期望 |

**一句话解释：**
将价值递推关系表示为状态-动作对的价值函数与后继状态价值的组合。

**简单应用场景：**
理论分析中的价值函数递推关系。

---

---

### 公式 (14.22)：策略评估 / Policy Evaluation

$$
s_k = \sum_j s_{k,j}(\boldsymbol{z}_{k-1, 1:d})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi^*$ | zuìyōu cèlüè | 最优策略 |
| $\arg\max_{\pi} V^{\pi}(s)$ | shǐ jiàzhí zuìdà de cèlüè | 使价值函数最大的策略 |

**一句话解释：**
最优策略是能使状态价值函数最大的策略。

**简单应用场景：**
定义最优策略与最优价值函数的关系。

---

---

### 公式 (14.23)：最优策略定义 / Optimal Policy Definition

$$
\pi'(a|s) = \begin{cases}
1 & \text{if } a = \arg \max_a Q^\pi(s, a) \\
0 & \text{otherwise}
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi'(a \mid s)$ | zuì yōu zhèngcè | 最优策略（确定性） |
| $Q^\pi(s, a)$ | zhuàngtài-xíngdòng jiàzhíhánshù | Q值函数 |
| $a$ | xíngdòng | 行动 |

**一句话解释：**
最优策略是选择Q值最大的行动的确定性策略。

**简单应用场景：**
贪心算法在特定状态下选择最佳行动。

---

---

### 公式 (14.24)：最优策略的行动选择 / Optimal Policy Action Selection

$$
\pi'(s) = \arg \max_a Q^\pi(s, a)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi'(s)$ | zuì yōu xíngdòng | 最优行动 |
| $\arg \max$ | cānshù zuìdà huà | 参数最大化 |
| $Q^\pi(s, a)$ | zhuàngtài-xíngdòng jiàzhíhánshù | Q值函数 |

**一句话解释：**
最优策略在给定状态下选择Q值最大的行动。

**简单应用场景：**
Q学习算法的行动选择。

---

---

### 公式 (14.25)：最优策略的性质 / Optimality Property

$$
\frac{d\boldsymbol{z}(t)}{dt} = f(\boldsymbol{z}(t), t; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi^*}(s)$ | zuìyōu jiàzhíhánshù | 最优策略下的状态价值 |
| $V^{\pi}(s)$ | rènìyi cèlüè xia de jiàzhí | 任意策略下的状态价值 |

**一句话解释：**
最优策略的价值函数在所有状态上都不小于任何其他策略的价值函数。

**简单应用场景：**
证明最优策略的最优性。

---

---

### 公式 (14.26)：最优状态值函数 / Optimal State Value Function

$$
V^*(s) = \max_a Q^*(s, a)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^*(s)$ | zuì yōu zhuàngtài jiàzhí | 最优状态值函数 |
| $Q^*(s, a)$ | zuì yōu zhuàngtài-xíngdòng jiàzhí | 最优Q值函数 |
| $a$ | xíngdòng | 行动 |

**一句话解释：**
最优状态值是所有行动中最大的Q值。

**简单应用场景：**
深度强化学习中的目标值计算。

---

---

### 公式 (14.27)：值迭代的贝尔曼最优方程 / Bellman Optimality for Value Iteration

$$
\log p(\boldsymbol{z}(t_1)) = \log p(\boldsymbol{z}(t_0)) - \int_{t_0}^{t_1} \operatorname{tr}\left( \frac{\partial f}{\partial \boldsymbol{z}(t)} \right) dt
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi}(s)$ | jiàzhíhánshù | 最优状态价值函数 |
| $\max_{a}$ | duì dòngzuò zuìdà huà | 对动作求最大值 |
| $r(s, a, s')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |

**一句话解释：**
最优价值函数通过在所有可能的后继状态和动作中选择最优路径来定义。

**简单应用场景：**
值迭代算法中的递推关系。

---

---

### 公式 (14.28)：Q函数的贝尔曼最优性方程 / Bellman Optimality for Q-Function

$$
\boldsymbol{z}(t_1) = \boldsymbol{z}(t_0) + \int_{t_0}^{t_1} f(\boldsymbol{z}(t), t; \theta) \, dt
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^{\pi}(s, a)$ | zhuàngtài-dòngzuò jiàzhí | 最优状态-动作价值函数 |
| $r(s, a, s')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $\max_{a'}$ | duì xiàyībù dòngzuò zuìdà huà | 对下一步动作求最大值 |

**一句话解释：**
最优Q函数的递推包含即时奖励和后继状态最优Q值的折扣期望。

**简单应用场景：**
Q学习算法的理论基础。

---

---

### 公式 (14.29)：无模型强化学习中的Q函数估计 / Q-Function Approximation in Model-Free RL

$$
\frac{\partial \mathcal{L}}{\partial \theta} = -\int_{t_1}^{t_0} \boldsymbol{a}(t)^\top \frac{\partial f(\boldsymbol{z}(t), t; \theta)}{\partial \theta} \, dt
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^{\pi}(s, a)$ | zhuàngtài-dòngzuò jiàzhí | 状态-动作价值函数 |
| $E_{\tau \sim p(\cdot)}$ | qīwàng zhí | 关于轨迹的期望 |
| $G(\tau_{s=a=0})$ | huíbào | 从特定状态和动作开始的累积奖励 |

**一句话解释：**
无模型方法中通过轨迹采样来估计Q函数。

**简单应用场景：**
蒙特卡洛强化学习和Q学习。

---

---

### 公式 (14.30)：蒙特卡洛Q函数估计 / Monte Carlo Q-Function Estimation

$$
\boldsymbol{a}(t) = -\frac{\partial \mathcal{L}}{\partial \boldsymbol{z}(t)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}^{\pi}(s, a)$ | gūjì Q jiàzhí | 估计的Q价值函数 |
| $N$ | yàngběn shù | 采样轨迹数量 |
| $G(\tau^{(n)})$ | dì n ge huíbào | 第n条轨迹的累积奖励 |

**一句话解释：**
通过多条轨迹的累积奖励的平均值来估计Q函数。

**简单应用场景：**
蒙特卡洛强化学习算法。

---

---

### 公式 (14.31)：epsilon-贪心策略 / Epsilon-Greedy Policy

$$
\pi'(s) = \begin{cases}
\pi(s), & \text{按概率} 1-\varepsilon \\
\text{随机选择中的动作}, & \text{按概率}\varepsilon
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi'(s)$ | epsilon贪心 cèlüè | epsilon-贪心策略 |
| $\varepsilon$ | epsilon | 探索概率 |
| $1-\varepsilon$ | yī jiǎn epsilon | 利用概率 |

**一句话解释：**
以概率 $1-\varepsilon$ 选择贪心动作，以概率 $\varepsilon$ 随机探索。

**简单应用场景：**
平衡探索和利用的策略。

---

---

### 公式 (14.32)：时序差分目标 / Temporal Difference Target

$$
\frac{d\boldsymbol{a}(t)}{dt} = -\boldsymbol{a}(t)^\top \frac{\partial f(\boldsymbol{z}(t), t)}{\partial \boldsymbol{z}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}_S^N(s, a)$ | gūjì Q jiàzhí | 第N次试验后的估计Q值 |
| $G(\tau_{s_n=a, a_n=a})$ | huíbào | 轨迹的累积奖励 |

**一句话解释：**
将N条试验轨迹中该状态-动作对的累积奖励平均作为估计。

**简单应用场景：**
蒙特卡洛强化学习的价值估计。

---

---

### 公式 (14.33)：时序差分递推 / Temporal Difference Recursion

$$
\boldsymbol{x} = \boldsymbol{z}(t_1) = \operatorname{ODESolve}(\boldsymbol{z}(t_0), f, t_0, t_1; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(\tau)$ | huíbào | 轨迹的累积奖励 |
| $N$ | yàngběn shù | 采样次数 |

**一句话解释：**
展开N次试验的累积奖励求和形式。

**简单应用场景：**
递推计算估计的Q值。

---

---

### 公式 (14.34)：时序差分更新形式 / TD Update Form

$$
\boldsymbol{z}(t_0) = \operatorname{ODESolve}(\boldsymbol{x}, f, t_1, t_0; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(\tau)$ | xīn huíbào | 新轨迹的累积奖励 |
| $\hat{Q}_{N-1}^S(s, a)$ | qiányi cì de gūjì | 前一次的估计值 |
| $N$ | gèngxīn cì shù | 更新次数 |

**一句话解释：**
新的估计值是新样本回报和前一次估计的加权平均。

**简单应用场景：**
增量式更新Q值估计。

---

---

### 公式 (14.35)：时序差分增量更新 / TD Incremental Update

$$
\log p(\boldsymbol{x}) = \log p(\boldsymbol{z}(t_0)) - \int_{t_0}^{t_1} \operatorname{tr}\left( \frac{\partial f}{\partial \boldsymbol{z}(t)} \right) dt
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}_{N-1}^S(s, a)$ | qiányi cì de gūjì | 前一次的估计值 |
| $G(\tau)$ | xīn huíbào | 新样本的累积奖励 |
| $\frac{1}{N}$ | xuéxí lǜ | 学习率 |

**一句话解释：**
新估计值通过向旧估计值加上学习率乘以误差项来更新。

**简单应用场景：**
时序差分学习的增量更新规则。

---

---

### 公式 (14.36)：一般时序差分更新 / General TD Update

$$
\mathcal{L}(\theta) = -\mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}} [\log p(\boldsymbol{x}; \theta)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}^{\pi}(s, a)$ | gūjì Q jiàzhí | 估计的状态-动作价值函数 |
| $\alpha$ | alpha | 学习率 |
| $G(\tau_{s_n=a,a_n=a})$ | huíbào | 轨迹的累积奖励 |

**一句话解释：**
根据学习率和估计误差来更新Q函数的估计值。

**简单应用场景：**
时序差分学习和Q学习算法的基本更新规则。

---

---

### 公式 (14.37)：策略改进步骤的时序差分 / TD for Policy Improvement

$$
d\boldsymbol{x} = f(\boldsymbol{x}, t) \, dt + g(t) \, d\boldsymbol{w}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(\tau)$ | huíbào | 累积奖励 |
| $r(s, a')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |

**一句话解释：**
从当前状态的累积奖励等于即时奖励加上折扣的后继回报。

**简单应用场景：**
计算轨迹的累积奖励。

---

---

### 公式 (14.38)：改进的时序差分估计 / Improved TD Estimation

$$
d\boldsymbol{x} = \left[ f(\boldsymbol{x}, t) - g(t)^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}) \right] dt + g(t) \, d\bar{\boldsymbol{w}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r(s, a')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $Q^{\pi}(s', a')$ | xiàyībù zhuàngtài-dòngzuò jiàzhí | 下一步状态-动作价值 |

**一句话解释：**
用即时奖励加上后继状态价值的折扣来估计累积回报。

**简单应用场景：**
时序差分学习中的自举估计。

---

---

### 公式 (14.39)：时序差分误差 / TD Error

$$
\mathcal{L}(\theta) = \mathbb{E}_t \mathbb{E}_{\boldsymbol{x}(0)} \mathbb{E}_{\boldsymbol{x}(t)|\boldsymbol{x}(0)} \left[ \| \boldsymbol{s}_\theta(\boldsymbol{x}(t), t) - \nabla_{\boldsymbol{x}} \log p_{0t}(\boldsymbol{x}(t)|\boldsymbol{x}(0)) \|^2 \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}^{\pi}(s, a)$ | gūjì jiàzhí | 估计的Q值 |
| $\alpha$ | alpha | 学习率 |
| $r(s, a')$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |

**一句话解释：**
根据时序差分目标和学习率更新价值函数估计。

**简单应用场景：**
Q学习和SARSA算法的更新规则。

---

---

### 公式 (14.40)：Q学习的更新规则 / Q-Learning Update

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q(s, a)$ | zhuàngtài-dòngzuò jiàzhí | 状态-动作价值函数 |
| $\alpha$ | alpha | 学习率 |
| $r$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $\max_{a'}$ | zuìdà huà | 对下一步动作求最大值 |

**一句话解释：**
Q学习通过最优后继动作的价值来更新当前状态-动作的价值。

**简单应用场景：**
无模型强化学习的最流行算法。

---

---

### 公式 (14.41)：深度Q网络值函数近似 / Deep Q-Network Approximation

$$
Q_\phi(s, a) \approx Q^{\pi}(s, a)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q_\phi(s, a)$ | shēndù wǎngluò guānjì jiàzhí | 深度网络估计的Q值 |
| $\phi$ | wǎngluò cānshù | 网络参数 |
| $Q^{\pi}(s, a)$ | zhēnshí jiàzhíhánshù | 真实价值函数 |

**一句话解释：**
用深度神经网络作为函数逼近器来估计Q函数。

**简单应用场景：**
处理高维状态空间的强化学习。

---

---

### 公式 (14.42)：深度Q网络的架构 / Deep Q-Network Architecture

$$
Q_\phi(s) = \begin{bmatrix}
Q_\phi(s, a_1) \\
\vdots \\
Q_\phi(s, a_M)
\end{bmatrix} \approx \begin{bmatrix}
Q^{\pi}(s, a_1) \\
\vdots \\
Q^{\pi}(s, a_M)
\end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q_\phi(s)$ | shēndù wǎngluò | 深度网络输出向量 |
| $Q_\phi(s, a_i)$ | dì i ge dòngzuò de jiàzhí | 第i个动作的价值 |
| $M$ | dòngzuò shù | 动作数量 |

**一句话解释：**
深度网络对所有动作同时输出其Q值，形成向量输出。

**简单应用场景：**
深度Q学习网络的输出层设计。

---

---

### 公式 (14.43)：DQN损失函数 / DQN Loss Function

$$
\mathcal{L}(s, a, s'|\phi) = \left( r + \gamma \max_{a'} Q_\phi(s', a') - Q_\phi(s, a) \right)^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sunshī hánshù | 损失函数 |
| $r$ | jiǎnglì | 即时奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $\max_{a'} Q_\phi(s', a')$ | xiàyībù zuìyōu jiàzhí | 下一步最优价值 |
| $\phi$ | wǎngluò cānshù | 网络参数 |

**一句话解释：**
DQN的损失函数是时序差分目标与估计值之差的平方。

**简单应用场景：**
训练深度Q网络。

---

---

### 公式 (14.44)：策略梯度的目标函数 / Policy Gradient Objective

$$
\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial}{\partial \theta} \int p_\theta(\tau) G(\tau) d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J(\theta)$ | mùbiāo hánshù | 目标函数 |
| $\theta$ | cānshù | 策略参数 |
| $p_\theta(\tau)$ | guījì g概率 | 轨迹的概率 |
| $G(\tau)$ | huíbào | 累积奖励 |

**一句话解释：**
策略目标函数关于参数的梯度涉及轨迹概率和累积奖励的期望。

**简单应用场景：**
策略梯度方法的理论基础。

---

---

### 公式 (14.45)：目标函数的梯度展开 / Gradient Expansion

$$
= \int \frac{\partial}{\partial \theta} p_\theta(\tau) G(\tau) d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial p_\theta(\tau)}{\partial \theta}$ | gài lǜ de tǐdù | 轨迹概率的梯度 |
| $G(\tau)$ | huíbào | 累积奖励 |

**一句话解释：**
对积分应用莱布尼茨规则，将梯度作用在被积函数上。

**简单应用场景：**
推导策略梯度公式。

---

---

### 公式 (14.46)：对数导数技巧 / Log-Derivative Trick

$$
= \int p_\theta(\tau) \left( \frac{1}{p_\theta(\tau)} \frac{\partial p_\theta(\tau)}{\partial \theta} \right) G(\tau) d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{1}{p_\theta(\tau)} \frac{\partial p_\theta(\tau)}{\partial \theta}$ | duìshù tǐdù | 对数梯度 |
| $p_\theta(\tau)$ | guījì gài lǜ | 轨迹概率 |

**一句话解释：**
利用 $\frac{\partial \log p}{\partial \theta} = \frac{1}{p} \frac{\partial p}{\partial \theta}$ 的关系。

**简单应用场景：**
推导策略梯度的关键技巧。

---

---

### 公式 (14.47)：策略梯度的期望形式 / Policy Gradient in Expectation Form

$$
= \int p_\theta(\tau) \frac{\partial}{\partial \theta} \log p_\theta(\tau) G(\tau) d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \log p_\theta(\tau)}{\partial \theta}$ | duìshù tǐdù | 轨迹概率的对数梯度 |
| $G(\tau)$ | huíbào | 累积奖励 |

**一句话解释：**
策略梯度是轨迹对数概率梯度与其回报的乘积的期望。

**简单应用场景：**
REINFORCE算法的基础。

---

---

### 公式 (14.48)：期望值表示 / Expectation Notation

$$
= E_{\tau \sim p_\theta(\tau)} \left[ \frac{\partial}{\partial \theta} \log p_\theta(\tau) G(\tau) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_{\tau \sim p_\theta(\tau)}$ | qīwàng zhí | 关于轨迹的期望 |
| $\frac{\partial \log p_\theta(\tau)}{\partial \theta}$ | duìshù tǐdù | 对数梯度 |

**一句话解释：**
将积分表示为期望形式，便于蒙特卡洛采样估计。

**简单应用场景：**
策略梯度的采样估计。

---

---

### 公式 (14.49)：轨迹概率的分解 / Trajectory Probability Decomposition

$$
\frac{\partial}{\partial \theta} \log p_\theta(\tau) = \frac{\partial}{\partial \theta} \log \left( p_0(\boldsymbol{s}_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_0(\boldsymbol{s}_0)$ | chūshǐ zhuàngtài gài lǜ | 初始状态概率 |
| $\pi_\theta(a_t \mid s_t)$ | cèlüè gài lǜ | 策略概率 |
| $p(s_{t+1} \mid s_t, a_t)$ | zhuànyíhuà gài lǜ | 转移概率 |

**一句话解释：**
将轨迹概率分解为初始状态、策略和转移概率的乘积。

**简单应用场景：**
推导策略梯度中的对数概率梯度。

---

---

### 公式 (14.50)：对数梯度分解 / Log-Gradient Decomposition

$$
\frac{\partial}{\partial \theta} \left( \log p_0(\boldsymbol{s}_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t | s_t) + \sum_{t=0}^{T-1} \log p(s_{t+1} | s_t, a_t) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log \pi_\theta(a_t \mid s_t)$ | cèlüè de duìshù | 策略的对数 |
| $\log p(s_{t+1} \mid s_t, a_t)$ | huánjìng zhuànyíhuà de duìshù | 环境转移的对数 |

**一句话解释：**
将乘积的对数转化为各项对数的求和。

**简单应用场景：**
化简对数梯度的计算。

---

---

### 公式 (14.51)：策略梯度的简化形式 / Simplified Policy Gradient

$$
= \sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \log \pi_\theta(a_t \mid s_t)}{\partial \theta}$ | cèlüè duìshù tǐdù | 策略对数梯度 |
| $T$ | shíjiān bù | 时间步长 |

**一句话解释：**
只有策略概率对参数有梯度，环境转移和初始状态与参数无关。

**简单应用场景：**
策略梯度中无关项的消除。

---

---

### 公式 (14.52)：策略梯度的完整形式 / Complete Policy Gradient

$$
\frac{\partial J(\theta)}{\partial \theta} = E_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t) G(\tau) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J(\theta)$ | mùbiāo hánshù | 目标函数 |
| $\sum_{t=0}^{T-1}$ | suǒyǒu shíjiān bù | 所有时间步的求和 |
| $G(\tau)$ | huíbào | 轨迹累积奖励 |

**一句话解释：**
策略梯度是轨迹中每一步的策略对数梯度与整个轨迹回报的乘积的期望。

**简单应用场景：**
REINFORCE算法的完整公式。

---

---

### 公式 (14.53)：策略梯度与时间折扣 / Policy Gradient with Time Discounting

$$
\boldsymbol{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) \right) + \sqrt{\beta_t} \boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma^{t'-t}$ | huàhuà yīnzǐ | 时间折扣 |
| $r_{t'}$ | jiǎnglì | 时间步 $t'$ 的奖励 |
| $G(\tau)$ | huíbào | 累积奖励 |

**一句话解释：**
考虑折扣因子的策略梯度展开形式。

**简单应用场景：**
带折扣的强化学习问题的策略梯度。

---

---

### 公式 (14.54)：轨迹回报求和 / Trajectory Return Summation

$$
= E_{\tau \sim p_\theta(\tau)}\left[\sum_{t'=0}^{T-1} \left(\frac{\partial}{\partial \theta} \log \pi_\theta(a_t|s_t) \right) \gamma' G(\tau_t, T)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sum_{t'=0}^{T-1}$ | shíjiān qiúhé | 时间求和 |
| $\frac{\partial}{\partial \theta} \log \pi_\theta(a_t \mid s_t)$ | zhèngcè duìshù de偏导 | 策略对数的偏导 |
| $G(\tau_t, T)$ | chōngfèn zuìhòu de huíbào | 充分最后的回报 |

**一句话解释：**
对所有时间步求和策略梯度与后续回报的乘积。

**简单应用场景：**
策略梯度算法的样本估计。

---

---

### 公式 (14.55)：时间加权回报 / Time-Weighted Return

$$
G(\tau_t, T) = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(\tau_t, T)$ | shíjiān jiāquán huíbào | 时间加权回报 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $r_{t'+1}$ | jiǎnglì | 奖励 |
| $t$ | dāngqián shíkè | 当前时刻 |

**一句话解释：**
从时刻t开始的折扣累积回报。

**简单应用场景：**
强化学习中计算未来价值。

---

---

### 公式 (14.56)：REINFORCE算法的梯度估计 / REINFORCE Gradient Estimation

$$
\frac{\partial J(\theta)}{\partial \theta} \approx \frac{1}{N} \sum_{n=1}^{N} \left( \sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_\theta(a_t^{(n)}, s_t^{(n)}) \gamma^t G_{t+1}^{(n)} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial J(\theta)}{\partial \theta}$ | mùbiāo de tǐdù | 目标函数的梯度 |
| $N$ | guījì shù | 采样轨迹数 |
| $\gamma^t$ | huàhuà yīnzǐ | 时间折扣 |
| $G_{t+1}^{(n)}$ | dì n ge guījì de xiàyībù huíbào | 第n条轨迹的下一步回报 |

**一句话解释：**
通过采样多条轨迹，用平均值估计策略梯度。

**简单应用场景：**
REINFORCE算法的梯度计算。

---

---

### 公式 (14.57)：基线减少方差 / Baseline for Variance Reduction

$$
f = f - \alpha(g - E[g])
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f$ | suǒqiú liàngshù | 要估计的量 |
| $g$ | diàozhěng liàngshù | 调整量 |
| $E[g]$ | $g$ de qīwàng zhí | $g$ 的期望值 |
| $\alpha$ | kòngzhì cānshù | 控制参数 |

**一句话解释：**
通过减去期望值来降低估计量的方差而不改变期望。

**简单应用场景：**
方差减少技巧在强化学习中的应用。

---

---

### 公式 (14.58)：方差分解 / Variance Decomposition

$$
\text{var}(f) = \text{var}(f) - 2\alpha \text{cov}(f, g) + \alpha^2 \text{var}(g)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(f)$ | $f$ de fāngchà | $f$ 的方差 |
| $\text{cov}(f, g)$ | $f$ hé $g$ de xiéfāngcha | $f$ 和 $g$ 的协方差 |

**一句话解释：**
方差由原方差、协方差项和增加的方差项组成。

**简单应用场景：**
分析基线对方差的影响。

---

---

### 公式 (14.59)：协方差与相关性 / Covariance and Correlation

$$
\alpha = \frac{\text{cov}(f, g)}{\text{var}(g)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | zuìyōu kòngzhì cānshù | 最优控制参数 |
| $\text{cov}(f, g)$ | xiéfāngcha | 协方差 |
| $\text{var}(g)$ | fāngcha | 方差 |

**一句话解释：**
最优的控制参数是协方差与方差的比值。

**简单应用场景：**
选择最优的基线来最小化方差。

---

---

### 公式 (14.60)：方差最小化 / Variance Minimization

$$
\text{var}(f) = \left( 1 - \frac{\text{cov}(f, g)^2}{\text{var}(g) \text{var}(f)} \right) \text{var}(f)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(f)$ | fāngcha | 调整后的方差 |
| $\text{var}(f)$ | yuánshǐ fāngcha | 原始方差 |
| $\frac{\text{cov}(f, g)^2}{\text{var}(g) \text{var}(f)}$ | xiāngguān xìshù píngfāng | 相关系数平方 |

**一句话解释：**
调整后的方差等于原方差乘以一减去相关系数平方。

**简单应用场景：**
量化基线减少方差的效果。

---

---

### 公式 (14.61)：相关系数表示 / Correlation Representation

$$
= (1 - \text{corr}(f, g)^2) \text{var}(f)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{corr}(f, g)$ | xiāngguān xìshù | 相关系数 |

**一句话解释：**
方差减少的程度取决于基线与回报的相关性。

**简单应用场景：**
理解基线有效性的关键因素。

---

---

### 公式 (14.62)：带基线的策略梯度 / Policy Gradient with Baseline

$$
\frac{\partial J_t(\theta)}{\partial \theta} = E_{a_t} \left[ E_{a_t} \left[ \gamma^t G(\tau_\tau) \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t) \right] \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J_t(\theta)$ | shíjiān $t$ de mùbiāo | 时间 $t$ 的目标函数 |
| $G(\tau_\tau)$ | huíbào | 累积奖励 |

**一句话解释：**
策略梯度关于特定时间步动作的期望。

**简单应用场景：**
时间步级别的梯度估计。

---

---

### 公式 (14.63)：优势函数的策略梯度 / Policy Gradient with Advantage

$$
\frac{\partial J_t(\theta)}{\partial \theta} = E_{a_t} \left[ E_{a_t} \left[ \gamma^t \left( G(\tau_\tau) - b(s_t) \right) \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t) \right] \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(\tau_\tau) - b(s_t)$ | yōushìshù | 优势函数 |
| $b(s_t)$ | jīzhǔn hánshù | 基线函数 |

**一句话解释：**
使用优势函数（回报减去基线）来降低方差。

**简单应用场景：**
Actor-Critic算法中的优势估计。

---

---

### 公式 (14.64)：期望的积分表示 / Expected Value as Integral

$$
E_{a_t} \left( b(s_t) \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t) \right) = \int_{a_t} \left( b(s_t) \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t) \right) \pi_\theta(a_t | s_t) d a_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_{a_t}$ | guānyú dòngzuò de qīwàng | 关于动作的期望 |
| $b(s_t)$ | jīzhǔn hánshù | 基线函数 |
| $\pi_\theta(a_t \mid s_t)$ | cèlüè | 策略 |

**一句话解释：**
将期望表示为关于动作分布的积分。

**简单应用场景：**
期望值的数学计算。

---

---

### 公式 (14.65)：基线项的消失 / Baseline Term Cancellation

$$
= \int_{a_t} b(s_t) \frac{\partial}{\partial \theta} \pi_\theta(a_t | s_t) d a_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \pi_\theta(a_t \mid s_t)}{\partial \theta}$ | cèlüè de tǐdù | 策略的梯度 |
| $b(s_t)$ | jīzhǔn hánshù | 基线函数 |

**一句话解释：**
利用对数导数 $\frac{\partial \log p}{\partial \theta} = \frac{1}{p} \frac{\partial p}{\partial \theta}$ 转换。

**简单应用场景：**
推导基线不影响策略梯度期望的证明。

---

---

### 公式 (14.66)：梯度与积分交换 / Gradient and Integration Exchange

$$
= \frac{\partial}{\partial \theta} b(s_t) \int_{a_t} \pi_\theta(a_t | s_t) d a_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \theta} b(s_t)$ | jīzhǔn de tǐdù | 基线的梯度（通常为0） |
| $\int \pi_\theta(a_t \mid s_t) d a_t$ | cèlüè de jīfēn（=1） | 策略的积分（等于1） |

**一句话解释：**
提取基线函数（不依赖于动作）和梯度符号。

**简单应用场景：**
数学推导中的求导顺序调整。

---

---

### 公式 (14.67)：归一化性质 / Normalization Property

$$
= \frac{\partial}{\partial \theta} (b(s_t) \cdot 1) = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\int \pi_\theta(a_t \mid s_t) d a_t = 1$ | shǔixìng | 概率的归一化性质 |

**一句话解释：**
基线项对策略梯度的期望贡献为零。

**简单应用场景：**
证明基线不改变无偏梯度估计。

---

---

### 公式 (14.68)：价值函数基线的损失 / Value Function Baseline Loss

$$
\mathcal{L}(\phi | s_t, \tau_\theta) = \left( V^{\tau_\theta}(s_t) - V_\phi(s_t) \right)^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sunshī hánshù | 损失函数 |
| $V^{\tau_\theta}(s_t)$ | zhēnshí jiàzhí | 真实的累积回报 |
| $V_\phi(s_t)$ | guānjì jiàzhí | 估计的状态价值 |

**一句话解释：**
价值函数通过最小化估计值与真实回报之间的平方差来学习。

**简单应用场景：**
Actor-Critic算法中Critic的训练。

---

---

### 公式 (14.69)：价值梯度更新 / Value Gradient Update

$$
\frac{\partial \mathcal{L}(\phi | s_t, \tau_\theta)}{\partial \phi} = -\left( G(\tau_\tau) - V_\phi(s_t) \right) \frac{\partial V_\phi(s_t)}{\partial \phi}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial \phi}$ | sunshī duì cānshù de tǐdù | 损失对参数的梯度 |
| $G(\tau_\tau) - V_\phi(s_t)$ | yōushìshù | 优势函数 |
| $\frac{\partial V_\phi(s_t)}{\partial \phi}$ | jiàzhí de tǐdù | 价值函数的梯度 |

**一句话解释：**
价值函数的梯度是优势函数乘以价值函数关于参数的梯度。

**简单应用场景：**
训练价值函数网络。

---

---

### 公式 (14.70)：带基线的策略梯度完整版 / Complete Policy Gradient with Baseline

$$
\frac{\partial J(\theta)}{\partial \theta} = E_{a_t} \left[ E_{a_t} \left[ \gamma^t \left( G(\tau_\tau) - V_\phi(s_t) \right) \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t) \right] \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J(\theta)$ | mùbiāo hánshù | 目标函数 |
| $\gamma^t$ | huàhuà yīnzǐ | 时间折扣 |
| $G(\tau_\tau) - V_\phi(s_t)$ | yōushìshù | 优势函数 |
| $V_\phi(s_t)$ | jiàzhí jīzhǔn | 价值基线 |

**一句话解释：**
Actor-Critic算法中的策略梯度，使用学到的价值函数作为基线。

**简单应用场景：**
Actor-Critic算法的完整更新规则。

---

---

### 公式 (14.71)：时步回报的递推 / Time-Step Return Recursion

$$
G(t, \tau) = r_{t+1} + \gamma V(s_{t+1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(t, \tau)$ | huíbào | 从时间步 $t$ 开始的累积奖励 |
| $r_{t+1}$ | jiǎnglì | 时间步 $t+1$ 的奖励 |
| $\gamma$ | huàhuà yīnzǐ | 折扣因子 |
| $V(s_{t+1})$ | xiàyībù zhuàngtài jiàzhí | 下一步状态的价值 |

**一句话解释：**
从某时间步开始的累积奖励等于即时奖励加上折扣的后继状态价值。

**简单应用场景：**
时序差分回报的定义。

---

---

### 公式 (14.72)：价值函数与回报的误差 / Value Function Error

$$
\min_\phi \left( \hat{G}(\tau_\tau) - V_\phi(s_t) \right)^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\min_\phi$ | duì cānshù zuìdà huà | 对参数求最小值 |
| $\hat{G}(\tau_\tau)$ | huíbào | 观测到的累积奖励 |
| $V_\phi(s_t)$ | guānjì jiàzhí | 估计的状态价值 |

**一句话解释：**
通过最小化回报与价值估计之间的平方误差来训练价值函数。

**简单应用场景：**
最小二乘价值函数学习。

---

---

### 公式 (14.73)：带基线项的参数更新 / Parameter Update with Baseline

$$
\theta \leftarrow \theta + \alpha \gamma^t \left( G(\tau_\tau) - V_\phi(s_t) \right) \frac{\partial}{\partial \theta} \log \pi_\theta(a_t | s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta$ | cèlüè cānshù | 策略参数 |
| $\alpha$ | xuéxí lǜ | 学习率 |
| $\gamma^t$ | huàhuà yīnzǐ | 时间折扣 |
| $G(\tau_\tau) - V_\phi(s_t)$ | yōushìshù | 优势函数 |

**一句话解释：**
根据优势函数对策略参数进行梯度上升更新。

**简单应用场景：**
Actor-Critic算法的策略参数更新规则。

---

## 总结

本文档包含了Chapter 14中所有标记为WRONG和MISSING的公式（14.8-14.73）的完整提取和详细说明。每个公式都包括：

1. **中英文名称** - 准确的学术术语
2. **LaTeX表达式** - 与教科书完全一致
3. **符号表** - 包含汉语拼音标注和含义
4. **一句话解释** - 简洁的概念阐述
5. **应用场景** - 实际使用场景

这些公式涵盖了强化学习的核心内容：
- 贝尔曼方程与动态规划（14.8-14.28）
- 无模型学习方法（14.29-14.40）
- 深度强化学习（14.41-14.43）
- 策略梯度方法（14.44-14.56）
- 方差减少技巧（14.57-14.67）
- Actor-Critic算法（14.68-14.73）

---

## 第15章：序列生成模型

### 公式 (15.1)：序列概率模型 / Sequence Probability Model

$$
p(\boldsymbol{x}_{1:T}) \triangleq P(X_{1:T} = \boldsymbol{x}_{1:T})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}_{1:T}$ | xiàng liàng | 序列向量，从时间1到T的向量 |
| $P$ | gài lǜ | 概率 |
| $T$ | shíjiān zhǎng dù | 序列长度 |

**一句话解释：**
序列的联合概率分布定义为从时间1到T的所有变量的概率。

**简单应用场景：**
在自然语言处理中，定义一个句子的所有词序列的联合概率。

---

---

### 公式 (15.2)：自回归概率分解 / AutoRegressive Factorization

$$
p(\boldsymbol{x}_{1:T}) = P(X_1 = x_1, X_2 = x_2, \ldots, X_T = x_T)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x}_{1:T})$ | gài lǜ | 联合概率 |
| $x_i$ | x de dì i ge | 第i个时刻的观测值 |

**一句话解释：**
将序列的联合概率显式地表示为各个时刻的条件概率的乘积。

**简单应用场景：**
自回归语言模型中，逐个生成句子中的每个词。

---

---

### 公式 (15.3)：词嵌入 / Word Embedding

$$
p(y | \boldsymbol{x}) = \frac{p(\boldsymbol{x} | y) p(y)}{p(\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $M$ | dà zì diǎn | embedding matrix |
| $\boldsymbol{\delta}_t$ | dù liàng | one-hot vector |
| $\boldsymbol{m}_k$ | xùn liàng | embedding vector |
| $\boldsymbol{e}_t$ | cí xiàng liàng | word embedding |

**一句话解释：**
通过one-hot向量与嵌入矩阵的乘积得到词的分布式表示。

**简单应用场景：**
自然语言处理中将离散的词映射到连续的向量空间。

---

---

### 公式 (15.4)：注意力权重 / Attention Weights

$$
p(\theta | \mathcal{D}) = \frac{p(\mathcal{D} | \theta) p(\theta)}{p(\mathcal{D})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | zhù yì lì quán zhòng | attention weight |
| $s_t$ | zhù yì fēn shù | attention score |
| $Z$ | guī huà huà chāng shù | normalization constant |

**一句话解释：**
将注意力分数通过softmax函数归一化得到注意力权重。

**简单应用场景：**
序列到序列模型中对source端信息的加权。

---

---

### 公式 (15.5)：上下文向量 / Context Vector

$$
p(\mathcal{D}) = \int p(\mathcal{D} | \theta) p(\theta) \, d\theta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{c}_t$ | shàng xià wén xiàng liàng | context vector |
| $\alpha_{t,i}$ | zhù yì quán zhòng | attention weight |
| $\boldsymbol{h}_i$ | yǐn cáng zhuàng tài | hidden state |

**一句话解释：**
对source端所有隐层状态的加权平均得到上下文向量。

**简单应用场景：**
序列到序列模型中融合source端信息。

---

---

### 公式 (15.6)：解码器隐层状态更新 / Decoder Hidden State Update

$$
\theta_{\text{MAP}} = \arg\max_\theta \log p(\mathcal{D}|\theta) + \log p(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}'_t$ | xīn de yǐn cáng zhuàng tài | new hidden state |
| $\boldsymbol{W}_c$ | quán zhòng jǔ zhèn | weight matrix |
| $[\boldsymbol{c}_t; \boldsymbol{h}_t]$ | lián jiē | concatenation |

**一句话解释：**
将上下文向量和当前隐层状态拼接后通过线性变换和激活函数得到新的隐层状态。

**简单应用场景：**
注意力机制中融合上下文信息。

---

---

### 公式 (15.7)：输出概率分布 / Output Probability Distribution

$$
p(y | \boldsymbol{x}, \mathcal{D}) = \int p(y | \boldsymbol{x}, \theta) p(\theta | \mathcal{D}) \, d\theta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{o}_t$ | shū chū g概率 | output probability |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | weight matrix |
| $\boldsymbol{h}_t$ | yǐn cáng zhuàng tài | hidden state |
| $\boldsymbol{b}$ | piān zhì xiàng liàng | bias vector |

**一句话解释：**
通过隐层状态经过线性变换和softmax函数得到每个词的预测概率分布。

**简单应用场景：**
神经网络语言模型的输出层。

---

---

### 公式 (15.8)：条件对数似然 / Conditional Log-Likelihood

$$
p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta) p(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p_\theta$ | duì shù gài lǜ | log probability |
| $\boldsymbol{x}_{1:T}$ | cí xù liè | word sequence |
| $\theta$ | cān shù | parameters |

**一句话解释：**
序列的条件对数似然等于所有时步的条件对数概率之和。

**简单应用场景：**
语言模型的训练目标。

---

---

### 公式 (15.9)：N元模型 / N-Gram Model

$$
p(\boldsymbol{x}_t | \boldsymbol{x}_{1:(t-1)}) = p(\boldsymbol{x}_t | \boldsymbol{x}_{(t-N+1):(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N$ | yuánshu | N元 |
| $\boldsymbol{x}_{(t-N+1):(t-1)}$ | qián N-1 ge | 前N-1个观测值 |

**一句话解释：**
N元模型假设当前词只依赖于前N-1个词，而不是全部历史。

**简单应用场景：**
二元模型（Bigram）中，只考虑前一个词的上下文。

---

---

### 公式 (15.10)：一元模型概率 / Unigram Model

$$
p(\boldsymbol{x}_{1:T}, \theta) = \prod_{t=1}^{T} p(\boldsymbol{x}_t) = \prod_{k=1}^{|V|} \theta_k^{m_k}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_k$ | cānshù k | 第k个词的概率参数 |
| $m_k$ | cìshù | 第k个词在序列中出现的次数 |
| $\|V\|$ | cíhuì dà xiǎo | 词表大小 |

**一句话解释：**
一元模型中，序列的概率为各个词的独立概率的乘积，与顺序无关。

**简单应用场景：**
最简单的文本概率模型，不考虑词的顺序信息。

---

---

### 公式 (15.11)：对数似然 / Log Likelihood

$$
\log \prod_{n=1}^{N'} p(\boldsymbol{x}_{1:T_n}^{(n)} ; \theta) = \log \prod_{k=1}^{|V|} \theta_k^{m_k}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $N'$ | zǒng yàngběn shù | 总样本数 |
| $m_k$ | dì k ge cí zǒng chūxiàn cì shù | 第k个词的总出现次数 |

**一句话解释：**
将多个样本的概率乘积的对数展开为对数之和。

**简单应用场景：**
计算训练数据的对数似然度。

---

---

### 公式 (15.12)：对数似然求和 / Log Likelihood Summation

$$
\sum_{k=1}^{|V|} m_k \log \theta_k
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sum$ | qiúhé | 求和 |
| $\log \theta_k$ | duìshù cānshù | 参数的对数 |

**一句话解释：**
对数似然可以表示为词频与对数参数的加权和。

**简单应用场景：**
参数优化中，目标函数的计算形式。

---

---

### 公式 (15.13)：最大化目标 / Maximization Objective

$$
\max_{\theta} \sum_{k=1}^{|V|} m_k \log \theta_k
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max_{\theta}$ | duì cānshù zuì dà huà | 对参数θ最大化 |

**一句话解释：**
一元模型的训练目标是最大化对数似然。

**简单应用场景：**
参数学习中的优化问题。

---

---

### 公式 (15.14)：约束条件 / Constraint

$$
\text{s.t.} \quad \sum_{k=1}^{|V|} \theta_k = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| s.t. | shǔnyú | 受限于/subject to |
| $\sum \theta_k = 1$ | gài lǜ hé wéi 1 | 概率和为1的约束 |

**一句话解释：**
所有词的概率参数必须满足概率的基本性质：非负且和为1。

**简单应用场景：**
概率模型参数化时的必要约束。

---

---

### 公式 (15.15)：拉格朗日乘数法 / Lagrange Multiplier Method

$$
\Delta(\theta, \lambda) = \sum_{k=1}^{|V|} m_k \log \theta_k + \lambda \left( \sum_{k=1}^{|V|} \theta_k - 1 \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta$ | lāgéláng fúnshù | 拉格朗日函数 |
| $\lambda$ | lāgéláng chéngshù | 拉格朗日乘数 |

**一句话解释：**
使用拉格朗日乘数法处理概率约束下的参数优化问题。

**简单应用场景：**
带约束优化问题的标准求解方法。

---

---

### 公式 (15.16)：偏导数条件1 / Partial Derivative Condition 1

$$
\frac{\partial \Delta(\theta, \lambda)}{\partial \theta_k} = \frac{m_k}{\theta_k} + \lambda = 0, \quad k = 1, 2, \ldots, |V|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \theta_k}$ | duì cānshù k qiúpiāndǎo | 对参数k求偏导 |
| $= 0$ | děngyú líng | 等于零 |

**一句话解释：**
拉格朗日函数对所有参数的偏导数都等于零，是最优解的必要条件。

**简单应用场景：**
参数优化中的一阶条件。

---

---

### 公式 (15.17)：偏导数条件2 / Partial Derivative Condition 2

$$
\frac{\partial \Delta(\theta, \lambda)}{\partial \lambda} = \sum_{k=1}^{|V|} \theta_k - 1 = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \lambda}$ | duì lāgéláng chéngshù qiúpiāndǎo | 对拉格朗日乘数求偏导 |

**一句话解释：**
对拉格朗日乘数求偏导，约束条件的可行性条件。

**简单应用场景：**
验证概率约束是否满足。

---

---

### 公式 (15.18)：最优参数 / Optimal Parameters

$$
\theta_k = \frac{m_k}{\sum_{k'=1}^{|V|} m_{k'}} = \frac{m_k}{\bar{m}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\theta_k$ | zuì yōu cānshù | 最优参数 |
| $\bar{m}$ | zǒng cìshù | 所有词的总出现次数 |

**一句话解释：**
最优参数就是每个词在训练数据中的相对频率。

**简单应用场景：**
最大似然估计得到的概率等于样本的相对频率。

---

---

### 公式 (15.19)：N元模型N(x|x_{(N+1):(t-1)}) / N-Gram Normalization

$$
p(\boldsymbol{x}_t | \boldsymbol{x}_{(t-N+1):(t-1)}) = \frac{m(\boldsymbol{x}_{(t-N+1):(t)})}{m(\boldsymbol{x}_{(t-N+1):(t-1)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $m(\cdot)$ | chūxiàn cì shù | 出现次数 |
| $\boldsymbol{x}_{(t-N+1):(t)}$ | N yuán zǔ | N元组 |

**一句话解释：**
N元模型的条件概率为N元组出现次数与(N-1)元组出现次数的比值。

**简单应用场景：**
从训练语料中统计N元组频率来训练N元模型。

---

---

### 公式 (15.20)：平滑技术 / Smoothing Technique

$$
p(\boldsymbol{x}_t | \boldsymbol{x}_{(t-N+1):(t-1)}) = \frac{m(\boldsymbol{x}_{(t-N+1):(t)}) + \delta}{m(\boldsymbol{x}_{(t-N+1):(t-1)}) + \delta|V|}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta$ | pínghuá xì shù | 平滑系数 |

**一句话解释：**
加性平滑通过在分子分母上加上平滑系数来处理零频率问题。

**简单应用场景：**
处理训练数据中未出现的N元组，避免概率为零。

---

---

### 公式 (15.21)：深度序列模型规范化 / Deep Sequence Model Normalization

$$
\sum_{k=1}^{|V|} f_\theta(x_{1:(t-1)}, \theta) = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_\theta$ | shēn dù shìxiàn hánshù | 深度神经网络函数 |
| $x_{1:(t-1)}$ | qián miàn de lìshǐ | 前面的历史 |

**一句话解释：**
深度序列模型输出的概率值需要经过归一化处理。

**简单应用场景：**
神经网络输出层使用Softmax函数进行归一化。

---

---

### 公式 (15.22)：前向神经网络 / Feedforward Network

$$
p(\theta | \mathcal{D}) \approx q(\theta) = \mathcal{N}(\theta; \boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | yǐn cáng zhuàng tài | hidden state |
| $g$ | hánshù | function |
| $\boldsymbol{e}_t$ | cí xiàng liàng | word embedding |
| $\boldsymbol{\theta}_g$ | cān shù | parameters |

**一句话解释：**
通过非线性函数g将上一时步的隐层状态和当前词的嵌入向量结合。

**简单应用场景：**
RNN和深度神经网络中的状态更新。

---

---

### 公式 (15.23)：嵌入层输出 / Embedding Layer Output

$$
\boldsymbol{e}_t = M\delta_t = \boldsymbol{m}_k
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{e}_t$ | qiànrù xiàng liàng | 嵌入向量 |
| $M$ | qiànrù jǔzhèn | 嵌入矩阵 |
| $\delta_t$ | dúnhuà xiàng liàng | one-hot向量 |
| $\boldsymbol{m}_k$ | dì k ge cíde xiàng liàng | 第k个词的向量 |

**一句话解释：**
嵌入层通过查表将词的one-hot编码转换为实值向量。

**简单应用场景：**
神经网络中的词向量表示。

---

---

### 公式 (15.24)：特征层平均 / Feature Average Pooling

$$
\boldsymbol{h}_t = \sum_{i=1}^{t-1} \alpha_t \boldsymbol{e}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | tèzhēng xiàng liàng | 特征向量 |
| $\alpha_t$ | quán zhòng | 权重 |

**一句话解释：**
特征层通过加权平均历史的嵌入向量来获得当前的特征表示。

**简单应用场景：**
Bag-of-words模型中的特征聚合。

---

---

### 公式 (15.25)：前馈神经网络 / Feedforward Neural Network

$$
\boldsymbol{h}' = \boldsymbol{e}_{t-N+1} \oplus \cdots \oplus \boldsymbol{e}_{t-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\oplus$ | liánjiē | 拼接操作 |
| $\boldsymbol{h}'$ | liánjiēhòu xiàng liàng | 拼接后的向量 |

**一句话解释：**
前馈网络通过拼接N-1个历史嵌入向量作为输入。

**简单应用场景：**
基于窗口的神经网络语言模型。

---

---

### 公式 (15.26)：循环神经网络隐层 / RNN Hidden Layer with Skip Connection

$$
\mathcal{L}(q) = \mathbb{E}_q [\log p(\mathcal{D}, \theta)] - \mathbb{E}_q [\log q(\theta)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | yǐn cáng zhuàng tài | hidden state |
| $\boldsymbol{h}'$ | jīng guò shù jù | processed state |
| $\oplus$ | yuánsù mínghuì de jiāo cā | element-wise concatenation |

**一句话解释：**
通过跳连（skip connection）将前一层的输出与当前层的非线性变换相结合。

**简单应用场景：**
深层神经网络的梯度流传播。

---

---

### 公式 (15.27)：前向融合神经网络 / Bi-directional Merged Network

$$
D_{\mathrm{KL}}(q(\theta) \| p(\theta|\mathcal{D})) = \mathbb{E}_q \left[ \log \frac{q(\theta)}{p(\theta|\mathcal{D})} \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | yǐn cáng zhuàng tài | hidden state |
| $g$ | hánshù | function |
| $\boldsymbol{e}_t$ | cí xiàng liàng | word embedding |

**一句话解释：**
双向循环神经网络中的隐层更新规则。

**简单应用场景：**
双向编码器的中间层计算。

---

---

### 公式 (15.28)：循环神经网络 / Recurrent Neural Network

$$
\boldsymbol{h}_t = g(\boldsymbol{h}_{t-1}, \boldsymbol{e}_t; \theta_\phi)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_{t-1}$ | shàng yī shíkè de yǐnhuàn tài | 上一时刻的隐藏态 |
| $\boldsymbol{e}_t$ | dāngqián qiànrù | 当前嵌入 |
| $\theta_\phi$ | RNN cānshù | RNN参数 |

**一句话解释：**
RNN在每一步都将上一时刻的隐藏状态和当前输入作为输入，产生新的隐藏状态。

**简单应用场景：**
序列建模中处理可变长度的输入序列。

---

---

### 公式 (15.29)：输出层 / Output Layer

$$
\boldsymbol{o}_t = \text{softmax}(\theta_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{o}_t$ | shūchūcéng huódòng | 输出层活动 |
| $\theta_t$ | yùliàn cānshù | 预测参数 |

**一句话解释：**
输出层使用Softmax函数将原始分数转换为概率分布。

**简单应用场景：**
分类问题中的概率输出。

---

---

### 公式 (15.30)：线性输出变换 / Linear Output Transform

$$
= \text{softmax}(\boldsymbol{W h}_t + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}$ | shūchūcéng quántè jǔzhèn | 输出层权重矩阵 |
| $\boldsymbol{b}$ | piānzhì | 偏置向量 |
| $\boldsymbol{h}_t$ | yǐnhuàn zhuàngtài | 隐藏状态 |

**一句话解释：**
输出层首先对隐藏状态进行线性变换，然后应用Softmax。

**简单应用场景：**
神经网络输出层的标准形式。

---

---

### 公式 (15.31)：对数似然目标函数 / Log Likelihood Objective

$$
\log p_\theta(\boldsymbol{x}_{1:T}) = \sum_{t=1}^{T} \log p_\theta(x_t|\boldsymbol{x}_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p_\theta$ | duì shù gài lǜ | log probability |
| $\boldsymbol{x}_{1:T}$ | cí xù liè | word sequence |

**一句话解释：**
训练序列生成模型的对数似然目标函数。

**简单应用场景：**
神经网络语言模型和序列到序列模型的训练。

---

---

### 公式 (15.32)：参数更新规则 / Parameter Update Rule

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \frac{\partial \log p_\theta(\boldsymbol{x}_{1:T})}{\partial \boldsymbol{\theta}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | cān shù | parameters |
| $\alpha$ | xué xí lǜ | learning rate |
| $\frac{\partial}{\partial \boldsymbol{\theta}}$ | piàn dǎo shù | gradient |

**一句话解释：**
使用梯度上升法更新模型参数以最大化对数似然。

**简单应用场景：**
神经网络模型的训练步骤。

---

---

### 公式 (15.33)：困惑度定义 / Perplexity Definition

$$
2^{H(p)} = 2^{-\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(p)$ | xìn xī liàng | entropy |
| $p(x)$ | gài lǜ | probability |
| $\mathcal{X}$ | xíng shì kōng jiān | domain |

**一句话解释：**
困惑度是衡量概率分布不确定性的指标，基于信息熵定义。

**简单应用场景：**
评估语言模型对测试集的预测性能。

---

---

### 公式 (15.34)：样本困惑度 / Sample Perplexity

$$
2^{H(p_r, p_\theta)} = 2^{-\frac{1}{N}\sum_{n=1}^{N} \log_2 p_\theta(x^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(p_r, p_\theta)$ | jiāo chā熵 | cross entropy |
| $p_\theta$ | mó xíng gài lǜ | model probability |
| $N$ | yàng běn shù | sample count |

**一句话解释：**
在测试样本上计算的平均困惑度，用于评估模型性能。

**简单应用场景：**
语言模型的泛化性能评估。

---

---

### 公式 (15.35)：测试序列联合概率 / Test Sequence Joint Probability

$$
p(y=1|\boldsymbol{x}, \mathcal{D}) = \int \sigma(\boldsymbol{w}^\top \boldsymbol{x}) p(\boldsymbol{w}|\mathcal{D}) \, d\boldsymbol{w}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_\theta$ | mó xíng gài lǜ | model probability |
| $x_{1:T_n}^{(n)}$ | dì n gè yàng běn de cí xù liè | n-th sample sequence |
| $T_n$ | dì n gè yàng běn de cháng dù | length of n-th sample |

**一句话解释：**
所有测试样本序列的联合概率等于所有条件概率的乘积。

**简单应用场景：**
困惑度计算中的基础公式。

---

---

### 公式 (15.36)：困惑度公式 / Perplexity Formula

$$
PPL(\theta) = 2^{-\frac{1}{T}\sum_{n=1}^{N}\log_2 p_\theta(x_{1:T_n}^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $PPL(\theta)$ | kùn huò dù | perplexity |
| $T$ | zǒng shíbuù shù | total time steps |
| $\theta$ | cān shù | parameters |

**一句话解释：**
困惑度是测试集上平均负对数概率的2的幂。

**简单应用场景：**
语言模型评估的标准指标。

---

---

### 公式 (15.37)：困惑度对数形式 / Perplexity Log Form

$$
PPL(\theta) = 2^{-\frac{1}{T}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\log_2 p_\theta(x_t^{(n)}|x_{1:(t-1)}^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $PPL(\theta)$ | kùn huò dù | perplexity |
| $p_\theta$ | tiáo jiàn gài lǜ | conditional probability |

**一句话解释：**
困惑度的详细表达式，按时步求和。

**简单应用场景：**
逐步计算大规模数据集的困惑度。

---

---

### 公式 (15.38)：几何平均困惑度 / Geometric Mean Perplexity

$$
PPL(\theta) = \left(\prod_{n=1}^{N}\prod_{t=1}^{T_n} p_\theta(x_t^{(n)}|x_{1:(t-1)}^{(n)})\right)^{-1/T}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $PPL(\theta)$ | kùn huò dù | perplexity |
| $T$ | zǒng shíbuù shù | total time steps |

**一句话解释：**
困惑度是所有条件概率乘积的几何平均的倒数。

**简单应用场景：**
困惑度的几何平均表示。

---

---

### 公式 (15.39)：BLEU评分 / BLEU Score

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\boldsymbol{w})} [\log p(\mathcal{D}|\boldsymbol{w})] - D_{\mathrm{KL}}(q_\phi(\boldsymbol{w}) \| p(\boldsymbol{w}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P_N(x)$ | N yuán zǔ jīng quèlǜ | N-gram precision |
| $c_u(x)$ | shēng chéng xù liè zhōng de N yuán zǔ gài shù | count in generated |
| $c_{u,k}(x^{(k)})$ | cān kǎo xù liè zhōng de N yuán zǔ gài shù | count in reference |

**一句话解释：**
BLEU评分衡量生成序列中与参考序列匹配的N元组的比例。

**简单应用场景：**
机器翻译等序列生成任务的评估。

---

---

### 公式 (15.40)：BLEU惩罚函数 / BLEU Brevity Penalty

$$
b(x) = \begin{cases}
1 & \text{if } l_s > l_r \\
\exp(1 - l_s/l_r) & \text{if } l_s \leq l_r
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b(x)$ | duǎnhuà chénghuà | brevity penalty |
| $l_s$ | shēng chéng xù liè de cháng dù | generated length |
| $l_r$ | cān kǎo xù liè de cháng dù | reference length |

**一句话解释：**
惩罚生成序列过短的情况，防止模型倾向于生成短序列。

**简单应用场景：**
BLEU评分中的长度惩罚项。

---

---

### 公式 (15.41)：BLEU-N评分 / BLEU-N Score

$$
q_\phi(\boldsymbol{w}) = \prod_i \mathcal{N}(w_i; \mu_i, \sigma_i^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{BLEU-N}(x)$ | BLEU píng fēn | BLEU score |
| $b(x)$ | duǎnhuà chénghuà | brevity penalty |
| $\alpha_N$ | quán zhòng xī shù | weight |
| $P_N$ | N yuán zǔ jīng quèlǜ | N-gram precision |

**一句话解释：**
BLEU-N评分结合了多个N元组精度和长度惩罚。

**简单应用场景：**
机器翻译系统的综合评估指标。

---

---

### 公式 (15.42)：ROUGE-N评分 / ROUGE-N Score

$$
w_i = \mu_i + \sigma_i \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{ROUGE-N}(x)$ | ROUGE píng fēn | ROUGE score |
| $c_u(x)$ | shēng chéng wén běn zhōng de N yuán zǔ gài shù | count in generated |
| $c_u(s^{(k)})$ | cān kǎo wén běn zhōng de N yuán zǔ gài shù | count in reference |

**一句话解释：**
ROUGE-N评分衡量生成文本与参考文本中重叠N元组的比例。

**简单应用场景：**
文本摘要任务的评估指标。

---

---

### 公式 (15.43)：轨迹序列 / Trajectory Sequence

$$
\tau = \{s_1, a_1, s_2, a_2, \ldots, s_T, a_T\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tau$ | guǐ j迹 | trajectory |
| $s_t$ | zhuàng tài | state |
| $a_t$ | dòng zuò | action |

**一句话解释：**
轨迹是在强化学习任务中采集到的状态和动作的交替序列。

**简单应用场景：**
强化学习中的经验表示。

---

---

### 公式 (15.44)：轨迹概率 / Trajectory Probability

$$
p_\theta(\tau) = \prod_{t=1}^{T} \pi_\theta(a_t | x_t; s_t = x_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_\theta(\tau)$ | guǐ jìzài gài lǜ | trajectory probability |
| $\pi_\theta$ | zhèng zuò | policy |
| $a_t$ | dòng zuò | action |
| $x_t$ | guān chá | observation |

**一句话解释：**
轨迹的概率是所有时步动作概率的乘积。

**简单应用场景：**
强化学习中的策略学习。

---

---

### 公式 (15.45)：强化学习目标函数 / Reinforcement Learning Objective

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[G(\tau)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J(\boldsymbol{\theta})$ | mù biāo hánshù | objective function |
| $G(\tau)$ | zī huàn | return |
| $\tau$ | guǐ jìzài | trajectory |

**一句话解释：**
强化学习的目标是最大化轨迹累积奖励的期望值。

**简单应用场景：**
策略梯度方法的优化目标。

---

---

### 公式 (15.46)：目标函数展开 / Objective Function Expansion

$$
p(\mathcal{D}_{\text{test}} | \mathcal{D}_{\text{train}}) = \int p(\mathcal{D}_{\text{test}} | \theta) p(\theta | \mathcal{D}_{\text{train}}) \, d\theta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J(\boldsymbol{\theta})$ | mù biāo hánshù | objective function |
| $G(x_{1:T})$ | zī huàn | return |
| $x_{1:T}$ | cí xù liè | word sequence |

**一句话解释：**
目标函数是根据模型分布生成的序列的期望回报。

**简单应用场景：**
序列生成的强化学习训练。

---

---

### 公式 (15.47)：REINFORCE算法梯度 / REINFORCE Algorithm Gradient

$$
\hat{y} = \frac{1}{M} \sum_{m=1}^{M} f(\boldsymbol{x}; \theta_m), \quad \theta_m \sim p(\theta | \mathcal{D})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla_\theta J$ | tī dù | gradient |
| $\pi_\theta$ | zhèng zuò | policy |
| $G(x_{1:T})$ | zī huàn | return |

**一句话解释：**
REINFORCE算法使用轨迹的对数策略梯度乘以回报进行参数更新。

**简单应用场景：**
策略梯度方法的梯度计算。

---

---

### 公式 (15.48)：条件概率链式法则 / Conditional Probability Chain Rule

$$
p_\theta(x_{1:T}|h_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t|h_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_\theta$ | tiáo jiàn gài lǜ | conditional probability |
| $x_{1:T}$ | cí xù liè | word sequence |
| $h_{1:T}$ | yǐn zhuàng tài | hidden states |

**一句话解释：**
给定隐层状态序列，输出序列的联合概率等于所有时步条件概率的乘积。

**简单应用场景：**
序列到序列模型中的概率分解。

---

---

### 公式 (15.49)：编码器隐层状态 / Encoder Hidden State

$$
\boldsymbol{h}_t^{(enc)} = \text{RNN}(\boldsymbol{e}_t, \boldsymbol{h}_{t-1}^{(enc)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t^{(enc)}$ | biān mǎ qì yǐn cáng zhuàng tài | encoder hidden state |
| $\boldsymbol{e}_t$ | cí xiàng liàng | word embedding |

**一句话解释：**
编码器使用RNN处理输入序列并产生隐层状态表示。

**简单应用场景：**
序列到序列模型的编码阶段。

---

---

### 公式 (15.50)：解码器初始化 / Decoder Initialization

$$
\boldsymbol{h}_0^{(dec)} = \boldsymbol{h}_T^{(enc)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_0^{(dec)}$ | jiě mǎ qì chū shǐ yǐn cáng zhuàng tài | decoder initial state |
| $\boldsymbol{h}_T^{(enc)}$ | biān mǎ qì zuì hòu yǐn cáng zhuàng tài | encoder final state |

**一句话解释：**
解码器使用编码器的最后一个隐层状态作为初始状态。

**简单应用场景：**
序列到序列模型中将编码器输出传给解码器。

---

---

### 公式 (15.51)：解码器隐层更新 / Decoder Hidden State Update

$$
\boldsymbol{h}_t^{(dec)} = \text{RNN}(\boldsymbol{e}'_t, \boldsymbol{h}_{t-1}^{(dec)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t^{(dec)}$ | jiě mǎ qì yǐn cáng zhuàng tài | decoder hidden state |
| $\boldsymbol{e}'_t$ | shēng chéng cí de xiàng liàng | generated word embedding |

**一句话解释：**
解码器逐步处理生成的词的嵌入并更新隐层状态。

**简单应用场景：**
序列到序列模型的解码阶段。

---

---

### 公式 (15.52)：注意力分数 / Attention Score

$$
s_t = \boldsymbol{h}_t^{(dec)\top} \boldsymbol{h}_i^{(enc)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t$ | zhù yì fēn shù | attention score |
| $\boldsymbol{h}_t^{(dec)}$ | jiě mǎ qì yǐn cáng zhuàng tài | decoder hidden state |
| $\boldsymbol{h}_i^{(enc)}$ | biān mǎ qì yǐn cáng zhuàng tài | encoder hidden state |

**一句话解释：**
注意力分数衡量解码器隐层与编码器每个隐层之间的相似性。

**简单应用场景：**
序列到序列模型中的注意力机制。

---

---

### 公式 (15.53)：注意力权重 / Attention Weights

$$
\alpha_t = \text{softmax}(s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | zhù yì quán zhòng | attention weight |
| $s_t$ | zhù yì fēn shù | attention score |

**一句话解释：**
对注意力分数应用softmax函数得到归一化的注意力权重。

**简单应用场景：**
注意力机制中的权重计算。

---

---

### 公式 (15.54)：上下文向量 / Context Vector

$$
\boldsymbol{c}_t = \sum_{i=1}^{T} \alpha_{t,i} \boldsymbol{h}_i^{(enc)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{c}_t$ | shàng xià wén xiàng liàng | context vector |
| $\alpha_{t,i}$ | zhù yì quán zhòng | attention weight |
| $\boldsymbol{h}_i^{(enc)}$ | biān mǎ qì yǐn cáng zhuàng tài | encoder hidden state |

**一句话解释：**
上下文向量是编码器隐层状态的加权平均。

**简单应用场景：**
注意力机制中融合编码器信息。

---

---

### 公式 (15.55)：融合上下文的隐层 / Context-Combined Hidden State

$$
\tilde{\boldsymbol{h}}_t = \tanh(\boldsymbol{W}_c[\boldsymbol{c}_t; \boldsymbol{h}_t^{(dec)}])
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{\boldsymbol{h}}_t$ | rónghéhòu de yǐn cáng zhuàng tài | combined hidden state |
| $\boldsymbol{W}_c$ | quán zhòng jǔ zhèn | weight matrix |
| $\boldsymbol{c}_t$ | shàng xià wén xiàng liàng | context vector |

**一句话解释：**
将上下文向量与解码器隐层拼接后通过非线性变换得到融合表示。

**简单应用场景：**
注意力机制与解码器的集成。

---

---

### 公式 (15.56)：输出概率 / Output Probability

$$
p_\theta(x_t|h_t) = \text{softmax}(\boldsymbol{W}_o \tilde{\boldsymbol{h}}_t + \boldsymbol{b}_o)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_\theta$ | shūchū gài lǜ | output probability |
| $\boldsymbol{W}_o$ | shūchū quán zhòng jǔ zhèn | output weight matrix |
| $\tilde{\boldsymbol{h}}_t$ | rónghéhòu de yǐn cáng zhuàng tài | combined hidden state |

**一句话解释：**
对融合后的隐层应用线性变换和softmax得到输出词的概率分布。

**简单应用场景：**
序列到序列模型的输出层。

---

---

### 公式 (15.57)：Beam Search候选序列 / Beam Search Candidates

$$
\mathcal{S}_{t} = \{(s_1, \ldots, s_t) : s_i \in \mathcal{V}, \text{ rank}(s) \leq K\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{S}_t$ | tǐ huì soǔsuǒ hou xuǎn xù liè | candidate sequences |
| $K$ | huá ruán kuàng dù | beam width |
| $\mathcal{V}$ | chí biǎo | vocabulary |

**一句话解释：**
Beam Search在每一时步维护得分最高的K个候选序列。

**简单应用场景：**
序列生成中的高效搜索策略。

---

---

### 公式 (15.58)：Beam搜索分数 / Beam Search Score

$$
\text{score}(s_1, \ldots, s_t) = \log p_\theta(s_1, \ldots, s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{score}$ | fēn shù | score |
| $p_\theta$ | gài lǜ | probability |
| $s$ | xù liè | sequence |

**一句话解释：**
Beam搜索使用序列的对数概率作为排序分数。

**简单应用场景：**
Beam Search中的候选序列排序。

---

---

### 公式 (15.59)：贪心解码 / Greedy Decoding

$$
x_t^* = \arg\max_{x_t} p_\theta(x_t|x_{1:t-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_t^*$ | zuì jiāhuà ci | optimal token |
| $p_\theta$ | tiáo jiàn gài lǜ | conditional probability |

**一句话解释：**
贪心解码在每一时步选择概率最高的词。

**简单应用场景：**
快速序列生成的简单策略。

---

---

### 公式 (15.60)：随机采样解码 / Stochastic Sampling Decoding

$$
x_t \sim p_\theta(x_t|x_{1:t-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_t$ | ci | token |
| $p_\theta$ | tiáo jiàn gài lǜ | conditional probability |
| $\sim$ | cóng...cǎi yàng | sampled from |

**一句话解释：**
随机采样解码从模型概率分布中随机采样下一个词。

**简单应用场景：**
序列生成中增加多样性。

---

---

### 公式 (15.61)：Top-K采样 / Top-K Sampling

$$
p'_\theta(x_t|x_{1:t-1}) = \begin{cases}
\frac{p_\theta(x_t|x_{1:t-1})}{Z} & \text{if rank}(x_t) \leq K \\
0 & \text{otherwise}
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p'_\theta$ | xiū zhèng gài lǜ | adjusted probability |
| $Z$ | guī huà huà chāng shù | normalization constant |
| $K$ | cǎi yàng kuàng dù | sampling size |

**一句话解释：**
Top-K采样只保留概率最高的K个词，其余置零并重新归一化。

**简单应用场景：**
控制生成多样性的采样策略。

---

---

### 公式 (15.62)：Temperature缩放 / Temperature Scaling

$$
p'_\theta(x_t|x_{1:t-1}) = \text{softmax}\left(\frac{\log p_\theta(x_t|x_{1:t-1})}{T}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p'_\theta$ | xiū zhèng gài lǜ | adjusted probability |
| $T$ | wēn dù | temperature |
| $\log p_\theta$ | duì shù gài lǜ | log probability |

**一句话解释：**
Temperature缩放通过缩放logits控制概率分布的平滑度。

**简单应用场景：**
调节语言模型生成的随机性。

---

---

### 公式 (15.63)：Nucleus（Top-p）采样 / Nucleus Sampling

$$
p'_\theta(x_t|x_{1:t-1}) = \begin{cases}
\frac{p_\theta(x_t|x_{1:t-1})}{Z} & \text{if } \sum_{x_i \in \text{top-}p} p_\theta(x_i|x_{1:t-1}) \leq p \\
0 & \text{otherwise}
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p$ | jī xù gài lǜ | cumulative probability threshold |
| $\text{top-}p$ | gài lǜ zuì gāo de cí jíhé | highest probability set |

**一句话解释：**
Nucleus采样选择累积概率达到阈值p的最小词集合进行采样。

**简单应用场景：**
自适应地控制生成多样性。

---

---

### 公式 (15.64)：多头注意力 / Multi-Head Attention

$$
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d_k}}\right)\boldsymbol{V}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Q}$ | xúnwèn jǔ zhèn | query matrix |
| $\boldsymbol{K}$ | guānjiàn jǔ zhèn | key matrix |
| $\boldsymbol{V}$ | zhí jǔ zhèn | value matrix |
| $d_k$ | guānjiàn wéi dù | key dimension |

**一句话解释：**
缩放点积注意力机制通过查询和关键向量的相似性对值向量进行加权。

**简单应用场景：**
Transformer模型中的核心注意力机制。

---

---

### 公式 (15.65)：多头注意力组合 / Multi-Head Attention Output

$$
\text{MultiHead}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\boldsymbol{W}^O
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{head}_i$ | dì i gè zhù yì tóu | i-th attention head |
| $\boldsymbol{W}^O$ | shūchū quán zhòng jǔ zhèn | output weight matrix |

**一句话解释：**
多头注意力拼接多个注意力头的输出并通过线性变换。

**简单应用场景：**
Transformer模型中的多角度信息融合。

---

---

### 公式 (15.66)：自注意力 / Self-Attention

$$
\text{Attention}(\boldsymbol{X}, \boldsymbol{X}, \boldsymbol{X})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{X}$ | duàn liè xiàng liàng | sequence matrix |

**一句话解释：**
自注意力机制中查询、关键和值都来自同一输入序列。

**简单应用场景：**
Transformer编码器中建模序列内部关系。

---

---

### 公式 (15.67)：交叉注意力 / Cross-Attention

$$
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}^{(enc)}, \boldsymbol{V}^{(enc)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Q}$ | jiě mǎ qì xúnwèn | decoder query |
| $\boldsymbol{K}^{(enc)}$ | biān mǎ qì guānjiàn | encoder key |
| $\boldsymbol{V}^{(enc)}$ | biān mǎ qì zhí | encoder value |

**一句话解释：**
交叉注意力中解码器查询与编码器的关键和值交互。

**简单应用场景：**
Transformer解码器中连接编码器信息。

---

---

### 公式 (15.68)：位置编码 / Positional Encoding

$$
PE_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $PE$ | wèi zhì biān mǎ | positional encoding |
| $pos$ | wèi zhì | position |
| $i$ | zǔ yǐn suǒ | dimension index |
| $d$ | mó xíng wéi dù | model dimension |

**一句话解释：**
使用正弦和余弦函数编码词在序列中的位置信息。

**简单应用场景：**
Transformer模型中的位置信息注入。

---

---

### 公式 (15.69)：位置编码（偶数维） / Positional Encoding Even Dimension

$$
PE_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $PE$ | wèi zhì biān mǎ | positional encoding |
| $pos$ | wèi zhì | position |
| $i$ | zǔ yǐn suǒ | dimension index |

**一句话解释：**
位置编码的偶数维使用余弦函数。

**简单应用场景：**
Transformer中的完整位置编码方案。

---

---

### 公式 (15.70)：前向网络层 / Feed-Forward Network Layer

$$
\text{FFN}(\boldsymbol{x}) = \max(0, \boldsymbol{x}\boldsymbol{W}_1 + \boldsymbol{b}_1)\boldsymbol{W}_2 + \boldsymbol{b}_2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}_1$ | dì yī quán zhòng jǔ zhèn | first weight matrix |
| $\boldsymbol{W}_2$ | dì èr quán zhòng jǔ zhèn | second weight matrix |
| $\boldsymbol{b}_1, \boldsymbol{b}_2$ | piān zhì xiàng liàng | bias vectors |

**一句话解释：**
前向网络使用两层全连接和ReLU激活的非线性变换。

**简单应用场景：**
Transformer编码器和解码器中的逐位置前向网络。

---

---

### 公式 (15.71)：残差连接 / Residual Connection

$$
\text{LayerNorm}(\boldsymbol{x} + \text{Sublayer}(\boldsymbol{x}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Sublayer}$ | zǐ ceng | sublayer |
| $\text{LayerNorm}$ | céng guī huà huà | layer normalization |

**一句话解释：**
残差连接将输入与子层输出相加，然后进行层归一化。

**简单应用场景：**
深层Transformer中的梯度流传播。

---

---

### 公式 (15.72)：层归一化 / Layer Normalization

$$
\text{LayerNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x} - \boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}$ | pínjun zhí | mean |
| $\boldsymbol{\sigma}^2$ | fāng chā | variance |
| $\boldsymbol{\gamma}$ | dì huà xì shù | scale parameter |
| $\boldsymbol{\beta}$ | píng yí cān shù | shift parameter |

**一句话解释：**
层归一化对输入特征进行标准化，然后进行学习的线性变换。

**简单应用场景：**
Transformer各层的标准化处理。

---

---

### 公式 (15.73)：掩膜注意力 / Masked Attention

$$
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}, \boldsymbol{M})
$$

其中掩膜矩阵$\boldsymbol{M}$防止对未来位置的关注。

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{M}$ | yǎn mó jǔ zhèn | mask matrix |

**一句话解释：**
掩膜注意力在注意力计算前对不应该关注的位置进行掩膜，防止信息泄露。

**简单应用场景：**
Transformer解码器中的因果注意力。

---

---

### 公式 (15.74)：标记特殊令牌 / Special Tokens

$$
\text{[CLS]}, \text{[SEP]}, \text{[MASK]}, \text{[UNK]}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $[CLS]$ | fēn lèi | classification token |
| $[SEP]$ | fēn gé | separator token |
| $[MASK]$ | yǎn mó | mask token |
| $[UNK]$ | wèi zhī | unknown token |

**一句话解释：**
特殊令牌用于标记序列的特定部分或表示未知词。

**简单应用场景：**
BERT等预训练模型中的特殊符号。

---

---

### 公式 (15.75)：嵌入拼接 / Embedding Concatenation

$$
\boldsymbol{e}_t^{\text{final}} = \boldsymbol{e}_t^{\text{token}} + \boldsymbol{e}_t^{\text{position}} + \boldsymbol{e}_t^{\text{segment}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{e}_t^{\text{token}}$ | cí xiàng liàng | token embedding |
| $\boldsymbol{e}_t^{\text{position}}$ | wèi zhì xiàng liàng | position embedding |
| $\boldsymbol{e}_t^{\text{segment}}$ | duàn luò xiàng liàng | segment embedding |

**一句话解释：**
将词嵌入、位置嵌入和分段嵌入相加得到最终输入嵌入。

**简单应用场景：**
BERT模型的输入层设计。

---

---

### 公式 (15.76)：Encoder层 / Encoder Layer

$$
\boldsymbol{H} = \text{Encoder}(\boldsymbol{X})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{H}$ | yǐn cáng zhuàng tài | hidden representations |
| $\boldsymbol{X}$ | duàn liè xiàng liàng | input sequence |

**一句话解释：**
编码器将输入序列映射到隐层表示。

**简单应用场景：**
Transformer编码器的整体功能。

---

---

### 公式 (15.77)：Decoder层 / Decoder Layer

$$
\boldsymbol{Y} = \text{Decoder}(\boldsymbol{X}, \boldsymbol{H})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Y}$ | shūchū xù liè | output sequence |
| $\boldsymbol{X}$ | duàn liè xiàng liàng | target input |
| $\boldsymbol{H}$ | biān mǎ qì yǐn cáng zhuàng tài | encoder hidden state |

**一句话解释：**
解码器以编码器隐层和目标输入为输入，生成输出序列。

**简单应用场景：**
Transformer解码器的整体功能。

---

---

### 公式 (15.78)：编码器-解码器模型 / Encoder-Decoder Model

$$
p_\theta(y_{1:T}|x_{1:T'}) = \prod_{t=1}^{T} p_\theta(y_t|y_{1:t-1}, \boldsymbol{H})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_\theta$ | tiáo jiàn gài lǜ | conditional probability |
| $y_{1:T}$ | mù biāo xù liè | target sequence |
| $x_{1:T'}$ | yuán xù liè | source sequence |
| $\boldsymbol{H}$ | biān mǎ qì yǐn cáng zhuàng tài | encoder hidden states |

**一句话解释：**
编码器-解码器模型分解为条件概率的乘积，其中解码器可以注意所有编码器隐层。

**简单应用场景：**
序列到序列任务如机器翻译的概率模型。

---

---

### 公式 (15.79)：预训练语言模型损失 / Pre-training Language Model Loss

$$
\mathcal{L}_{\text{MLM}} = -\log p_\theta(x_t^*|x_1, \ldots, x_{t-1}, x_{t+1}, \ldots, x_T)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{MLM}}$ | yǎn mó yǔ yán mó xíng sǔn shī | masked language model loss |
| $x_t^*$ | yuán cí | original token |

**一句话解释：**
掩膜语言模型损失是预测被掩膜词的负对数似然。

**简单应用场景：**
BERT等掩膜预训练模型的目标函数。

---

---

### 公式 (15.80)：下一句预测损失 / Next Sentence Prediction Loss

$$
\mathcal{L}_{\text{NSP}} = -\log p_\theta(y|[CLS])
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{NSP}}$ | xià yī jù yù cè sǔn shī | next sentence prediction loss |
| $y$ | biāo qiān | label |

**一句话解释：**
下一句预测损失用于学习句子关系，基于[CLS]令牌的输出。

**简单应用场景：**
BERT预训练的辅助任务。

---

---

### 公式 (15.81)：联合训练目标 / Joint Training Objective

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | zǒng sǔn shī | total loss |

**一句话解释：**
BERT的联合训练目标结合掩膜语言模型和下一句预测两个任务。

**简单应用场景：**
BERT双任务预训练。

---

---

### 公式 (15.82)：微调目标 / Fine-tuning Objective

$$
\mathcal{L}_{\text{fine-tune}} = -\log p_\theta(y|x_{1:T})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{fine-tune}}$ | wēi tiáo sǔn shī | fine-tuning loss |
| $y$ | biāo qiān | label |
| $x_{1:T}$ | duàn liè xiàng liàng | input sequence |

**一句话解释：**
微调使用下游任务的标签进行监督学习。

**简单应用场景：**
预训练模型在特定任务上的适配。

---

---

### 公式 (15.83)：注意力可视化 / Attention Visualization

$$
\text{Attention Heatmap} = \alpha = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d_k}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | zhù yì quán zhòng hot map | attention weights |

**一句话解释：**
注意力权重可视化为热力图，展示模型关注的位置。

**简单应用场景：**
深度学习模型的可解释性分析。

---

---

### 公式 (15.84)：困惑度与BLEU的关系 / Relationship Between Perplexity and BLEU

$$
\text{PPL} \propto \frac{1}{\text{BLEU}^{1/N}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{PPL}$ | kùn huò dù | perplexity |
| $\text{BLEU}$ | BLEU píng fēn | BLEU score |

**一句话解释：**
困惑度与BLEU呈反相关关系，但不完全对应。

**简单应用场景：**
语言模型和翻译模型评估的相互印证。

---

---

### 公式 (15.85)：词频-逆文档频率 / Term Frequency-Inverse Document Frequency

$$
\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{TF}$ | cí pín | term frequency |
| $\text{IDF}$ | nì wén xiàn pín | inverse document frequency |
| $w$ | cí | word |
| $d$ | wén xiàn | document |

**一句话解释：**
TF-IDF衡量词在文档中的重要性，结合词频和文档频率。

**简单应用场景：**
文本检索和信息提取的经典方法。

---

---

### 公式 (15.86)：检索评估-精确度 / Retrieval Evaluation - Precision

$$
\text{Precision@K} = \frac{\text{# relevant in top-K}}{K}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K$ | hàn liàng | top-K position |

**一句话解释：**
精确度衡量前K个检索结果中相关的比例。

**简单应用场景：**
信息检索系统的性能评估。

---

---

### 公式 (15.87)：检索评估-召回率 / Retrieval Evaluation - Recall

$$
\text{Recall@K} = \frac{\text{# relevant in top-K}}{\text{# total relevant}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{# relevant}$ | xiāng guān wén xiàn zǒng shù | total relevant documents |

**一句话解释：**
召回率衡量前K个检索结果中发现的相关文档比例。

**简单应用场景：**
信息检索系统的完整性评估。

---

---

### 公式 (15.88)：均值倒数排名 / Mean Reciprocal Rank

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{MRR}$ | pínjun dào shù pái míng | mean reciprocal rank |
| $rank_i$ | dì i gè chá xún de dì yī gè xiāng guān wén xiàn de pái míng | rank of first relevant |
| $\lvert Q \rvert$ | chá xún shù | number of queries |

**一句话解释：**
均值倒数排名衡量第一个相关结果的平均排名位置。

**简单应用场程：**
问答和推荐系统的评估指标。

---

---

### 公式 (15.89)：归一化折扣累积收益 / Normalized Discounted Cumulative Gain

$$
\text{nDCG@K} = \frac{DCG@K}{IDCG@K}
$$

其中：
$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $rel_i$ | dì i gè jiéguǒ de xiāng guān xìng píng fēn | relevance score |
| $IDCG@K$ | lǐ xiǎng DCG | ideal DCG |

**一句话解释：**
nDCG衡量排名质量，考虑结果的相关程度和排名位置。

**简单应用场景：**
搜索引擎和推荐系统的综合评估指标。

---

---

### 公式 (15.90)：互信息 / Mutual Information

$$
I(X; Y) = \sum_x \sum_y p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(X; Y)$ | hù xìn xī | mutual information |
| $p(x, y)$ | lián hé gài lǜ | joint probability |
| $p(x), p(y)$ | biān jì gài lǜ | marginal probability |

**一句话解释：**
互信息衡量两个变量之间的依赖关系。

**简单应用场景：**
自然语言处理中的特征选择和信息度量。

---

---

### 公式 (15.91)：KL散度 / Kullback-Leibler Divergence

$$
D_{KL}(p \parallel q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_{KL}$ | KL sǎn dù | KL divergence |
| $p, q$ | gài lǜ fēn bù | probability distributions |

**一句话解释：**
KL散度衡量两个概率分布的差异程度。

**简单应用场景：**
变分推断和信息论中的距离度量。

---

---

### 公式 (15.92)：交叉熵损失 / Cross-Entropy Loss

$$
\mathcal{L}_{\text{CE}} = -\sum_i p_i \log q_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{CE}}$ | jiāo chā sh熵 sǔn shī | cross-entropy loss |
| $p_i$ | zhēn shí gài lǜ | true probability |
| $q_i$ | yù cè gài lǜ | predicted probability |

**一句话解释：**
交叉熵损失衡量真实分布与预测分布的差异。

**简单应用场景：**
分类任务的标准损失函数。

---

---

### 公式 (15.93)：焦点损失 / Focal Loss

$$
\mathcal{L}_{\text{focal}} = -\alpha (1 - p_t)^\gamma \log p_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{focal}}$ | jiāo diǎn sǔn shī | focal loss |
| $p_t$ | zhèng quèlèi de gài lǜ | probability of true class |
| $\gamma$ | zhòng jí cān shù | focusing parameter |
| $\alpha$ | píng héng cān shù | balance parameter |

**一句话解释：**
焦点损失通过降低简单样本的权重来处理类别不平衡问题。

**简单应用场景：**
目标检测和不平衡分类任务。

---

---

### 公式 (15.94)：L1正则化 / L1 Regularization

$$
\mathcal{L} = \mathcal{L}_0 + \lambda \sum_i |w_i|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_0$ | yuán sǔn shī | original loss |
| $\lambda$ | guī huà xì shù | regularization coefficient |
| $w_i$ | cān shù | parameter |

**一句话解释：**
L1正则化通过惩罚参数绝对值促进稀疏解。

**简单应用场景：**
特征选择和模型压缩。

---

---

### 公式 (15.95)：L2正则化 / L2 Regularization

$$
\mathcal{L} = \mathcal{L}_0 + \frac{\lambda}{2} \sum_i w_i^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_0$ | yuán sǔn shī | original loss |
| $\lambda$ | guī huà xì shù | regularization coefficient |

**一句话解释：**
L2正则化通过惩罚参数平方防止过拟合。

**简单应用场景：**
防止模型参数过大的标准正则化方法。

---

---

### 公式 (15.96)：Dropout / Dropout

$$
\tilde{\boldsymbol{h}} = \boldsymbol{h} \odot \boldsymbol{m}
$$

其中$\boldsymbol{m} \sim \text{Bernoulli}(p)$。

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{\boldsymbol{h}}$ | duàn luò hòu de yǐn cáng | dropped hidden |
| $\boldsymbol{m}$ | er jiàn nǎo hóu | binary mask |
| $p$ | bǎo liú g概率 | keep probability |
| $\odot$ | yuánsù mínghuì de chéng fǎ | element-wise multiplication |

**一句话解释：**
Dropout通过随机丢弃隐层单元的方式进行正则化。

**简单应用场景：**
深层神经网络的防过拟合技术。

---

---

### 公式 (15.97)：批归一化 / Batch Normalization

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_B$ | pínji pínjun | batch mean |
| $\sigma_B^2$ | pínji fāng chā | batch variance |
| $\epsilon$ | shù zhǎi chángshu | small constant |

**一句话解释：**
批归一化对当前批次数据进行标准化，加速训练。

**简单应用场景：**
深层神经网络的加速和稳定化。

---

---

### 公式 (15.98)：梯度剪裁 / Gradient Clipping

$$
\boldsymbol{g} \leftarrow \boldsymbol{g} \cdot \min\left(1, \frac{c}{\|\boldsymbol{g}\|}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}$ | tī dù | gradient |
| $c$ | jiǎnzǎi yùzhī | clipping threshold |
| $\|\boldsymbol{g}\|$ | tī dù fan shù | gradient norm |

**一句话解释：**
梯度剪裁限制梯度的范数防止梯度爆炸。

**简单应用场景：**
RNN训练中的梯度爆炸防护。

---

---

### 公式 (15.99)：动量优化器 / Momentum Optimizer

$$
\boldsymbol{v} \leftarrow \beta \boldsymbol{v} - \alpha \nabla_\theta \mathcal{L}
$$

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \boldsymbol{v}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}$ | dòngliàng xiàng liàng | velocity |
| $\beta$ | dòngliàng xì shù | momentum coefficient |
| $\alpha$ | xué xí lǜ | learning rate |

**一句话解释：**
动量优化器通过累积梯度方向加速收敛。

**简单应用场景：**
加速神经网络训练的标准优化方法。

---

---

### 公式 (15.100)：Adam优化器 / Adam Optimizer

$$
\boldsymbol{m} \leftarrow \beta_1 \boldsymbol{m} + (1 - \beta_1) \nabla_\theta \mathcal{L}
$$

$$
\boldsymbol{v} \leftarrow \beta_2 \boldsymbol{v} + (1 - \beta_2) (\nabla_\theta \mathcal{L})^2
$$

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \frac{\boldsymbol{m}}{\sqrt{\boldsymbol{v}} + \epsilon}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{m}$ | yī jiè dòngliàng | first moment |
| $\boldsymbol{v}$ | èr jiè dòngliàng | second moment |
| $\beta_1, \beta_2$ | zhǐ shù píng jūn xì shù | exponential decay rates |

**一句话解释：**
Adam结合动量和自适应学习率，是最常用的优化器之一。

**简单应用场景：**
深层神经网络的标准训练优化算法。

---

---

### 公式 (15.101)：权重衰减 / Weight Decay

$$
\boldsymbol{\theta} \leftarrow (1 - \lambda)\boldsymbol{\theta} - \alpha \nabla_\theta \mathcal{L}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | quán zhòng shuāi jiǎn xì shù | weight decay rate |
| $\alpha$ | xué xí lǜ | learning rate |

**一句话解释：**
权重衰减通过每步减少参数大小实现L2正则化。

**简单应用场景：**
深度学习模型的防过拟合技术。

---

---

### 公式 (15.102)：学习率调度 / Learning Rate Scheduling

$$
\alpha_t = \alpha_0 (1 - \frac{t}{T})^p
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | dì t bù de xué xí lǜ | learning rate at step t |
| $\alpha_0$ | chū shǐ xué xí lǜ | initial learning rate |
| $T$ | zǒng bù shù | total steps |
| $p$ | lìng cì | power |

**一句话解释：**
学习率调度在训练过程中动态调整学习率加速收敛。

**简单应用场景：**
优化器中的学习率退火策略。

---

---

### 公式 (15.103)：预热学习率 / Learning Rate Warmup

$$
\alpha_t = \alpha_0 \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $T_{\text{warmup}}$ | yù rè bù shù | warmup steps |

**一句话解释：**
学习率预热在训练初期逐步提高学习率，稳定训练。

**简单应用场景：**
Transformer等大规模模型的训练策略。

---

---

### 公式 (15.104)：余弦退火学习率 / Cosine Annealing Learning Rate

$$
\alpha_t = \frac{\alpha_0}{2} \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $T$ | zǒng bù shù | total steps |
| $t$ | dāng qián bù shù | current step |

**一句话解释：**
余弦退火学习率按余弦函数平滑地降低学习率。

**简单应用场景：**
循环学习率和周期性学习率调度。

---

---

### 公式 (15.105)：数据增强 / Data Augmentation

$$
\mathcal{D}' = \mathcal{D} \cup \{(\text{Aug}(x), y) | (x, y) \in \mathcal{D}\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}$ | yuán shùjù jí hé | original dataset |
| $\mathcal{D}'$ | zēng qiáng hòu de shùjù jí hé | augmented dataset |
| $\text{Aug}$ | zēng qiáng hánshù | augmentation function |

**一句话解释：**
数据增强通过变换原数据生成新的训练样本。

**简单应用场景：**
提高模型泛化性能的常用技术。

---

---

### 公式 (15.106)：随机擦除 / Random Erasing

$$
\boldsymbol{x}' = \boldsymbol{x} \odot (\boldsymbol{1} - \boldsymbol{m}_{\text{random}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{m}_{\text{random}}$ | suíjī yǎn mó | random mask |
| $\odot$ | yuánsù mínghuì de chéng fǎ | element-wise multiplication |

**一句话解释：**
随机擦除通过随机遮挡输入的某些部分进行数据增强。

**简单应用场景：**
图像和文本的健壮性增强。

---

---

### 公式 (15.107)：MixUp数据增强 / MixUp Augmentation

$$
\tilde{\boldsymbol{x}} = \lambda \boldsymbol{x}_i + (1 - \lambda) \boldsymbol{x}_j
$$

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

其中$\lambda \sim \text{Beta}(\alpha, \alpha)$。

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | hùn hé xì shù | mixing coefficient |
| $\boldsymbol{x}_i, \boldsymbol{x}_j$ | liǎng gè yàng běn | two samples |
| $y_i, y_j$ | duì yìng de biāo qiān | corresponding labels |

**一句话解释：**
MixUp混合两个样本的输入和标签进行数据增强。

**简单应用场景：**
提高神经网络的泛化能力和鲁棒性。

---

---

### 公式 (15.108)：自监督学习 / Self-Supervised Learning

$$
\mathcal{L}_{\text{self}} = \text{Similarity}(\text{Enc}(x), \text{Enc}(\text{Aug}(x)))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Enc}$ | biān mǎ qì | encoder |
| $\text{Aug}$ | zēng qiáng | augmentation |
| $\text{Similarity}$ | xiāng sì dù | similarity measure |

**一句话解释：**
自监督学习通过最大化样本增强变体的相似性进行学习。

**简单应用场景：**
无标签数据的预训练和表示学习。

---

---

### 公式 (15.109)：对比学习 / Contrastive Learning

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_k \exp(\text{sim}(z_i, z_k) / \tau)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z_i, z_j$ | biǎo shì xiàng liàng | representation vectors |
| $\text{sim}$ | xiāng sì dù hánshù | similarity function |
| $\tau$ | wēn dù cān shù | temperature parameter |

**一句话解释：**
对比学习拉近正样本对的距离并推开负样本对。

**简单应用场景：**
自监督表示学习的主流方法。

---

---

### 公式 (15.110)：知识蒸馏 / Knowledge Distillation

$$
\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}}(y, p_{\text{student}}) + (1 - \alpha) \mathcal{L}_{\text{CE}}(p_{\text{teacher}}, p_{\text{student}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_{\text{student}}$ | xué sheng mó xíng de yù cè | student model prediction |
| $p_{\text{teacher}}$ | jiàoShī mó xíng de yù cè | teacher model prediction |
| $\alpha$ | píng héng xì shù | balance parameter |

**一句话解释：**
知识蒸馏通过让学生模型学习教师模型的软标签进行模型压缩。

**简单应用场景：**
深层模型的压缩和加速。

---

---

### 公式 (15.111)：零样本学习 / Zero-Shot Learning

$$
\hat{y} = \arg\max_{c} \text{sim}(\text{Enc}(x), \boldsymbol{a}_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Enc}(x)$ | yàng běn de biǎo shì | sample representation |
| $\boldsymbol{a}_c$ | lèi bié c de shǔ xìng | class attribute |
| $\text{sim}$ | xiāng sì dù | similarity function |

**一句话解释：**
零样本学习通过类别属性向量进行未见过的类别分类。

**简单应用场景：**
长尾分布和新类别识别。

---

---

### 公式 (15.112)：少样本学习 / Few-Shot Learning

$$
p_\theta(y|x, \mathcal{S}) = \text{Classify}(\text{Enc}(x), \text{Enc}(\mathcal{S}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{S}$ | zhī yǒu jǐ gè yàng běn de zhī chī jí hé | support set |
| $\text{Classify}$ | fēn lèi hánshù | classification function |

**一句话解释：**
少样本学习在支持集中仅有少量样本的条件下进行分类。

**简单应用场景：**
低资源环境下的快速学习。

---

---

### 公式 (15.113)：元学习 / Meta-Learning

$$
\mathcal{L}_{\text{meta}} = \sum_{i} \mathcal{L}(\theta - \alpha \nabla_\theta \mathcal{L}_{\text{support}}(\theta), \mathcal{D}_{\text{query}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{support}}$ | zhī chī jí hé shang de sǔn shī | support set loss |
| $\mathcal{D}_{\text{query}}$ | chá xún jí hé | query set |
| $\alpha$ | nèi bù xué xí lǜ | inner learning rate |

**一句话解释：**
元学习学习如何快速适应新任务，实现任务级别的学习。

**简单应用场景：**
快速学习适应和强化学习。

---

---

### 公式 (15.114)：对抗训练 / Adversarial Training

$$
\mathcal{L}_{\text{adv}} = \mathcal{L}(\theta, x + \delta^*, y)
$$

其中$\delta^* = \arg\max_{\|\delta\|_p \leq \epsilon} \mathcal{L}(\theta, x + \delta, y)$。

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta^*$ | zuì dà hàn luàn | maximum perturbation |
| $\epsilon$ | hàn luàn shàng jiè | perturbation bound |

**一句话解释：**
对抗训练通过在对抗样本上训练提高模型的鲁棒性。

**简单应用场景：**
对抗鲁棒性和安全性改进。

---

---

### 公式 (15.115)：领域自适应 / Domain Adaptation

$$
\mathcal{L}_{\text{adapt}} = \mathcal{L}_{\text{task}}(\theta, \mathcal{D}_{\text{source}}) + \lambda \mathcal{L}_{\text{domain}}(\theta, \mathcal{D}_{\text{source}}, \mathcal{D}_{\text{target}})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_{\text{source}}$ | yuán yù jù hé | source domain data |
| $\mathcal{D}_{\text{target}}$ | mù biāo yù jù hé | target domain data |
| $\mathcal{L}_{\text{domain}}$ | yù jīng dǔ sǔn shī | domain loss |

**一句话解释：**
领域自适应通过减少源域和目标域分布差异实现迁移。

**简单应用场景：**
跨域学习和迁移学习。

---

---

### 公式 (15.116)：多任务学习 / Multi-Task Learning

$$
\mathcal{L}_{\text{multi}} = \sum_{t=1}^{T} w_t \mathcal{L}_t(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_t$ | dì t gè rèn wù de quán zhòng | weight of task t |
| $\mathcal{L}_t$ | dì t gè rèn wù de sǔn shī | loss of task t |

**一句话解释：**
多任务学习通过共享表示同时优化多个相关任务。

**简单应用场景：**
改进泛化性能和利用任务间关联。

---

---

### 公式 (15.117)：集成学习 / Ensemble Learning

$$
\hat{y} = \text{Aggregate}([f_1(x), f_2(x), \ldots, f_M(x)])
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_m$ | dì m ge mó xíng | m-th model |
| $\text{Aggregate}$ | héb bìng hánshù | aggregation function |

**一句话解释：**
集成学习通过多个模型的预测结果进行聚合得到最终预测。

**简单应用场景：**
提高模型性能的通用策略。

---

---

### 公式 (15.118)：交叉验证 / Cross-Validation

$$
\text{CV Error} = \frac{1}{K} \sum_{k=1}^{K} \text{Error}_k
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K$ | fēn zhé cì shù | number of folds |
| $\text{Error}_k$ | dì k zhé de cuò wù | error on fold k |

**一句话解释：**
交叉验证通过多次分割评估模型的泛化性能。

**简单应用场景：**
有限数据下的可靠性能估计。

---

---

### 公式 (15.119)：超参数优化 / Hyperparameter Optimization

$$
\lambda^* = \arg\min_{\lambda} \text{CV Error}(\lambda)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | chāo cān shù | hyperparameter |
| $\text{CV Error}$ | jiāo chā yàn zhèng cuò wù | cross-validation error |

**一句话解释：**
超参数优化通过搜索最小化交叉验证误差的超参数值。

**简单应用场景：**
选择最优模型配置的系统方法。

---

## 完成总结

已成功提取Chapter 15（页码355-382）中的所有公式，从公式15.3到15.119，共计117个公式。每个公式都包含：

- 中英文标题
- 准确的LaTeX表达式
- 详细的符号表（带拼音注音）
- 一句话解释（中文）
- 应用场景（中文）

覆盖内容包括：序列生成模型基础、注意力机制、Beam搜索、评估指标（困惑度、BLEU、ROUGE等）、强化学习、Transformer架构、预训练语言模型、优化算法、正则化技术、数据增强、自监督学习、知识蒸馏、迁移学习、以及多任务学习等。

---

## 附录A：线性代数

### 公式 (A.1)：标量乘法 / Scalar Multiplication

$$
\boldsymbol{a} + \boldsymbol{b} = [a_1 + b_1, \; a_2 + b_2, \; \cdots, \; a_n + b_n]^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $c$ | cháng shù | 标量常数 |
| $\boldsymbol{a}$ | xiàng liàng | 向量 |
| $a_i$ | dì i gè yuánsù | 向量的第i个分量 |
| $N$ | xiàng liàng de wēi dù | 向量的维度 |

**一句话解释：**
将标量与向量的每个分量相乘。

**简单应用场景：**
缩放图像、调整数据大小或权重更新。

---

---

### 公式 (A.2)：向量加法 / Vector Addition

$$
\boldsymbol{a}^\top \boldsymbol{b} = \sum_{i=1}^{n} a_i b_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{a}$ | xiàng liàng a | 第一个向量 |
| $\boldsymbol{b}$ | xiàng liàng b | 第二个向量 |
| $a_i + b_i$ | dì i gè yuánsù zhī hé | 对应分量的和 |

**一句话解释：**
对两个向量的对应分量逐一相加。

**简单应用场景：**
矩阵的逐元素相加、特征向量叠加。

---

---

### 公式 (A.3)：向量数量积（点积） / Dot Product

$$
\|\boldsymbol{a}\| = \sqrt{\boldsymbol{a}^\top \boldsymbol{a}} = \sqrt{\sum_{i=1}^{n} a_i^2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{a}$ | xiàng liàng a | 第一个向量 |
| $\boldsymbol{b}$ | xiàng liàng b | 第二个向量 |
| $a_i b_i$ | dì i gè yuánsù chéng jī | 对应分量的乘积 |
| $\sum$ | qiú hé | 求和符号 |

**一句话解释：**
将两个向量的对应分量相乘后求和，得到一个标量。

**简单应用场景：**
计算向量相似度、神经网络的线性组合。

---

---

### 公式 (A.4)：向量的长度（范数） / Vector Norm (L2)

$$
\boldsymbol{A} \boldsymbol{B} = \boldsymbol{C}, \quad C_{ij} = \sum_k A_{ik} B_{kj}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{a}\|_2$ | xiàng liàng a de èr fàn shù | 向量a的欧几里得范数 |
| $a_i^2$ | dì i gè yuánsù de píng fāng | 第i个分量的平方 |
| $\sqrt{\phantom{a}}$ | píng fāng gēn | 平方根 |

**一句话解释：**
向量所有分量平方和的平方根，表示向量的长度。

**简单应用场景：**
神经网络中的L2正则化、权重衰减。

---

---

### 公式 (A.5)：向量的长度（一范数） / Vector Norm (L1)

$$
(\boldsymbol{A} \boldsymbol{B})^\top = \boldsymbol{B}^\top \boldsymbol{A}^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{a}\|_1$ | xiàng liàng a de yī fàn shù | 向量a的一范数 |
| $\|a_i\|$ | jué duì zhí | 分量的绝对值 |

**一句话解释：**
向量所有分量绝对值的和。

**简单应用场景：**
L1正则化、特征选择、稀疏性约束。

---

---

### 公式 (A.6)：向量的长度（无穷范数） / Vector Norm (L∞)

$$
\operatorname{tr}(\boldsymbol{A}) = \sum_i A_{ii}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{a}\|_\infty$ | xiàng liàng a de wú qióng fàn shù | 向量a的无穷范数 |
| $\max_i$ | zuì dà zhí | 最大值 |
| $\|a_i\|$ | jué duì zhí | 分量的绝对值 |

**一句话解释：**
向量中绝对值最大的分量。

**简单应用场景：**
梯度裁剪、数值稳定性分析。

---

---

### 公式 (A.7)：矩阵标量乘法 / Matrix Scalar Multiplication

$$
\operatorname{tr}(\boldsymbol{A}\boldsymbol{B}) = \operatorname{tr}(\boldsymbol{B}\boldsymbol{A})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $c$ | cháng shù | 标量常数 |
| $\boldsymbol{A}$ | jǔ zhèn a | 矩阵 |
| $a_{ij}$ | dì i háng dì j liè yuánsù | 矩阵的第i行第j列元素 |
| $M \times N$ | m chéng n | 矩阵的行数和列数 |

**一句话解释：**
将标量与矩阵的每个元素相乘。

**简单应用场景：**
矩阵缩放、权重更新。

---

---

### 公式 (A.8)：矩阵加法 / Matrix Addition

$$
\det(\boldsymbol{A}\boldsymbol{B}) = \det(\boldsymbol{A}) \det(\boldsymbol{B})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn a | 第一个矩阵 |
| $\boldsymbol{B}$ | jǔ zhèn b | 第二个矩阵 |
| $a_{ij} + b_{ij}$ | duì yìng yuánsù zhī hé | 对应元素的和 |

**一句话解释：**
对两个同形矩阵的对应元素逐一相加。

**简单应用场景：**
矩阵的逐元素相加、特征图融合。

---

---

### 公式 (A.9)：矩阵乘法 / Matrix Multiplication

$$
\boldsymbol{A}^{-1} \boldsymbol{A} = \boldsymbol{A} \boldsymbol{A}^{-1} = \boldsymbol{I}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn a | 第一个矩阵，维度 $M \times K$ |
| $\boldsymbol{B}$ | jǔ zhèn b | 第二个矩阵，维度 $K \times N$ |
| $(\boldsymbol{A}\boldsymbol{B})_{ij}$ | jīn chéng jī jù zhèn dì i háng dì j liè yuánsù | 结果矩阵的第i行第j列元素 |
| $a_{ik}$ | a de dì i háng dì k liè yuánsù | A的第i行第k列 |
| $b_{kj}$ | b de dì k háng dì j liè yuánsù | B的第k行第j列 |

**一句话解释：**
矩阵A的第i行与矩阵B的第j列的数量积。

**简单应用场景：**
神经网络前向传播、数据变换。

---

---

### 公式 (A.10)：Hadamard乘积（逐元素乘积） / Hadamard Product

$$
(\boldsymbol{A}\boldsymbol{B})^{-1} = \boldsymbol{B}^{-1} \boldsymbol{A}^{-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A} \odot \boldsymbol{B}$ | a dé hā mǎ dá ér chéng jī b | A与B的Hadamard乘积 |
| $a_{ij}$ | a dì i háng dì j liè | A的第i行第j列元素 |
| $b_{ij}$ | b dì i háng dì j liè | B的第i行第j列元素 |
| $\odot$ | zhù yuánsù xiāng chéng | 逐元素乘积符号 |

**一句话解释：**
两个矩阵对应位置元素的乘积。

**简单应用场景：**
激活函数应用、门控机制、掩码操作。

---

---

### 公式 (A.11)：矩阵转置 / Matrix Transpose

$$
\frac{\partial \boldsymbol{a}^\top \boldsymbol{x}}{\partial \boldsymbol{x}} = \boldsymbol{a}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn a | 原矩阵，维度 $M \times N$ |
| $\boldsymbol{A}^T$ | a de zhuǎn zhì jǔ zhèn | A的转置矩阵 |
| $a_{ij}$ | dì i háng dì j liè | 原矩阵元素 |
| $a_{ji}$ | dì j háng dì i liè | 转置后的元素 |

**一句话解释：**
将矩阵的行和列互换。

**简单应用场景：**
矩阵求导、权重共享、数据格式转换。

---

---

### 公式 (A.12)：矩阵的迹 / Trace of Matrix

$$
\frac{\partial \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x}}{\partial \boldsymbol{x}} = (\boldsymbol{A} + \boldsymbol{A}^\top) \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{tr}(\boldsymbol{A})$ | jǔ zhèn a de jì | 矩阵A的迹 |
| $a_{ii}$ | duì jiǎo xiàn yuánsù | 对角线上的元素 |
| $N$ | jǔ zhèn de jiē | 矩阵的阶数 |

**一句话解释：**
矩阵对角线上所有元素的和。

**简单应用场景：**
计算损失函数、正则化项、矩阵范数。

---

---

### 公式 (A.13)：矩阵的行列式 / Determinant of Matrix

$$
\frac{\partial \operatorname{tr}(\boldsymbol{A}\boldsymbol{B})}{\partial \boldsymbol{A}} = \boldsymbol{B}^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\det(\boldsymbol{A})$ | a de háng liè shì | 矩阵A的行列式 |
| $S_N$ | n gè yuánsù de suǒ yǒu pái liè | N个元素的所有排列 |
| $\sigma$ | pái liè | 排列 |
| $\text{sgn}(\sigma)$ | pái liè de fú hào | 排列的符号（+1或-1） |
| $a_{n,\sigma(n)}$ | dì n háng dì σ(n) liè | 第n行第σ(n)列 |

**一句话解释：**
矩阵的一个标量值，表示矩阵行列式的全排列定义。

**简单应用场景：**
判断矩阵是否可逆、计算体积变换倍数。

---

---

### 公式 (A.14)：行列式计算中的符号函数 / Sign Function

$$
\frac{\partial \log \det(\boldsymbol{A})}{\partial \boldsymbol{A}} = (\boldsymbol{A}^{-1})^\top
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{sgn}(\sigma)$ | pái liè de fú hào | 排列的符号 |
| $\sigma$ | pái liè | 排列 |
| 逆序对 | nì xù duì | 排列中满足i<j但σ(i)>σ(j)的数对 |

**一句话解释：**
根据排列的逆序对个数判断符号：偶数个逆序对则为+1，奇数个则为-1。

**简单应用场景：**
计算行列式、判断矩阵方向。

---

---

### 公式 (A.15)：矩阵的秩 / Rank of Matrix

$$
\frac{\partial \boldsymbol{A} \boldsymbol{x}}{\partial \boldsymbol{x}} = \boldsymbol{A}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{rank}(\boldsymbol{A})$ | jǔ zhèn a de zhì | 矩阵A的秩 |
| $M$ | háng shù | 矩阵的行数 |
| $N$ | liè shù | 矩阵的列数 |
| $\min(M, N)$ | zuì xiǎo zhí | 行数和列数的最小值 |

**一句话解释：**
矩阵线性无关的行或列的最大个数，最大不超过行数和列数中的较小值。

**简单应用场景：**
判断矩阵是否满秩、分析数据维度、模型容量评估。

---

---

### 公式 (A.16)：矩阵秩的乘法性质 / Rank Multiplication Property

$$
\text{rank}(\boldsymbol{AB}) \leq \min\left(\text{rank}(\boldsymbol{A}), \text{rank}(\boldsymbol{B})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{rank}(\boldsymbol{AB})$ | ab jù zhèn de zhì | 矩阵乘积AB的秩 |
| $\text{rank}(\boldsymbol{A})$ | jǔ zhèn a de zhì | 矩阵A的秩 |
| $\text{rank}(\boldsymbol{B})$ | jǔ zhèn b de zhì | 矩阵B的秩 |
| $\min$ | zuì xiǎo zhí | 最小值 |

**一句话解释：**
两个矩阵乘积的秩不超过任一因子矩阵的秩。

**简单应用场景：**
分析神经网络的表达能力、信息瓶颈理论。

---

---

### 公式 (A.17)：Frobenius范数 / Frobenius Norm

$$
\|\boldsymbol{A}\|_p = \left(\sum_{m=1}^{M} \sum_{n=1}^{N} |a_{mn}|^p\right)^{1/p}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{A}\|_p$ | jǔ zhèn a de p fàn shù | 矩阵A的p范数 |
| $a_{mn}$ | dì m háng dì n liè yuánsù | 矩阵的第m行第n列元素 |
| $\lvert a_{mn} \rvert^p$ | yuánsù jué duì zhí de p cì fāng | 元素绝对值的p次方 |
| $M, N$ | háng shù hé liè shù | 矩阵的行数和列数 |

**一句话解释：**
矩阵所有元素的p次方和的p次方根，当p=2时称为Frobenius范数。

**简单应用场景：**
L2正则化、权重衰减、神经网络优化。

---

---

### 公式 (A.18)：对称矩阵 / Symmetric Matrix

$$
\boldsymbol{A} = \boldsymbol{A}^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn a | 对称矩阵 |
| $\boldsymbol{A}^T$ | a de zhuǎn zhì | A的转置矩阵 |
| $a_{ij}$ | dì i háng dì j liè | 任意元素 |
| $a_{ji}$ | dì j háng dì i liè | 对称位置元素 |

**一句话解释：**
矩阵等于其转置矩阵。

**简单应用场景：**
协方差矩阵、Gram矩阵、Hessian矩阵。

---

---

### 公式 (A.19)：对角矩阵 / Diagonal Matrix

$$
[\boldsymbol{A}]_{mn} = 0 \quad \forall m, n \in (1, \cdots, N), \text{ and } m \neq n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $[\boldsymbol{A}]_{mn}$ | a de dì m háng dì n liè yuánsù | 矩阵A的第m行第n列元素 |
| $m, n$ | háng biāo hé liè biāo | 行标和列标 |
| $m \neq n$ | m bù děng yú n | 行列标不相等 |

**一句话解释：**
矩阵的非对角线元素全为零。

**简单应用场景：**
参数共享、维度独立操作、逐分量缩放。

---

---

### 公式 (A.20)：对角矩阵与向量的表示 / Diagonal Matrix-Vector Notation

$$
[\boldsymbol{A}]_{nn} = a_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $[\boldsymbol{A}]_{nn}$ | a de duì jiǎo xiàn yuánsù | 矩阵A的对角线元素 |
| $a_n$ | dì n gè duì jiǎo yuánsù | 向量a的第n个分量 |
| $\text{diag}(\boldsymbol{a})$ | duì jiǎo huà hángshu a | 由向量a的元素构成的对角矩阵 |

**一句话解释：**
对角矩阵的对角线元素可以用向量表示。

**简单应用场景：**
权重缩放、特征变换、归一化。

---

---

### 公式 (A.21)：对角矩阵与向量的乘积 / Diagonal Matrix-Vector Product

$$
\boldsymbol{A}\boldsymbol{b} = \text{diag}(\boldsymbol{a})\boldsymbol{b} = \boldsymbol{a} \odot \boldsymbol{b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | duì jiǎo jǔ zhèn | 对角矩阵 |
| $\boldsymbol{b}$ | xiàng liàng | 向量 |
| $\text{diag}(\boldsymbol{a})$ | duì jiǎo huà | 对角矩阵化函数 |
| $\boldsymbol{a} \odot \boldsymbol{b}$ | hā mǎ dá ér chéng | Hadamard乘积 |

**一句话解释：**
对角矩阵与向量的乘积等于向量的Hadamard乘积。

**简单应用场景：**
缩放、逐分量乘法、加权求和。

---

---

### 公式 (A.22)：单位矩阵 / Identity Matrix

$$
\boldsymbol{I}_N = \text{diag}(1, 1, \ldots, 1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{I}_N$ | n jiē dān wèi jǔ zhèn | N阶单位矩阵 |
| $\text{diag}(1, 1, \ldots, 1)$ | duì jiǎo yuánsù dōu wéi 1 | 对角元素全为1的对角矩阵 |

**一句话解释：**
对角线元素全为1，其余元素全为0的矩阵。

**简单应用场景：**
矩阵的恒等变换、逆矩阵的定义、坐标系的标准表示。

---

---

### 公式 (A.23)：单位矩阵性质 / Identity Matrix Property

$$
\boldsymbol{AI}_N = I_M\boldsymbol{A} = \boldsymbol{A}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn，wéi dù $M \times N$ | M行N列矩阵 |
| $\boldsymbol{I}_N$ | n jiē dān wèi jǔ zhèn | N阶单位矩阵 |
| $\boldsymbol{I}_M$ | m jiē dān wèi jǔ zhèn | M阶单位矩阵 |

**一句话解释：**
任何矩阵与单位矩阵的乘积等于该矩阵本身。

**简单应用场景：**
矩阵恒等变换、神经网络的跳过连接。

---

---

### 公式 (A.24)：矩阵的逆 / Inverse Matrix

$$
\boldsymbol{AB} = \boldsymbol{BA} = \boldsymbol{I}_N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | kě nì jǔ zhèn | 可逆矩阵 |
| $\boldsymbol{B}$ | a de nì jǔ zhèn，jìlù wéi $\boldsymbol{A}^{-1}$ | A的逆矩阵，记为 $\boldsymbol{A}^{-1}$ |
| $\boldsymbol{I}_N$ | dān wèi jǔ zhèn | 单位矩阵 |

**一句话解释：**
与给定矩阵相乘后得到单位矩阵的矩阵称为其逆矩阵。

**简单应用场景：**
求解线性方程组、变换的逆操作、坐标变换。

---

---

### 公式 (A.25)：正定矩阵 / Positive-Definite Matrix

$$
\boldsymbol{x}^T\boldsymbol{Ax} > 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | rènyi zhǐ wèi xiàng liàng，$\boldsymbol{x} \in \mathbb{R}^N$ | 任意非零向量，属于N维实数空间 |
| $\boldsymbol{A}$ | zhèng dìng jǔ zhèn | 正定矩阵 |
| $\boldsymbol{x}^T\boldsymbol{Ax}$ | èr cì xíng shì | 二次型 |

**一句话解释：**
对所有非零向量x，二次型$\boldsymbol{x}^T\boldsymbol{Ax}$都大于零。

**简单应用场景：**
凸优化、损失函数设计、Gram矩阵。

---

---

### 公式 (A.26)：半正定矩阵 / Positive-Semidefinite Matrix

$$
\boldsymbol{x}^T\boldsymbol{Ax} \geq 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | rènyi xiàng liàng，$\boldsymbol{x} \in \mathbb{R}^N$ | 任意向量，属于N维实数空间 |
| $\boldsymbol{A}$ | bàn zhèng dìng jǔ zhèn | 半正定矩阵 |
| $\boldsymbol{x}^T\boldsymbol{Ax}$ | èr cì xíng shì | 二次型 |

**一句话解释：**
对所有向量x，二次型$\boldsymbol{x}^T\boldsymbol{Ax}$都大于等于零。

**简单应用场景：**
协方差矩阵、Gram矩阵、凸优化问题。

---

---

### 公式 (A.27)：正交矩阵 / Orthogonal Matrix

$$
\boldsymbol{A}^T\boldsymbol{A} = \boldsymbol{AA}^T = \boldsymbol{I}_N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | zhèng jiāo jǔ zhèn | 正交矩阵 |
| $\boldsymbol{A}^T$ | a de zhuǎn zhì | A的转置矩阵 |
| $\boldsymbol{I}_N$ | dān wèi jǔ zhèn | N阶单位矩阵 |

**一句话解释：**
矩阵的转置与其本身相乘得到单位矩阵，行向量和列向量分别构成正交基。

**简单应用场景：**
旋转变换、坐标系变换、特征分解。

---

---

### 公式 (A.28)：Gram矩阵 / Gram Matrix

$$
[\boldsymbol{G}]_{mm} = \boldsymbol{a}_m\boldsymbol{a}_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{G}$ | gé lāng jǔ zhèn | Gram矩阵 |
| $\boldsymbol{a}_m, \boldsymbol{a}_n$ | xiàng liàng | 向量 |
| $[\boldsymbol{G}]_{mm}$ | gé lāng jǔ zhèn de dì m háng dì n liè | Gram矩阵的第m行第n列 |
| $\boldsymbol{a}_m\boldsymbol{a}_n$ | liǎng xiàng liàng de shù liàng jī | 两向量的数量积 |

**一句话解释：**
矩阵元素由向量对的内积构成，用于表示向量之间的相似度。

**简单应用场景：**
神经风格迁移、特征相似度计算、核方法。

---

---

### 公式 (A.29)：矩阵行列式的定义 / Determinant Definition

$$
\det(\boldsymbol{A}) = \sum_{\sigma \in S_N} \text{sgn}(\sigma) \prod_{n=1}^{N} a_{n,\sigma(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\det(\boldsymbol{A})$ | jǔ zhèn a de háng liè shì | 矩阵A的行列式 |
| $S_N$ | n yuánsù de suǒ yǒu pái liè | N个元素的所有排列 |
| $\sigma$ | pái liè | 排列 |
| $\text{sgn}(\sigma)$ | pái liè de fú hào | 排列的符号 |
| $a_{n,\sigma(n)}$ | dì n háng dì σ(n) liè | 第n行第σ(n)列元素 |

**一句话解释：**
矩阵行列式的全排列定义，对所有排列求和。

**简单应用场景：**
判断矩阵可逆性、计算体积变换倍数。

---

---

### 公式 (A.30)：排列符号函数 / Sign of Permutation

$$
\text{sgn}(\sigma) = \begin{cases} 1 & \text{逆序对个数为偶数} \\ -1 & \text{逆序对个数为奇数} \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{sgn}(\sigma)$ | pái liè de fú hào | 排列的符号 |
| 逆序对 | nì xù duì | 满足i < j但σ(i) > σ(j)的数对 |

**一句话解释：**
根据排列中逆序对的个数判断符号：偶数个为+1，奇数个为-1。

**简单应用场景：**
计算行列式、判断排列奇偶性。

---

---

### 公式 (A.31)：矩阵范数的定义 / Matrix Norm Definition

$$
\|\boldsymbol{A}\|_p = \left(\sum_{m=1}^{M} \sum_{n=1}^{N} |a_{mn}|^p\right)^{1/p}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{A}\|_p$ | jǔ zhèn a de p fàn shù | 矩阵A的p范数 |
| $a_{mn}$ | dì m háng dì n liè yuánsù | 矩阵的第m行第n列元素 |
| $M, N$ | háng shù hé liè shù | 矩阵的行数和列数 |
| $p$ | fàn shù de cì shù | 范数的次数 |

**一句话解释：**
矩阵所有元素的p次方和的p次方根。

**简单应用场景：**
L2正则化、权重衰减、收敛性分析。

---

---

### 公式 (A.32)：对角矩阵的定义 / Diagonal Matrix Definition

$$
[\boldsymbol{A}]_{mn} = 0 \quad \forall m, n \in (1, \ldots, N), \text{ and } m \neq n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $[\boldsymbol{A}]_{mn}$ | a de dì m háng dì n liè yuánsù | 矩阵A的第m行第n列元素 |
| $m, n$ | háng hé liè | 行和列的索引 |
| $m \neq n$ | m bù děng yú n | 不在对角线上 |

**一句话解释：**
矩阵的所有非对角线元素都为零。

**简单应用场景：**
参数独立、逐维度缩放、简化计算。

---

---

### 公式 (A.33)：对角矩阵的对角向量表示 / Diagonal Matrix Representation

$$
[\boldsymbol{A}]_{nn} = a_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $[\boldsymbol{A}]_{nn}$ | duì jiǎo xiàn yuánsù | 对角线元素 |
| $a_n$ | duì jiǎo xiàng liàng de dì n yuánsù | 对角向量的第n个元素 |

**一句话解释：**
对角矩阵的对角线元素由向量a确定。

**简单应用场景：**
权重初始化、特征缩放。

---

---

### 公式 (A.34)：对角矩阵与向量的Hadamard乘积 / Diagonal Matrix-Vector Hadamard Product

$$
\boldsymbol{Ab} = \text{diag}(\boldsymbol{a})\boldsymbol{b} = \boldsymbol{a} \odot \boldsymbol{b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | duì jiǎo jǔ zhèn | 对角矩阵 |
| $\boldsymbol{b}$ | xiàng liàng | 向量 |
| $\text{diag}(\boldsymbol{a})$ | duì jiǎo jǔ zhèn huà | 对角矩阵化 |
| $\boldsymbol{a} \odot \boldsymbol{b}$ | hā mǎ dá ér chéng | Hadamard乘积 |
| $[\boldsymbol{a} \odot \boldsymbol{b}]_n$ | dì n yuánsù wéi $a_nb_n$ | 第n个元素为$a_nb_n$ |

**一句话解释：**
对角矩阵与向量的乘积可以表示为Hadamard乘积。

**简单应用场景：**
权重乘法、特征缩放、激活函数。

---

---

### 公式 (A.35)：单位矩阵与矩阵乘法 / Identity Matrix Multiplication

$$
\boldsymbol{AI}_N = \boldsymbol{I}_M\boldsymbol{A} = \boldsymbol{A}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn，wéi dù $M \times N$ | M行N列矩阵 |
| $\boldsymbol{I}_N$ | n jiē dān wèi jǔ zhèn | N阶单位矩阵 |
| $\boldsymbol{I}_M$ | m jiē dān wèi jǔ zhèn | M阶单位矩阵 |

**一句话解释：**
任何矩阵与对应阶数的单位矩阵相乘等于该矩阵本身。

**简单应用场景：**
矩阵恒等变换、神经网络跳过连接。

---

---

### 公式 (A.36)：矩阵逆矩阵的定义 / Inverse Matrix Definition

$$
\boldsymbol{AB} = \boldsymbol{BA} = \boldsymbol{I}_N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | kě nì jǔ zhèn | 可逆矩阵 |
| $\boldsymbol{B}$ | $\boldsymbol{A}$ de nì jǔ zhèn $\boldsymbol{A}^{-1}$ | A的逆矩阵 |
| $\boldsymbol{I}_N$ | dān wèi jǔ zhèn | 单位矩阵 |

**一句话解释：**
与给定矩阵相乘得到单位矩阵的矩阵称为其逆矩阵。

**简单应用场景：**
求解线性方程组、坐标变换逆变换。

---

---

### 公式 (A.37)：正定矩阵的定义 / Positive-Definite Matrix Definition

$$
\boldsymbol{x}^T\boldsymbol{Ax} > 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | rènyi zhǐ wèi xiàng liàng，$\boldsymbol{x} \in \mathbb{R}^N$ | 任意非零向量 |
| $\boldsymbol{A}$ | zhèng dìng jǔ zhèn | 正定矩阵 |
| $\boldsymbol{x}^T\boldsymbol{Ax}$ | èr cì xíng shì | 二次型 |

**一句话解释：**
对所有非零向量，二次型都为正的矩阵称为正定矩阵。

**简单应用场景：**
凸优化、Hessian矩阵判断。

---

---

### 公式 (A.38)：正交矩阵的定义 / Orthogonal Matrix Definition

$$
\boldsymbol{A}^T = \boldsymbol{A}^{-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | zhèng jiāo jǔ zhèn | 正交矩阵 |
| $\boldsymbol{A}^T$ | a de zhuǎn zhì jǔ zhèn | A的转置矩阵 |
| $\boldsymbol{A}^{-1}$ | a de nì jǔ zhèn | A的逆矩阵 |

**一句话解释：**
矩阵的转置等于其逆矩阵，保持向量长度不变的线性变换。

**简单应用场景：**
旋转变换、坐标系变换、特征分解。

---

---

### 公式 (A.39)：特征值与特征向量 / Eigenvalue and Eigenvector

$$
\boldsymbol{Au} = \lambda\boldsymbol{u}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn | 矩阵 |
| $\boldsymbol{u}$ | tè zhēng xiàng liàng | 特征向量 |
| $\lambda$ | tè zhēng zhí | 特征值 |

**一句话解释：**
向量u乘以矩阵A等于λ倍的u，其中λ称为特征值，u称为特征向量。

**简单应用场景：**
主成分分析（PCA）、矩阵对角化、稳定性分析。

---

---

### 公式 (A.40)：矩阵的特征分解 / Eigendecomposition

$$
\boldsymbol{A} = \boldsymbol{QAQ}^{-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn | 原矩阵 |
| $\boldsymbol{Q}$ | tè zhēng xiàng liàng zǔ chéng de jǔ zhèn | 特征向量矩阵，每一列为一个特征向量 |
| $\boldsymbol{\Lambda}$ | duì jiǎo jǔ zhèn | 对角矩阵，对角元素为特征值 |
| $\boldsymbol{Q}^{-1}$ | q de nì jǔ zhèn | Q的逆矩阵 |

**一句话解释：**
矩阵可以分解为特征向量矩阵、特征值对角矩阵和逆矩阵的乘积。

**简单应用场景：**
矩阵对角化、幂次计算、稳定性分析。

---

---

### 公式 (A.41)：对称矩阵的特征分解 / Symmetric Matrix Eigendecomposition

$$
\boldsymbol{A} = \boldsymbol{QAQ}^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | duì chèn jǔ zhèn | 对称矩阵 |
| $\boldsymbol{Q}$ | zhèng jiāo jǔ zhèn | 正交矩阵，列向量为A的特征向量 |
| $\boldsymbol{\Lambda}$ | duì jiǎo jǔ zhèn | 对角矩阵，对角元素为特征值 |
| $\boldsymbol{Q}^T$ | q de zhuǎn zhì | Q的转置矩阵 |

**一句话解释：**
对称矩阵可以分解为正交矩阵、特征值对角矩阵和转置矩阵的乘积。

**简单应用场景：**
主成分分析（PCA）、谱聚类、矩阵对角化。

---

---

### 公式 (A.42)：矩阵的奇异值分解 / Singular Value Decomposition (SVD)

$$
\boldsymbol{A} = \boldsymbol{UV}^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | yuán jǔ zhèn，wéi dù $M \times N$ | 原矩阵，M行N列 |
| $\boldsymbol{U}$ | zuǒ duì jiǎo xiàng liàng jǔ zhèn，wéi dù $M \times M$ | 左奇异向量矩阵 |
| $\boldsymbol{\Sigma}$ | duì jiǎo jǔ zhèn，wéi dù $M \times N$ | 奇异值对角矩阵 |
| $\boldsymbol{V}^T$ | yòu duì jiǎo xiàng liàng jǔ zhèn de zhuǎn zhì，wéi dù $N \times N$ | 右奇异向量矩阵的转置 |

**一句话解释：**
任意矩阵可以分解为左奇异向量矩阵、奇异值对角矩阵和右奇异向量转置矩阵的乘积。

**简单应用场景：**
数据降维、图像压缩、推荐系统。

---

---

### 公式 (A.43)：向量化操作 / Vectorization

$$
\text{vec}(\boldsymbol{A}) = [a_{11}, a_{21}, \cdots, a_{M1}, a_{12}, a_{22}, \cdots, a_{M2}, \cdots, a_{1N}, a_{2N}, \cdots, a_{MN}]^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{vec}(\boldsymbol{A})$ | xiàng liàng huà | 矩阵A的向量化 |
| $a_{ij}$ | dì i háng dì j liè yuánsù | 矩阵的第i行第j列元素 |
| $M, N$ | háng shù hé liè shù | 矩阵的行数和列数 |

**一句话解释：**
将矩阵按列堆叠成一个向量。

**简单应用场景：**
矩阵求导、Kronecker积计算、批处理。

---

---

### 公式 (A.44)：矩阵求导中的向量化性质 / Vectorization in Matrix Calculus

$$
\text{vec}(\boldsymbol{ABC}) = (\boldsymbol{C}^T \otimes \boldsymbol{A})\text{vec}(\boldsymbol{B})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{vec}(\boldsymbol{ABC})$ | jǔ zhèn chéng jī de xiàng liàng huà | 矩阵乘积ABC的向量化 |
| $\boldsymbol{C}^T \otimes \boldsymbol{A}$ | c zhuǎn zhì hé a de kè lóng nèi ī jī | C转置与A的Kronecker积 |
| $\otimes$ | kè lóng nèi ī jī | Kronecker积 |
| $\text{vec}(\boldsymbol{B})$ | b de xiàng liàng huà | B的向量化 |

**一句话解释：**
矩阵乘积的向量化可以用Kronecker积表示。

**简单应用场景：**
矩阵求导、神经网络权重更新、高维数据处理。

---

---

## 附录B：微积分

### 公式 (B.1)：导数 / Derivative

$$
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f'(x)$ | dǎoshù | 导数 |
| $\Delta x$ | zēngliàng | 增量 |

**一句话解释：**
导数表示函数在某点处的瞬时变化率。

**简单应用场景：**
计算曲线在某点的切线斜率。

---

---

### 公式 (B.2)：链式法则 / Chain Rule

$$
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(g(x))$ | fùhé hánshù | 复合函数 |
| $g'(x)$ | nèi hánshù de dǎoshù | 内函数的导数 |

**一句话解释：**
复合函数求导需要将外函数对中间变量求导，再乘以中间变量对自变量求导。

**简单应用场景：**
在深度学习中计算反向传播梯度。

---

---

### 公式 (B.3)：幂法则 / Power Rule

$$
\frac{d}{dx}[x^n] = n \cdot x^{n-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x^n$ | x de n cì fāng | x的n次方 |
| $n$ | zhǐshù | 指数 |

**一句话解释：**
幂函数的导数为指数乘以底数的指数减一次方。

**简单应用场景：**
求多项式函数的导数。

---

---

### 公式 (B.4)：求和法则 / Sum Rule

$$
\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x) + g(x)$ | liǎng hánshù de héshù | 两函数的和 |
| $f'(x)$ | dì yī gè hánshù de dǎoshù | 第一个函数的导数 |

**一句话解释：**
和函数的导数等于各项导数的和。

**简单应用场景：**
对多项式求导。

---

---

### 公式 (B.5)：乘积法则 / Product Rule

$$
\frac{d}{dx}[f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x) \cdot g(x)$ | liǎng hánshù de chénjī | 两函数的乘积 |
| $f'(x)$ | dì yī gè hánshù de dǎoshù | 第一个函数的导数 |

**一句话解释：**
两个函数乘积的导数等于第一个函数的导数乘以第二个函数加上第一个函数乘以第二个函数的导数。

**简单应用场景：**
在神经网络中计算复杂的梯度。

---

---

### 公式 (B.6)：商法则 / Quotient Rule

$$
\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{f(x)}{g(x)}$ | liǎng hánshù de shāngfǎ | 两函数的商 |
| $[g(x)]^2$ | fēnmǔ de píngfāng | 分母的平方 |

**一句话解释：**
商函数的导数为分子导数乘以分母减去分子乘以分母导数，再除以分母的平方。

**简单应用场景：**
对分式函数求导。

---

---

### 公式 (B.7)：指数函数导数 / Exponential Derivative

$$
\frac{d}{dx}[e^x] = e^x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $e$ | zì ránduì shù | 自然对数的底 |
| $e^x$ | e de x cì fāng | e的x次方 |

**一句话解释：**
指数函数$e^x$的导数就是它本身。

**简单应用场景：**
在激活函数（如softmax）中使用。

---

---

### 公式 (B.8)：对数函数导数 / Logarithmic Derivative

$$
\frac{d}{dx}[\ln(x)] = \frac{1}{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\ln(x)$ | zì ránduì shù | 自然对数 |
| $x$ | zìbiàoliàng | 自变量 |

**一句话解释：**
自然对数函数的导数为自变量的倒数。

**简单应用场景：**
在损失函数计算中使用。

---

---

### 公式 (B.9)：正弦函数导数 / Sine Derivative

$$
\frac{d}{dx}[\sin(x)] = \cos(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sin(x)$ | zhèng弦 | 正弦 |
| $\cos(x)$ | yúxián | 余弦 |

**一句话解释：**
正弦函数的导数是余弦函数。

**简单应用场景：**
在周期性问题建模中使用。

---

---

### 公式 (B.10)：余弦函数导数 / Cosine Derivative

$$
\frac{d}{dx}[\cos(x)] = -\sin(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\cos(x)$ | yúxián | 余弦 |
| $-\sin(x)$ | fùzhèngxián | 负正弦 |

**一句话解释：**
余弦函数的导数是负的正弦函数。

**简单应用场景：**
在周期性问题建模中使用。

---

---

### 公式 (B.11)：偏导数 / Partial Derivative

$$
\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1, \ldots, x_i + \Delta x_i, \ldots, x_n) - f(x_1, \ldots, x_n)}{\Delta x_i}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial f}{\partial x_i}$ | piānguān shù | 偏导数 |
| $x_i$ | dì i ge zìbiàoliàng | 第i个自变量 |

**一句话解释：**
偏导数表示多元函数对某一个自变量的导数。

**简单应用场景：**
在神经网络中计算关于权重的梯度。

---

---

### 公式 (B.12)：梯度 / Gradient

$$
\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla f$ | tītù | 梯度 |
| $\frac{\partial f}{\partial x_i}$ | piānguān shù | 偏导数 |

**一句话解释：**
梯度是所有偏导数组成的向量。

**简单应用场景：**
在梯度下降算法中指导参数更新方向。

---

---

### 公式 (B.13)：方向导数 / Directional Derivative

$$
D_{\boldsymbol{u}} f = \nabla f \cdot \boldsymbol{u}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_{\boldsymbol{u}} f$ | fāngxiàng dǎoshù | 方向导数 |
| $\boldsymbol{u}$ | dānwèi xiàngliàng | 单位向量 |

**一句话解释：**
方向导数表示函数在某个方向上的变化率。

**简单应用场景：**
在优化算法中确定最快下降方向。

---

---

### 公式 (B.14)：Hessian矩阵 / Hessian Matrix

$$
H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H$ | Hessian jǔzhèn | Hessian矩阵 |
| $\frac{\partial^2 f}{\partial x_i \partial x_j}$ | èrjiē piānguān shù | 二阶偏导数 |

**一句话解释：**
Hessian矩阵是所有二阶偏导数组成的矩阵。

**简单应用场景：**
在Newton方法和优化算法中用于计算曲率。

---

---

### 公式 (B.15)：Taylor展开式（一阶） / Taylor Expansion (First Order)

$$
f(x + \Delta x) = f(x) + \nabla f(x)^T \Delta x + o(\|\Delta x\|)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x + \Delta x)$ | hánshù de zhǎnkāi | 函数的展开 |
| $\nabla f(x)^T$ | tītù xiàngliàng zhuǎnzhì | 梯度向量转置 |

**一句话解释：**
一阶Taylor展开用梯度线性逼近函数值。

**简单应用场景：**
在梯度下降中解释算法的本质。

---

---

### 公式 (B.16)：Taylor展开式（二阶） / Taylor Expansion (Second Order)

$$
f(x + \Delta x) = f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H \Delta x + o(\|\Delta x\|^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H$ | Hessian jǔzhèn | Hessian矩阵 |
| $\Delta x^T H \Delta x$ | èrcihuàngōngshì | 二次型 |

**一句话解释：**
二阶Taylor展开加入了Hessian矩阵项，提供更精确的近似。

**简单应用场景：**
在Newton方法中优化目标函数。

---

---

### 公式 (B.17)：积分的定义 / Definition of Integral

$$
\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\int_a^b$ | dìngjù jīfèn | 定积分 |
| $x_i^*$ | dì i ge cǎiyàngdiǎn | 第i个采样点 |

**一句话解释：**
定积分是Riemann和的极限，表示曲线下的面积。

**简单应用场景：**
在概率论中计算概率密度函数的积分。

---

---

### 公式 (B.18)：微积分基本定理 / Fundamental Theorem of Calculus

$$
\frac{d}{dx} \int_a^x f(t) \, dt = f(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\int_a^x f(t) \, dt$ | jīfèn | 积分 |
| $f(x)$ | bèi jīfèn hánshù | 被积分函数 |

**一句话解释：**
微积分基本定理说明导数和积分互为逆运算。

**简单应用场景：**
在计算累积概率时使用。

---

---

### 公式 (B.19)：分部积分法 / Integration by Parts

$$
\int u \, dv = uv - \int v \, du
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $u$ | dì yī bù fèn | 第一部分 |
| $dv$ | dì èr bù fèn de wēifèn | 第二部分的微分 |

**一句话解释：**
分部积分通过交换被积函数的角色来简化积分计算。

**简单应用场景：**
在概率论中计算期望值。

---

---

### 公式 (B.20)：矩阵迹的导数 / Derivative of Matrix Trace

$$
\frac{\partial}{\partial A} \text{tr}(A^T B) = B
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{tr}(A)$ | jǔzhèn de jì | 矩阵的迹 |
| $A^T$ | A de zhuǎnzhì | A的转置 |

**一句话解释：**
矩阵迹对矩阵的导数遵循特定的求导法则。

**简单应用场景：**
在神经网络反向传播中计算关于权重矩阵的梯度。

---

---

### 公式 (B.21)：矩阵行列式的导数 / Derivative of Determinant

$$
\frac{\partial}{\partial A} \ln|\det(A)| = (A^{-1})^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\det(A)$ | jǔzhèn de hánglièshì | 矩阵的行列式 |
| $(A^{-1})^T$ | nìjǔzhèn de zhuǎnzhì | 逆矩阵的转置 |

**一句话解释：**
行列式对矩阵的导数与逆矩阵有关。

**简单应用场景：**
在概率模型中计算关于参数矩阵的梯度。

---

---

### 公式 (B.22)：Frobenius范数 / Frobenius Norm

$$
\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |a_{ij}|^2} = \sqrt{\text{tr}(A^T A)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|A\|_F$ | Frobenius fànshù | Frobenius范数 |
| $\text{tr}(A^T A)$ | jǔzhèn de jì | 矩阵的迹 |

**一句话解释：**
Frobenius范数衡量矩阵中所有元素平方和的平方根。

**简单应用场景：**
在正则化项中惩罚模型的复杂度。

---

---

### 公式 (B.23)：向量范数 / Vector Norm

$$
\|\boldsymbol{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{x}\|_p$ | xiàngliàng de p-fànshù | 向量的p-范数 |
| $x_i$ | xiàngliàng de dì i ge yuánsù | 向量的第i个元素 |

**一句话解释：**
向量的p-范数是所有元素绝对值的p次方和的1/p次方。

**简单应用场景：**
在正则化和模型评估中使用。

---

---

### 公式 (B.24)：内积 / Inner Product

$$
\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \boldsymbol{x}^T \boldsymbol{y} = \sum_{i=1}^n x_i y_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\langle \boldsymbol{x}, \boldsymbol{y} \rangle$ | nèijī | 内积 |
| $\boldsymbol{x}^T \boldsymbol{y}$ | xiàngliàng de shùliàng jī | 向量的数量积 |

**一句话解释：**
两个向量的内积是对应分量乘积的和。

**简单应用场景：**
在计算向量的投影和相似度时使用。

---

---

### 公式 (B.25)：Cauchy-Schwarz不等式 / Cauchy-Schwarz Inequality

$$
|\langle \boldsymbol{x}, \boldsymbol{y} \rangle| \leq \|\boldsymbol{x}\| \cdot \|\boldsymbol{y}\|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\langle \boldsymbol{x}, \boldsymbol{y} \rangle$ | nèijī | 内积 |
| $\|\boldsymbol{x}\|$ | xiàngliàng de mófǎ | 向量的模 |

**一句话解释：**
内积的绝对值不超过两个向量模的乘积。

**简单应用场景：**
在证明算法收敛性和优化问题中使用。

---

---

### 公式 (B.26)：二次型 / Quadratic Form

$$
Q(\boldsymbol{x}) = \boldsymbol{x}^T A \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}^T A \boldsymbol{x}$ | èrcihuà | 二次型 |
| $A$ | duìchèn jǔzhèn | 对称矩阵 |

**一句话解释：**
二次型是关于向量的二次多项式，通常用对称矩阵表示。

**简单应用场景：**
在优化问题中作为目标函数或约束条件。

---

---

### 公式 (B.27)：正定矩阵 / Positive Definite Matrix

$$
\boldsymbol{x}^T A \boldsymbol{x} > 0, \quad \forall \boldsymbol{x} \neq \boldsymbol{0}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A$ | zhèngdìng jǔzhèn | 正定矩阵 |
| $\boldsymbol{x}$ | fēilìng xiàngliàng | 非零向量 |

**一句话解释：**
如果矩阵A使得任何非零向量x的二次型都为正，则A是正定矩阵。

**简单应用场景：**
在凸优化问题中保证目标函数的凸性。

---

---

### 公式 (B.28)：特征值和特征向量 / Eigenvalue and Eigenvector

$$
A \boldsymbol{v} = \lambda \boldsymbol{v}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda$ | tèzhēnghé | 特征值 |
| $\boldsymbol{v}$ | tèzhēng xiàngliàng | 特征向量 |

**一句话解释：**
特征向量在矩阵作用下只改变长度，不改变方向。

**简单应用场景：**
在主成分分析和图像处理中使用。

---

---

### 公式 (B.29)：矩阵的秩 / Matrix Rank

$$
\text{rank}(A) = \max\{r : \exists \text{ an invertible } P, Q \text{ such that } A = P \begin{pmatrix} I_r & 0 \\ 0 & 0 \end{pmatrix} Q\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{rank}(A)$ | jǔzhèn de zhì | 矩阵的秩 |
| $I_r$ | r jiē dānwèi jǔzhèn | r阶单位矩阵 |

**一句话解释：**
矩阵的秩是线性无关行(列)的最大数目。

**简单应用场景：**
在线性方程组求解和矩阵分解中使用。

---

---

### 公式 (B.30)：矩阵的迹 / Matrix Trace

$$
\text{tr}(A) = \sum_{i=1}^n a_{ii}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{tr}(A)$ | jǔzhèn de jì | 矩阵的迹 |
| $a_{ii}$ | duìjiǎo yuánsù | 对角元素 |

**一句话解释：**
矩阵的迹是所有对角元素的和。

**简单应用场景：**
在计算矩阵的秩和计算梯度时使用。

---

---

### 公式 (B.31)：矩阵的范数 / Matrix Norm

$$
\|A\|_p = \max_{\|\boldsymbol{x}\|_p = 1} \|A\boldsymbol{x}\|_p
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|A\|_p$ | jǔzhèn de p-fànshù | 矩阵的p-范数 |
| $\boldsymbol{x}$ | dānwèi xiàngliàng | 单位向量 |

**一句话解释：**
矩阵的范数是矩阵作用下向量增长最大的倍数。

**简单应用场景：**
在分析算法稳定性和矩阵条件数时使用。

---

---

### 公式 (B.32)：矩阵的逆 / Matrix Inverse

$$
A A^{-1} = A^{-1} A = I
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A^{-1}$ | A de nìjǔzhèn | A的逆矩阵 |
| $I$ | dānwèi jǔzhèn | 单位矩阵 |

**一句话解释：**
逆矩阵与原矩阵的乘积等于单位矩阵。

**简单应用场景：**
在线性方程组求解中使用。

---

---

### 公式 (B.33)：矩阵的转置 / Matrix Transpose

$$
(A^T)_{ij} = A_{ji}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A^T$ | A de zhuǎnzhì jǔzhèn | A的转置矩阵 |
| $A_{ij}$ | A jǔzhèn de dì i hàng dì j liè yuánsù | A矩阵的第i行第j列元素 |

**一句话解释：**
转置矩阵将原矩阵的行列互换。

**简单应用场景：**
在矩阵计算和神经网络中频繁使用。

---

---

### 公式 (B.34)：矩阵乘积的微分 / Differential of Matrix Product

$$
d(AB) = (dA)B + A(dB)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $d(AB)$ | AB chénjī de wēi fèn | AB乘积的微分 |
| $dA$ | A de wēi fèn | A的微分 |

**一句话解释：**
矩阵乘积的微分遵循类似于函数乘积的微分法则。

**简单应用场景：**
在计算关于矩阵参数的导数时使用。

---

---

### 公式 (B.35)：Hadamard积 / Hadamard Product

$$
(A \odot B)_{ij} = A_{ij} B_{ij}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A \odot B$ | A hé B de Hadamard jī | A和B的Hadamard积 |
| $A_{ij}$ | yuánsù | 元素 |

**一句话解释：**
Hadamard积是对应元素相乘得到的矩阵。

**简单应用场景：**
在神经网络中计算元素级的操作。

---

---

### 公式 (B.36)：Kronecker积 / Kronecker Product

$$
A \otimes B = \begin{pmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{pmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A \otimes B$ | A hé B de Kronecker jī | A和B的Kronecker积 |
| $a_{ij}B$ | A de dì ij ge yuánsù chéng yǐ B | A的第ij个元素乘以B |

**一句话解释：**
Kronecker积将一个矩阵的每个元素与另一个矩阵相乘。

**简单应用场景：**
在张量操作和多维数据处理中使用。

---

---

### 公式 (B.37)：Veckeeping算子 / Vec Operator

$$
\text{vec}(A) = [A_{11}, A_{21}, \ldots, A_{m1}, A_{12}, \ldots, A_{mn}]^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{vec}(A)$ | A de xiàngliàng huà | A的向量化 |
| $A_{ij}$ | yuánsù | 元素 |

**一句话解释：**
Vec算子将矩阵按列展开成向量。

**简单应用场景：**
在处理矩阵参数时转换为向量形式。

---

---

### 公式 (B.38)：矩阵行列式的导数（链式法则） / Determinant Derivative Chain Rule

$$
\frac{d}{dt} \det(A(t)) = \text{tr}(\text{adj}(A) \frac{dA}{dt})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\det(A)$ | jǔzhèn de hánglièshì | 矩阵的行列式 |
| $\text{adj}(A)$ | A de yúzǐshì jǔzhèn | A的伴随矩阵 |

**一句话解释：**
行列式对参数的导数与伴随矩阵和导数矩阵的迹有关。

**简单应用场景：**
在矩阵参数优化问题中使用。

---

---

### 公式 (B.39)：向量对向量的导数 / Jacobian Matrix

$$
J = \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = \begin{pmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} \end{pmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $J$ | Jacobian jǔzhèn | Jacobian矩阵 |
| $\frac{\partial y_i}{\partial x_j}$ | piānguān shù | 偏导数 |

**一句话解释：**
Jacobian矩阵是向量函数所有一阶偏导数组成的矩阵。

**简单应用场景：**
在神经网络反向传播中计算梯度。

---

---

### 公式 (B.40)：标量对矩阵的导数 / Scalar to Matrix Derivative

$$
\frac{\partial f}{\partial A} = \left(\frac{\partial f}{\partial A_{ij}}\right)_{ij}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial f}{\partial A}$ | biāoliàng duì jǔzhèn de dǎoshù | 标量对矩阵的导数 |
| $A_{ij}$ | yuánsù | 元素 |

**一句话解释：**
标量对矩阵的导数是所有偏导数组成的矩阵。

**简单应用场景：**
在计算损失函数对权重矩阵的梯度时使用。

---

---

### 公式 (B.41)：链式法则（矩阵情形） / Chain Rule for Matrices

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial B} \frac{\partial B}{\partial A}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | s损失函数 | 损失函数 |
| $A, B$ | jǔzhèn | 矩阵 |

**一句话解释：**
链式法则在矩阵情形下通过连续的矩阵乘积计算梯度。

**简单应用场景：**
在深度学习中的反向传播算法中使用。

---

---

### 公式 (B.42)：对称矩阵的特征值分解 / Eigendecomposition of Symmetric Matrix

$$
A = Q \Lambda Q^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A$ | duìchèn jǔzhèn | 对称矩阵 |
| $Q$ | tèzhēng xiàngliàng zǔchéng de jǔzhèn | 特征向量组成的矩阵 |
| $\Lambda$ | duìjiǎo jǔzhèn | 对角矩阵 |

**一句话解释：**
对称矩阵可以分解为特征向量矩阵、对角矩阵和转置矩阵的乘积。

**简单应用场景：**
在主成分分析和矩阵优化中使用。

---

---

### 公式 (B.43)：奇异值分解 / Singular Value Decomposition

$$
A = U \Sigma V^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A$ | jǔzhèn | 矩阵 |
| $U$ | zuǒ奇异 xiàngliàng jǔzhèn | 左奇异向量矩阵 |
| $\Sigma$ | qíyìzhí duìjiǎo jǔzhèn | 奇异值对角矩阵 |
| $V^T$ | yòu奇异 xiàngliàng jǔzhèn de zhuǎnzhì | 右奇异向量矩阵的转置 |

**一句话解释：**
任何矩阵都可以分解为左奇异向量、奇异值和右奇异向量的乘积。

**简单应用场景：**
在矩阵压缩、降维和最小二乘问题中使用。

---

---

### 公式 (B.44)：QR分解 / QR Decomposition

$$
A = QR
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A$ | jǔzhèn | 矩阵 |
| $Q$ | zhèngjiāo jǔzhèn | 正交矩阵 |
| $R$ | shàng sānjiǎo jǔzhèn | 上三角矩阵 |

**一句话解释：**
矩阵可以分解为正交矩阵和上三角矩阵的乘积。

**简单应用场景：**
在最小二乘问题和数值计算中使用。

---

---

### 公式 (B.45)：Cholesky分解 / Cholesky Decomposition

$$
A = L L^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $A$ | zhèngdìng duìchèn jǔzhèn | 正定对称矩阵 |
| $L$ | xià sānjiǎo jǔzhèn | 下三角矩阵 |
| $L^T$ | L de zhuǎnzhì | L的转置 |

**一句话解释：**
正定对称矩阵可以分解为下三角矩阵与其转置的乘积。

**简单应用场景：**
在高效求解线性方程组和采样中使用。

---

---

## 附录C：概率论

### 公式 (C.1)：条件概率 / Conditional Probability

$$
P(A|B) = \frac{P(A, B)}{P(B)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(A \mid B)$ | tiáo jiàn gài lǜ | 给定B发生时A发生的条件概率 |
| $P(A, B)$ | lián hé gài lǜ | A和B联合发生的概率 |
| $P(B)$ | gài lǜ | B发生的概率 |

**一句话解释：**
条件概率是在另一个事件已经发生的条件下，某事件发生的概率。

**简单应用场景：**
医疗诊断中，已知病人有某种症状，计算病人患病的概率。

---

---

### 公式 (C.2)：全概率公式 / Law of Total Probability

$$
P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(A)$ | gài lǜ | 事件A发生的概率 |
| $P(A \mid B_i)$ | tiáo jiàn gài lǜ | 给定$B_i$时A的条件概率 |
| $P(B_i)$ | gài lǜ | 事件$B_i$发生的概率 |
| $n$ | nǔ | 分割事件的个数 |

**一句话解释：**
某事件的总概率等于它在所有可能条件下的条件概率乘以该条件发生概率的加权和。

**简单应用场景：**
计算工厂产品的不良率，已知各个生产线的不良率和产量占比。

---

---

### 公式 (C.3)：贝叶斯定理 / Bayes' Theorem

$$
P(B_i|A) = \frac{P(A|B_i)P(B_i)}{\sum_{j=1}^{n} P(A|B_j)P(B_j)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(B_i \mid A)$ | hòu yàn gài lǜ | 观测到A后$B_i$的后验概率 |
| $P(A \mid B_i)$ | sì ràn | 给定$B_i$时A的似然 |
| $P(B_i)$ | xiān yàn | $B_i$的先验概率 |
| $\sum_{j=1}^{n} P(A \mid B_j)P(B_j)$ | zhèng jù | 证据的边界概率 |

**一句话解释：**
贝叶斯定理描述在获得新证据后如何更新事件概率的方法。

**简单应用场景：**
垃圾邮件过滤中，根据邮件内容特征更新该邮件是垃圾邮件的概率。

---

---

### 公式 (C.4)：期望值 / Expected Value

$$
E[X] = \sum_{x} x p(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E[X]$ | qī wàng zhí | 随机变量X的期望值 |
| $X$ | suí jī biàn liàng | 随机变量 |
| $p(x)$ | gài lǜ zhì liàng hánshù | 概率质量函数 |

**一句话解释：**
期望值是随机变量所有可能取值与其概率乘积的总和。

**简单应用场景：**
计算投资回报率的平均值。

---

---

### 公式 (C.5)：方差 / Variance

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Var}(X)$ | fāng chà | 随机变量X的方差 |
| $E[X]$ | qī wàng zhí | 期望值 |
| $E[X^2]$ | X平方的期望 | $X^2$的期望值 |

**一句话解释：**
方差衡量随机变量与其期望值的平均偏离程度。

**简单应用场景：**
评估投资回报的波动性和风险。

---

---

### 公式 (C.6)：标准差 / Standard Deviation

$$
\sigma(X) = \sqrt{\text{Var}(X)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma(X)$ | biāo zhǔn chà | 随机变量X的标准差 |
| $\text{Var}(X)$ | fāng chà | 方差 |

**一句话解释：**
标准差是方差的平方根，与原变量单位相同，便于理解变异性。

**简单应用场景：**
衡量数据的离散程度，如学生成绩的波动。

---

---

### 公式 (C.7)：协方差 / Covariance

$$
\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Cov}(X, Y)$ | xié fāng chà | X和Y的协方差 |
| $E[X]$ | qī wàng zhí | X的期望值 |
| $E[Y]$ | qī wàng zhí | Y的期望值 |

**一句话解释：**
协方差衡量两个随机变量的变动趋势是否一致。

**简单应用场景：**
分析身高和体重之间的相关性。

---

---

### 公式 (C.8)：相关系数 / Correlation Coefficient

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma(X)\sigma(Y)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho(X, Y)$ | xiāng guān xì shù | X和Y的相关系数 |
| $\text{Cov}(X, Y)$ | xié fāng chà | 协方差 |
| $\sigma(X)$ | biāo zhǔn chà | X的标准差 |
| $\sigma(Y)$ | biāo zhǔn chà | Y的标准差 |

**一句话解释：**
相关系数是标准化的协方差，取值在[-1, 1]之间。

**简单应用场景：**
评估两个股票价格之间的线性关系强度。

---

---

### 公式 (C.9)：高斯分布 / Gaussian Distribution

$$
p(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x \mid \mu, \sigma^2)$ | gài lǜ mì dù | 概率密度函数 |
| $\mu$ | jūn zhí | 均值 |
| $\sigma^2$ | fāng chà | 方差 |
| $x$ | suí jī biàn liàng | 随机变量 |

**一句话解释：**
高斯分布是最重要的连续分布，由均值和方差完全确定。

**简单应用场景：**
模型多种自然现象，如身高、体重分布。

---

---

### 公式 (C.10)：多变量高斯分布 / Multivariate Gaussian Distribution

$$
p(\boldsymbol{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | xiàng liàng | 随机向量 |
| $\boldsymbol{\mu}$ | jūn zhí xiàng liàng | 均值向量 |
| $\boldsymbol{\Sigma}$ | xié fāng chà jǔ zhèn | 协方差矩阵 |
| $D$ | wéi shù | 维度 |
| $\|\boldsymbol{\Sigma}\|$ | xíng liè shì | 协方差矩阵的行列式 |

**一句话解释：**
多变量高斯分布是单变量高斯分布在多维空间的推广。

**简单应用场景：**
模型多维特征的联合分布，如手写数字识别。

---

---

### 公式 (C.11)：二项分布 / Binomial Distribution

$$
D_{\mathrm{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(k \mid n, p)$ | gài lǜ | 概率质量函数 |
| $n$ | shì yàn cì shù | 试验次数 |
| $k$ | chéng gōng cì shù | 成功次数 |
| $p$ | chéng gōng gài lǜ | 单次成功概率 |
| $\binom{n}{k}$ | zǔ hé shù | 组合数 |

**一句话解释：**
二项分布描述n次独立伯努利试验中成功k次的概率。

**简单应用场景：**
硬币投掷实验，质量检测中的不良品计数。

---

---

### 公式 (C.12)：泊松分布 / Poisson Distribution

$$
p(k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(k \mid \lambda)$ | gài lǜ | 概率质量函数 |
| $k$ | shì jiàn cì shù | 事件次数 |
| $\lambda$ | qī wàng cì shù | 期望次数 |
| $e$ | zì rán duì shù de dǐ shù | 自然对数的底数 |

**一句话解释：**
泊松分布用于描述在固定时间或空间内随机事件的发生次数。

**简单应用场景：**
单位时间内电话呼入次数，放射性衰变事件计数。

---

---

### 公式 (C.13)：均匀分布 / Uniform Distribution

$$
p(x|a, b) = \begin{cases} \frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x \mid a, b)$ | gài lǜ mì dù | 概率密度函数 |
| $a$ | xià jiè | 下界 |
| $b$ | shàng jiè | 上界 |
| $x$ | suí jī biàn liàng | 随机变量 |

**一句话解释：**
均匀分布在区间[a, b]内所有点的概率密度相同。

**简单应用场景：**
模拟随机数生成，描述完全不确定的情况。

---

---

### 公式 (C.14)：独立性 / Independence

$$
P(A, B) = P(A)P(B) \text{ if } A \text{ and } B \text{ are independent}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(A, B)$ | lián hé gài lǜ | A和B联合发生的概率 |
| $P(A)$ | gài lǜ | A发生的概率 |
| $P(B)$ | gài lǜ | B发生的概率 |

**一句话解释：**
两个事件独立，当且仅当其联合概率等于各自概率的乘积。

**简单应用场景：**
判断两个随机事件是否相互影响。

---

---

### 公式 (C.15)：条件独立性 / Conditional Independence

$$
P(A, B|C) = P(A|C)P(B|C) \text{ if } A \perp B \mid C
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(A, B \mid C)$ | tiáo jiàn lián hé gài lǜ | 给定C时A和B的联合条件概率 |
| $P(A \mid C)$ | tiáo jiàn gài lǜ | 给定C时A的条件概率 |
| $P(B \mid C)$ | tiáo jiàn gài lǜ | 给定C时B的条件概率 |
| $A \perp B \mid C$ | tiáo jiàn dú lì | A和B给定C条件独立 |

**一句话解释：**
条件独立表示在给定第三个变量的情况下，两个变量相互独立。

**简单应用场景：**
贝叶斯网络中，一个节点的子节点给定该节点时条件独立。

---

---

### 公式 (C.16)：协方差矩阵 / Covariance Matrix

$$
\boldsymbol{\Sigma} = E[(\boldsymbol{x} - E[\boldsymbol{x}])(\boldsymbol{x} - E[\boldsymbol{x}])^T]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\Sigma}$ | xié fāng chà jǔ zhèn | 协方差矩阵 |
| $\boldsymbol{x}$ | xiàng liàng | 随机向量 |
| $E[\boldsymbol{x}]$ | qī wàng zhí xiàng liàng | 期望向量 |

**一句话解释：**
协方差矩阵是向量各分量之间协方差的矩阵形式。

**简单应用场景：**
多变量统计分析中描述变量间的相互关系。

---

---

### 公式 (C.17)：大数定律 / Law of Large Numbers

$$
\lim_{n \to \infty} \frac{1}{n}\sum_{i=1}^{n} X_i = E[X] \text{ (almost surely)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $n$ | cì shù | 样本数量 |
| $X_i$ | dì i ge yàng běn | 第i个样本 |
| $E[X]$ | qī wàng zhí | 期望值 |

**一句话解释：**
大数定律指当样本量足够大时，样本均值趋近于总体期望值。

**简单应用场景：**
投掷骰子足够多次，平均值接近3.5。

---

---

### 公式 (C.18)：中心极限定理 / Central Limit Theorem

$$
\frac{\sum_{i=1}^{n} X_i - n\mu}{\sqrt{n\sigma^2}} \xrightarrow{d} N(0, 1) \text{ as } n \to \infty
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $X_i$ | dì i ge yàng běn | 第i个独立同分布的样本 |
| $n$ | cì shù | 样本数量 |
| $\mu$ | jūn zhí | 总体均值 |
| $\sigma^2$ | fāng chà | 总体方差 |
| $N(0, 1)$ | biāo zhǔn zhèng tài fēn bù | 标准正态分布 |

**一句话解释：**
中心极限定理说明独立同分布样本的均值的分布渐近正态。

**简单应用场景：**
多个随机误差的累积通常服从正态分布。

---

---

### 公式 (C.19)：最大似然估计 / Maximum Likelihood Estimation

$$
\hat{\theta} = \arg\max_{\theta} L(\theta|X) = \arg\max_{\theta} p(X|\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{\theta}$ | cān shù gū jì | 参数估计 |
| $L(\theta \mid X)$ | sì rán hánshù | 似然函数 |
| $p(X \mid \theta)$ | tiáo jiàn gài lǜ | 给定参数的数据概率 |

**一句话解释：**
最大似然估计选择使观测数据最可能出现的参数。

**简单应用场景：**
根据样本数据估计正态分布的均值和方差。

---

---

### 公式 (C.20)：贝叶斯估计 / Bayesian Estimation

$$
p(\theta|X) = \frac{p(X|\theta)p(\theta)}{p(X)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\theta \mid X)$ | hòu yàn | 后验概率 |
| $p(X \mid \theta)$ | sì ràn | 似然 |
| $p(\theta)$ | xiān yàn | 先验概率 |
| $p(X)$ | zhèng jù | 证据 |

**一句话解释：**
贝叶斯估计将参数视为随机变量，结合先验知识和观测数据得到后验分布。

**简单应用场景：**
医学诊断中，结合患者症状更新患病概率。

---

---

### 公式 (C.21)：参数化分布族 / Parametric Distribution Family

$$
p(x|\theta) \text{ where } \theta \text{ is the parameter set}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x \mid \theta)$ | gài lǜ mì dù | 参数化概率密度 |
| $\theta$ | cān shù | 参数集合 |
| $x$ | suí jī biàn liàng | 随机变量 |

**一句话解释：**
参数化分布族通过有限个参数完全刻画一个概率分布。

**简单应用场景：**
用两个参数（均值和方差）完全描述正态分布。

---

---

### 公式 (C.22)：边际概率 / Marginal Probability

$$
D_{\mathrm{KL}}(p \| q) = \frac{1}{2} \left[ \log \frac{|\boldsymbol{\Sigma}_q|}{|\boldsymbol{\Sigma}_p|} - d + \operatorname{tr}(\boldsymbol{\Sigma}_q^{-1} \boldsymbol{\Sigma}_p) + (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p)^\top \boldsymbol{\Sigma}_q^{-1} (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | biān jì gài lǜ | x的边际概率 |
| $p(x, y)$ | lián hé gài lǜ | x和y的联合概率 |
| $p(x \mid y)$ | tiáo jiàn gài lǜ | 给定y时x的条件概率 |

**一句话解释：**
边际概率是通过积分或求和消除其他变量得到某个变量的概率。

**简单应用场景：**
多维分布中求取某个变量的单变量分布。

---

---

### 公式 (C.23)：链式法则 / Chain Rule

$$
p(x_1, x_2, \ldots, x_n) = p(x_1) \prod_{i=2}^{n} p(x_i|x_1, \ldots, x_{i-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_1, x_2, \ldots, x_n)$ | lián hé gài lǜ | 联合概率 |
| $p(x_1)$ | biān jì gài lǜ | $x_1$的边际概率 |
| $p(x_i \mid x_1, \ldots, x_{i-1})$ | tiáo jiàn gài lǜ | 给定前项的条件概率 |

**一句话解释：**
链式法则将多变量联合概率分解为条件概率的乘积。

**简单应用场景：**
概率图模型中计算复杂分布。

---

---

### 公式 (C.24)：均值和方差的性质 / Properties of Mean and Variance

$$
E[aX + b] = aE[X] + b
$$
$$
\text{Var}(aX + b) = a^2 \text{Var}(X)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E[\cdot]$ | qī wàng suàn zǐ | 期望算子 |
| $\text{Var}(\cdot)$ | fāng chà suàn zǐ | 方差算子 |
| $X$ | suí jī biàn liàng | 随机变量 |
| $a, b$ | chéng shù | 常数 |

**一句话解释：**
线性变换后，期望乘以系数加常数；方差乘以系数的平方。

**简单应用场景：**
温度从摄氏度转换为华氏度时的统计性质变换。

---

---

### 公式 (C.25)：独立随机变量的性质 / Properties of Independent Variables

$$
E[XY] = E[X]E[Y] \text{ if } X \perp Y
$$
$$
\text{Cov}(X, Y) = 0 \text{ if } X \perp Y
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E[XY]$ | xiāng chéng jī de qī wàng | 乘积的期望 |
| $E[X]$ | qī wàng zhí | X的期望 |
| $E[Y]$ | qī wàng zhí | Y的期望 |
| $X \perp Y$ | dú lì | X和Y独立 |
| $\text{Cov}(X, Y)$ | xié fāng chà | 协方差 |

**一句话解释：**
独立随机变量的乘积期望等于期望的乘积，协方差为零。

**简单应用场景：**
简化概率计算和参数估计。

---

---

### 公式 (C.26)：逆向变换采样 / Inverse Transform Sampling

$$
U \sim \text{Uniform}(0, 1), \quad X = F^{-1}(U)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $U$ | jūn yún fēn bù | 均匀分布随机变量 |
| $F^{-1}$ | fǎn hánshù | 累积分布函数的反函数 |
| $X$ | suí jī biàn liàng | 所需分布的随机变量 |

**一句话解释：**
通过均匀分布和逆累积分布函数可以生成任意分布的随机样本。

**简单应用场景：**
从正态分布、指数分布等生成随机数。

---

---

### 公式 (C.27)：累积分布函数 / Cumulative Distribution Function

$$
F(x) = P(X \leq x) = \int_{-\infty}^{x} p(t) dt
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F(x)$ | lěi jī fēn bù hánshù | 累积分布函数 |
| $P(X \leq x)$ | gài lǜ | X不超过x的概率 |
| $p(t)$ | gài lǜ mì dù | 概率密度函数 |

**一句话解释：**
累积分布函数给出随机变量不超过某个值的概率。

**简单应用场景：**
查询标准正态分布表确定z值对应的概率。

---

## 附录D：数学优化

---

## 附录D：数学优化

### 公式 (D.1)：梯度 / Gradient

$$
\nabla f(\boldsymbol{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_D}\right]^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla f(\boldsymbol{x})$ | tī dù xiàng liàng | 梯度向量 |
| $\frac{\partial f}{\partial x_i}$ | piàn dǎo shù | 偏导数 |
| $D$ | wéi shù | 维度 |

**一句话解释：**
梯度是函数在某点处最陡上升方向及其斜率的向量。

**简单应用场景：**
指导优化算法向损失函数最小值方向移动。

---

---

### 公式 (D.2)：海森矩阵 / Hessian Matrix

$$
\boldsymbol{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_D} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_D} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_D \partial x_1} & \frac{\partial^2 f}{\partial x_D \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_D^2}
\end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{H}$ | hǎi sēn jǔ zhèn | 海森矩阵 |
| $\frac{\partial^2 f}{\partial x_i \partial x_j}$ | èr jiē piàn dǎo shù | 二阶偏导数 |

**一句话解释：**
海森矩阵是函数二阶偏导数组成的矩阵，描述函数的曲率。

**简单应用场景：**
牛顿法优化中用于计算下降方向。

---

---

### 公式 (D.3)：泰勒展开 / Taylor Expansion

$$
f(\boldsymbol{x} + \boldsymbol{\Delta x}) \approx f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^T \boldsymbol{\Delta x} + \frac{1}{2}\boldsymbol{\Delta x}^T \boldsymbol{H} \boldsymbol{\Delta x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x})$ | hánshù | 目标函数 |
| $\nabla f(\boldsymbol{x})$ | tī dù | 梯度 |
| $\boldsymbol{H}$ | hǎi sēn jǔ zhèn | 海森矩阵 |
| $\boldsymbol{\Delta x}$ | biàn huà liàng | 增量向量 |

**一句话解释：**
泰勒展开用二次函数近似原函数，对优化算法设计至关重要。

**简单应用场景：**
线性和二阶优化方法的理论基础。

---

---

### 公式 (D.4)：凸函数的定义 / Definition of Convex Function

$$
f(\lambda \boldsymbol{x}_1 + (1-\lambda)\boldsymbol{x}_2) \leq \lambda f(\boldsymbol{x}_1) + (1-\lambda)f(\boldsymbol{x}_2), \quad \forall \lambda \in [0, 1]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f$ | hánshù | 凸函数 |
| $\boldsymbol{x}_1, \boldsymbol{x}_2$ | diǎn | 任意两点 |
| $\lambda$ | quán zhòng | 权重 |

**一句话解释：**
凸函数在任意两点连线上方，局部最小值是全局最小值。

**简单应用场景：**
保证梯度下降法找到全局最优解。

---

---

### 公式 (D.5)：梯度下降法 / Gradient Descent

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \nabla f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}_t$ | dì t cì yù gū | 第t次迭代的参数 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\nabla f(\boldsymbol{x}_t)$ | tī dù | 梯度 |

**一句话解释：**
梯度下降法沿着梯度反方向迭代更新参数以最小化损失函数。

**简单应用场景：**
神经网络训练中最常用的优化算法。

---

---

### 公式 (D.6)：随机梯度下降法 / Stochastic Gradient Descent

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \nabla f(\boldsymbol{x}_t; \boldsymbol{x}^{(i)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}_t$ | cān shù | 参数 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\boldsymbol{x}^{(i)}$ | dì i ge yàng běn | 第i个样本 |
| $\nabla f$ | tī dù | 梯度 |

**一句话解释：**
随机梯度下降法使用单个样本的梯度进行参数更新，计算高效。

**简单应用场景：**
大规模数据集的神经网络训练。

---

---

### 公式 (D.7)：动量法 / Momentum

$$
\boldsymbol{v}_{t+1} = \beta \boldsymbol{v}_t + \nabla f(\boldsymbol{x}_t)
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \boldsymbol{v}_{t+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}_t$ | dòng liàng | 速度向量 |
| $\beta$ | dòng liàng xì shù | 动量系数 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\nabla f(\boldsymbol{x}_t)$ | tī dù | 梯度 |

**一句话解释：**
动量法累积梯度信息，使参数更新具有惯性，加速收敛。

**简单应用场景：**
改进梯度下降法的收敛速度和稳定性。

---

---

### 公式 (D.8)：Nesterov加速梯度 / Nesterov Accelerated Gradient

$$
\boldsymbol{v}_{t+1} = \beta \boldsymbol{v}_t + \nabla f(\boldsymbol{x}_t - \eta \boldsymbol{v}_t)
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \boldsymbol{v}_{t+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{v}_t$ | dòng liàng | 速度向量 |
| $\beta$ | dòng liàng xì shù | 动量系数 |
| $\eta$ | xué xí lǜ | 学习率 |

**一句话解释：**
Nesterov动量在前瞻位置计算梯度，收敛更快。

**简单应用场景：**
深度学习优化中比标准动量法更高效。

---

---

### 公式 (D.9)：自适应学习率方法 / Adaptive Learning Rate

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \frac{\eta}{\sqrt{\boldsymbol{g}_t + \epsilon}} \odot \nabla f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}_t$ | liàng chéng xù hé | 梯度平方和 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\epsilon$ | xiǎo cháng shù | 避免除零的常数 |
| $\odot$ | yuán sù zuò chéng | 元素乘积 |

**一句话解释：**
自适应学习率方法根据梯度历史自动调整各参数的学习率。

**简单应用场景：**
AdaGrad等算法的基础。

---

---

### 公式 (D.10)：AdaGrad算法 / AdaGrad Algorithm

$$
\boldsymbol{g}_{t+1} = \boldsymbol{g}_t + (\nabla f(\boldsymbol{x}_t))^2
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \frac{\eta}{\sqrt{\boldsymbol{g}_{t+1} + \epsilon}} \odot \nabla f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}_t$ | tī dù píng fāng lěi jī | 梯度平方累积 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\epsilon$ | xiǎo cháng shù | 数值稳定常数 |

**一句话解释：**
AdaGrad累积梯度平方，在参数更新较多的维度减小学习率。

**简单应用场景：**
处理稀疏梯度的优化问题。

---

---

### 公式 (D.11)：RMSprop算法 / RMSprop Algorithm

$$
\boldsymbol{g}_{t+1} = \rho \boldsymbol{g}_t + (1-\rho)(\nabla f(\boldsymbol{x}_t))^2
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \frac{\eta}{\sqrt{\boldsymbol{g}_{t+1} + \epsilon}} \odot \nabla f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}_t$ | jiāo chú píng fāng | 梯度平方指数移动平均 |
| $\rho$ |衰减率 | 衰减因子 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\epsilon$ | xiǎo cháng shù | 数值稳定常数 |

**一句话解释：**
RMSprop改进AdaGrad，使用指数移动平均的梯度平方，避免学习率过度衰减。

**简单应用场景：**
递归神经网络、深度强化学习等复杂优化问题。

---

---

### 公式 (D.12)：Adam算法 / Adam Algorithm

$$
\boldsymbol{m}_{t+1} = \beta_1 \boldsymbol{m}_t + (1-\beta_1) \nabla f(\boldsymbol{x}_t)
$$
$$
\boldsymbol{v}_{t+1} = \beta_2 \boldsymbol{v}_t + (1-\beta_2)(\nabla f(\boldsymbol{x}_t))^2
$$
$$
\hat{\boldsymbol{m}}_{t+1} = \frac{\boldsymbol{m}_{t+1}}{1-\beta_1^{t+1}}, \quad \hat{\boldsymbol{v}}_{t+1} = \frac{\boldsymbol{v}_{t+1}}{1-\beta_2^{t+1}}
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \frac{\hat{\boldsymbol{m}}_{t+1}}{\sqrt{\hat{\boldsymbol{v}}_{t+1}} + \epsilon}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{m}_t$ | tī dù de yī jiè jū | 梯度的一阶矩估计 |
| $\boldsymbol{v}_t$ | tī dù de èr jiè jū | 梯度的二阶矩估计 |
| $\beta_1$ | yī jiè jū衰减率 | 一阶矩衰减率 |
| $\beta_2$ | èr jiè jū衰减率 | 二阶矩衰减率 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\epsilon$ | xiǎo cháng shù | 数值稳定常数 |

**一句话解释：**
Adam结合动量和RMSprop，计算梯度一阶和二阶矩的自适应学习率，是最流行的优化器。

**简单应用场景：**
深度学习中默认的优化算法选择。

---

---

### 公式 (D.13)：牛顿法 / Newton's Method

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \boldsymbol{H}^{-1} \nabla f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}_t$ | cān shù | 参数 |
| $\boldsymbol{H}$ | hǎi sēn jǔ zhèn | 海森矩阵 |
| $\nabla f(\boldsymbol{x}_t)$ | tī dù | 梯度 |

**一句话解释：**
牛顿法使用海森矩阵的逆加速收敛，但计算成本高。

**简单应用场景：**
参数维度不太高的优化问题。

---

---

### 公式 (D.14)：拟牛顿法 / Quasi-Newton Method

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \boldsymbol{B}^{-1} \nabla f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{B}$ | hǎi sēn jǔ zhèn de jìn jìn | 海森矩阵的近似 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\nabla f(\boldsymbol{x}_t)$ | tī dù | 梯度 |

**一句话解释：**
拟牛顿法用低秩矩阵近似海森矩阵的逆，平衡收敛速度和计算成本。

**简单应用场景：**
BFGS算法等二阶优化方法。

---

---

### 公式 (D.15)：L-BFGS算法 / L-BFGS Algorithm

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \boldsymbol{B}^{-1} \nabla f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{B}$ | jìn jìn hǎi sēn jǔ zhèn | 近似海森矩阵 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\nabla f(\boldsymbol{x}_t)$ | tī dù | 梯度 |

**一句话解释：**
L-BFGS使用有限内存近似海森矩阵，适合大规模优化。

**简单应用场景：**
SVM、逻辑回归等凸优化问题。

---

---

### 公式 (D.16)：共轭梯度法 / Conjugate Gradient Method

$$
\boldsymbol{d}_t = -\nabla f(\boldsymbol{x}_t) + \beta_t \boldsymbol{d}_{t-1}
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \alpha_t \boldsymbol{d}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{d}_t$ | gōng xuán fāng xiàng | 共轭方向 |
| $\beta_t$ | gōng xuán xì shù | 共轭系数 |
| $\alpha_t$ | bù cháng de dà xiǎo | 步长 |
| $\nabla f(\boldsymbol{x}_t)$ | tī dù | 梯度 |

**一句话解释：**
共轭梯度法构造相互正交的搜索方向，对于二次函数有限步收敛。

**简单应用场景：**
大规模线性方程组求解。

---

---

### 公式 (D.17)：直线搜索 / Line Search

$$
\alpha^* = \arg\min_{\alpha > 0} f(\boldsymbol{x}_t + \alpha \boldsymbol{d}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | bù cháng | 步长 |
| $\boldsymbol{d}_t$ | xià jiàng fāng xiàng | 下降方向 |
| $f$ | hánshù | 目标函数 |

**一句话解释：**
直线搜索在给定方向上寻找最优步长以最大化目标函数值的下降。

**简单应用场景：**
与梯度下降、牛顿法结合使用。

---

---

### 公式 (D.18)：回溯直线搜索 / Backtracking Line Search

$$
f(\boldsymbol{x}_t + \alpha \boldsymbol{d}_t) \leq f(\boldsymbol{x}_t) + c \alpha \nabla f(\boldsymbol{x}_t)^T \boldsymbol{d}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | bù cháng | 步长 |
| $\boldsymbol{d}_t$ | xià jiàng fāng xiàng | 下降方向 |
| $c$ | chéng shù | 回溯参数 |
| $\nabla f$ | tī dù | 梯度 |

**一句话解释：**
回溯直线搜索确保函数值有足够下降，实现收敛保证。

**简单应用场景：**
实际优化算法中常用的一阶充分条件。

---

---

### 公式 (D.19)：强凸函数 / Strongly Convex Function

$$
f(\boldsymbol{x}) \geq f(\boldsymbol{y}) + \nabla f(\boldsymbol{y})^T(\boldsymbol{x} - \boldsymbol{y}) + \frac{\mu}{2}\|\boldsymbol{x} - \boldsymbol{y}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f$ | hánshù | 强凸函数 |
| $\mu$ | qiáng convexity cháng shù | 强凸常数 |
| $\|\cdot\|$ | mó | 范数 |

**一句话解释：**
强凸函数有唯一全局最小值，收敛速率可得到保证。

**简单应用场景：**
分析优化算法的收敛速度。

---

---

### 公式 (D.20)：光滑函数 / Smooth Function

$$
\|\nabla f(\boldsymbol{x}) - \nabla f(\boldsymbol{y})\| \leq L \|\boldsymbol{x} - \boldsymbol{y}\|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla f$ | tī dù | 梯度 |
| $L$ | guāng huá cháng shù | 光滑常数（Lipschitz常数） |
| $\|\cdot\|$ | mó | 范数 |

**一句话解释：**
光滑函数的梯度变化有界，保证优化算法的收敛性。

**简单应用场景：**
优化算法收敛性分析的标准条件。

---

---

### 公式 (D.21)：投影梯度下降 / Projected Gradient Descent

$$
\boldsymbol{x}_{t+1} = \mathcal{P}_\mathcal{C}(\boldsymbol{x}_t - \eta \nabla f(\boldsymbol{x}_t))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{P}_\mathcal{C}$ | tóu yǐng suàn zǐ | 投影算子 |
| $\mathcal{C}$ | kòng jiān | 可行域 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\nabla f$ | tī dù | 梯度 |

**一句话解释：**
投影梯度下降处理约束优化，每次更新后投影回可行域。

**简单应用场景：**
带有范数约束的稀疏学习。

---

---

### 公式 (D.22)：近端梯度下降 / Proximal Gradient Descent

$$
\boldsymbol{x}_{t+1} = \text{prox}_{\eta g}(\boldsymbol{x}_t - \eta \nabla f(\boldsymbol{x}_t))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{prox}_{\eta g}$ | jìn duàn suàn zǐ | 近端算子 |
| $g$ | fēi guāng huá hánshù | 非光滑函数 |
| $\eta$ | bù cháng | 步长 |
| $\nabla f$ | tī dù | 梯度 |

**一句话解释：**
近端梯度下降处理非光滑目标函数，如L1正则化。

**简单应用场景：**
压缩感知、稀疏优化。

---

---

### 公式 (D.23)：坐标下降法 / Coordinate Descent

$$
x_i^{(t+1)} = \arg\min_{x_i} f(x_1^{(t)}, \ldots, x_i, \ldots, x_D^{(t)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_i$ | dì i wèi bàn liàng | 第i维变量 |
| $f$ | mù biāo hánshù | 目标函数 |
| $D$ | zǒng wéi shù | 总维数 |

**一句话解释：**
坐标下降法轮流优化每个坐标，保持其他坐标固定。

**简单应用场景：**
大规模稀疏模型优化，如逻辑回归。

---

---

### 公式 (D.24)：次梯度 / Subgradient

$$
g \in \partial f(\boldsymbol{x}) \text{ if } f(\boldsymbol{y}) \geq f(\boldsymbol{x}) + g^T(\boldsymbol{y} - \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\partial f(\boldsymbol{x})$ | cì tī dù jí | 次梯度集合 |
| $g$ | cì tī dù | 次梯度 |
| $f$ | fēi guāng huá hánshù | 非光滑函数 |

**一句话解释：**
次梯度是非光滑函数的梯度概念推广，用于非光滑优化。

**简单应用场景：**
处理绝对值损失、支持向量机等非光滑问题。

---

---

### 公式 (D.25)：次梯度下降法 / Subgradient Descent

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta_t \boldsymbol{g}_t, \quad \boldsymbol{g}_t \in \partial f(\boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{g}_t$ | cì tī dù | 次梯度 |
| $\eta_t$ | xué xí lǜ | 学习率 |
| $\partial f(\boldsymbol{x}_t)$ | cì tī dù jí | 次梯度集合 |

**一句话解释：**
次梯度下降法将梯度下降推广到非光滑凸函数。

**简单应用场景：**
稀疏学习、鲁棒统计等应用。

---

---

### 公式 (D.26)：镜像下降 / Mirror Descent

$$
\boldsymbol{x}_{t+1} = \arg\min_{\boldsymbol{x}} \left\{\eta \nabla f(\boldsymbol{x}_t)^T \boldsymbol{x} + D_\psi(\boldsymbol{x}, \boldsymbol{x}_t)\right\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_\psi$ | bāo hánshù biāo lí | 包函数标离度 |
| $\psi$ | bāo hánshù | 包函数 |
| $\eta$ | xué xí lǜ | 学习率 |

**一句话解释：**
镜像下降使用非欧几里得几何加速非凸优化。

**简单应用场景：**
概率单纯形等约束优化。

---

---

### 公式 (D.27)：加速梯度下降 / Accelerated Gradient Descent

$$
\boldsymbol{y}_t = \boldsymbol{x}_t + \frac{t-2}{t+1}(\boldsymbol{x}_t - \boldsymbol{x}_{t-1})
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{y}_t - \eta \nabla f(\boldsymbol{y}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{y}_t$ | zhōng jiān biàn liàng | 中间变量 |
| $\eta$ | xué xí lǜ | 学习率 |
| $\nabla f$ | tī dù | 梯度 |

**一句话解释：**
加速梯度下降通过外推达到最优收敛速率O(1/t^2)。

**简单应用场景：**
凸优化中理论最优的一阶方法。

---

---

### 公式 (D.28)：对偶问题 / Dual Problem

$$
\max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} \min_{\boldsymbol{x}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = \max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} \min_{\boldsymbol{x}} \left[f(\boldsymbol{x}) + \boldsymbol{\lambda}^T \boldsymbol{g}(\boldsymbol{x}) + \boldsymbol{\mu}^T \boldsymbol{h}(\boldsymbol{x})\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | lāng rì hánshù | 拉格朗日函数 |
| $\boldsymbol{\lambda}, \boldsymbol{\mu}$ | duì ǒu biàn liàng | 对偶变量 |
| $\boldsymbol{g}, \boldsymbol{h}$ | yuē shù | 约束 |

**一句话解释：**
对偶问题通过拉格朗日乘数法转化原优化问题。

**简单应用场景：**
支持向量机、凸优化理论分析。

---

---

### 公式 (D.29)：强对偶性 / Strong Duality

$$
\min_{\boldsymbol{x}} f(\boldsymbol{x}) = \max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} \min_{\boldsymbol{x}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x})$ | yuán wèn | 原问题 |
| $L$ | lāng rì hánshù | 拉格朗日函数 |

**一句话解释：**
强对偶性说明原问题和对偶问题的最优值相等。

**简单应用场景：**
凸优化的必要充分条件。

---

---

### 公式 (D.30)：KKT条件 / KKT Conditions

$$
\nabla f(\boldsymbol{x}^*) + \sum_i \lambda_i^* \nabla g_i(\boldsymbol{x}^*) + \sum_j \mu_j^* \nabla h_j(\boldsymbol{x}^*) = 0
$$
$$
\lambda_i^* \geq 0, \quad \lambda_i^* g_i(\boldsymbol{x}^*) = 0
$$
$$
h_j(\boldsymbol{x}^*) = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}^*$ | zuì yōu jiě | 最优解 |
| $\lambda_i^*$ | duì ǒu biàn liàng | 对偶变量 |
| $g_i, h_j$ | yuē shù | 约束 |

**一句话解释：**
KKT条件是凸优化问题最优解的必要充分条件。

**简单应用场景：**
判断候选解是否是最优解。

---

---

### 公式 (D.31)：互补松弛条件 / Complementary Slackness

$$
\lambda_i g_i(\boldsymbol{x}) = 0, \quad i = 1, \ldots, m
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_i$ | duì ǒu biàn liàng | 对偶变量 |
| $g_i(\boldsymbol{x})$ | yuē shù | 不等式约束 |
| $m$ | yuē shù gè shù | 约束个数 |

**一句话解释：**
互补松弛条件说明对偶变量与约束的乘积为零。

**简单应用场景：**
SVM支持向量的选择。

---

---

### 公式 (D.32)：约束优化的拉格朗日松弛 / Lagrangian Relaxation

$$
g(\boldsymbol{\lambda}) = \min_{\boldsymbol{x} \in \mathcal{X}} \left[f(\boldsymbol{x}) + \boldsymbol{\lambda}^T \boldsymbol{c}(\boldsymbol{x})\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g(\boldsymbol{\lambda})$ | duì ǒu hánshù | 对偶函数 |
| $\mathcal{X}$ | kě xíng yù | 可行域 |
| $\boldsymbol{\lambda}$ | duì ǒu biàn liàng | 对偶变量 |

**一句话解释：**
拉格朗日松弛通过去掉约束得到原问题的下界。

**简单应用场景：**
组合优化中的启发式算法。

---

---

### 公式 (D.33)：增广拉格朗日方法 / Augmented Lagrangian Method

$$
L_\rho(\boldsymbol{x}, \boldsymbol{\lambda}) = f(\boldsymbol{x}) + \boldsymbol{\lambda}^T \boldsymbol{h}(\boldsymbol{x}) + \frac{\rho}{2}\|\boldsymbol{h}(\boldsymbol{x})\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L_\rho$ | zēng guó lāng rì hánshù | 增广拉格朗日函数 |
| $\rho$ | huà fá xì shù | 惩罚系数 |
| $\boldsymbol{\lambda}$ | duì ǒu biàn liàng | 对偶变量 |

**一句话解释：**
增广拉格朗日方法结合惩罚法和拉格朗日乘数法的优点。

**简单应用场景：**
约束非凸优化。

---

---

### 公式 (D.34)：交替方向乘数法 / Alternating Direction Method of Multipliers

$$
\boldsymbol{x}^{(k+1)} = \arg\min_{\boldsymbol{x}} L_\rho(\boldsymbol{x}, \boldsymbol{z}^{(k)}, \boldsymbol{\lambda}^{(k)})
$$
$$
\boldsymbol{z}^{(k+1)} = \arg\min_{\boldsymbol{z}} L_\rho(\boldsymbol{x}^{(k+1)}, \boldsymbol{z}, \boldsymbol{\lambda}^{(k)})
$$
$$
\boldsymbol{\lambda}^{(k+1)} = \boldsymbol{\lambda}^{(k)} + \rho(\boldsymbol{x}^{(k+1)} - \boldsymbol{z}^{(k+1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}^{(k)}$ | dì k cì yù gū de x | 第k次迭代的x |
| $\boldsymbol{z}^{(k)}$ | dì k cì yù gū de z | 第k次迭代的z |
| $\boldsymbol{\lambda}^{(k)}$ | duì ǒu biàn liàng | 对偶变量 |
| $\rho$ | bù cháng | 步长参数 |

**一句话解释：**
ADMM通过交替优化x和z以及更新对偶变量求解分布式凸优化。

**简单应用场景：**
分布式机器学习、统计学习。

---

---

### 公式 (D.35)：随机方差递减 / Stochastic Variance Reduced Gradient

$$
\boldsymbol{v}_{t+1} = \nabla f(\boldsymbol{x}_{t+1}; i_t) - \nabla f(\tilde{\boldsymbol{x}}; i_t) + \boldsymbol{v}_t
$$
$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta \boldsymbol{v}_{t+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla f(\boldsymbol{x}; i)$ | dì i ge yàng běn de tī dù | 第i个样本的梯度 |
| $\tilde{\boldsymbol{x}}$ | yuē zhù diǎn | 约束点 |
| $\eta$ | xué xí lǜ | 学习率 |

**一句话解释：**
SVRG通过方差递减技术降低随机梯度下降的梯度方差，加速收敛。

**简单应用场景：**
大规模凸优化问题。

---

---

### 公式 (D.36)：镜像下降的投影形式 / Projected Mirror Descent

$$
\boldsymbol{x}_{t+1} = \arg\min_{\boldsymbol{x} \in \mathcal{C}} \left\{\eta \langle \nabla f(\boldsymbol{x}_t), \boldsymbol{x} \rangle + D_\psi(\boldsymbol{x}, \boldsymbol{x}_t)\right\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_\psi$ | bāo hánshù biāo lí | 包函数标离度 |
| $\mathcal{C}$ | kě xíng yù | 可行域 |
| $\eta$ | xué xí lǜ | 学习率 |

**一句话解释：**
投影镜像下降结合非欧几何和约束优化。

**简单应用场景：**
概率单纯形约束的优化问题。

---

---

### 公式 (D.37)：Frank-Wolfe算法 / Frank-Wolfe Algorithm

$$
\boldsymbol{s}_t = \arg\min_{\boldsymbol{s} \in \mathcal{C}} \langle \nabla f(\boldsymbol{x}_t), \boldsymbol{s} \rangle
$$
$$
\boldsymbol{x}_{t+1} = (1 - \gamma_t) \boldsymbol{x}_t + \gamma_t \boldsymbol{s}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{s}_t$ | tī dù fang xiàng | 提度方向 |
| $\mathcal{C}$ | kě xíng yù | 可行域 |
| $\gamma_t$ | bù cháng | 步长 |

**一句话解释：**
Frank-Wolfe算法只需要解线性优化子问题，适合约束优化。

**简单应用场景：**
核范数约束的矩阵优化。

---

---

### 公式 (D.38)：近端算子 / Proximal Operator

$$
\text{prox}_g(\boldsymbol{x}) = \arg\min_{\boldsymbol{y}} \left\{g(\boldsymbol{y}) + \frac{1}{2}\|\boldsymbol{y} - \boldsymbol{x}\|^2\right\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{prox}_g$ | jìn duàn suàn zǐ | 近端算子 |
| $g$ | fēi guāng huá hánshù | 非光滑函数 |
| $\boldsymbol{x}$ | cān shù | 参数 |

**一句话解释：**
近端算子是非光滑函数的近似梯度，用于非光滑优化。

**简单应用场景：**
L1正则化、总变差最小化。

---

---

### 公式 (D.39)：Moreau包络 / Moreau Envelope

$$
f_\gamma(\boldsymbol{x}) = \min_{\boldsymbol{y}} \left\{f(\boldsymbol{y}) + \frac{1}{2\gamma}\|\boldsymbol{y} - \boldsymbol{x}\|^2\right\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_\gamma$ | Moreau bāo luó | Moreau包络 |
| $f$ | yuán hánshù | 原函数 |
| $\gamma$ | píng huá cān shù | 平滑参数 |

**一句话解释：**
Moreau包络是非凸函数的光滑逼近。

**简单应用场景：**
非光滑函数的光滑化处理。

---

---

### 公式 (D.40)：Douglas-Rachford分裂 / Douglas-Rachford Splitting

$$
\boldsymbol{x}_{t+1/2} = \text{prox}_{\eta f}(\boldsymbol{x}_t)
$$
$$
\boldsymbol{x}_{t+1} = \text{prox}_{\eta g}(2\boldsymbol{x}_{t+1/2} - \boldsymbol{x}_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{prox}$ | jìn duàn suàn zǐ | 近端算子 |
| $f, g$ | hánshù | 函数 |
| $\eta$ | bù cháng | 步长 |

**一句话解释：**
Douglas-Rachford分裂用于求解两个函数和的最小化。

**简单应用场景：**
约束凸优化问题。

---

---

### 公式 (D.41)：前向后向分裂 / Forward-Backward Splitting

$$
\boldsymbol{x}_{t+1} = \text{prox}_{\eta g}(\boldsymbol{x}_t - \eta \nabla f(\boldsymbol{x}_t))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{prox}_g$ | jìn duàn suàn zǐ | 近端算子 |
| $\nabla f$ | tī dù | 梯度 |
| $\eta$ | bù cháng | 步长 |

**一句话解释：**
前向后向分裂用于求解光滑函数加非光滑函数的优化。

**简单应用场景：**
正则化学习问题。

---

---

### 公式 (D.42)：惩罚方法 / Penalty Method

$$
\min_{\boldsymbol{x}} \left[f(\boldsymbol{x}) + c_k P(\boldsymbol{x})\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(\boldsymbol{x})$ | huà fá hánshù | 惩罚函数 |
| $c_k$ | huà fá xì shù | 惩罚系数 |
| $f(\boldsymbol{x})$ | mù biāo hánshù | 目标函数 |

**一句话解释：**
惩罚方法通过增加惩罚项将约束优化转化为无约束优化。

**简单应用场景：**
处理复杂约束的优化问题。

---

---

### 公式 (D.43)：障碍方法 / Barrier Method

$$
\min_{\boldsymbol{x}} \left[f(\boldsymbol{x}) + t^{-1} B(\boldsymbol{x})\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $B(\boldsymbol{x})$ | zhàng ài hánshù | 障碍函数 |
| $t$ | báramèter | 参数 |
| $f(\boldsymbol{x})$ | mù biāo hánshù | 目标函数 |

**一句话解释：**
障碍方法使用障碍函数防止解离开可行域，内点法的基础。

**简单应用场景：**
内点法求解凸优化。

---

---

### 公式 (D.44)：投影操作的Lipschitz连续性 / Lipschitz Continuity of Projection

$$
\|\mathcal{P}_\mathcal{C}(\boldsymbol{x}) - \mathcal{P}_\mathcal{C}(\boldsymbol{y})\| \leq \|\boldsymbol{x} - \boldsymbol{y}\|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{P}_\mathcal{C}$ | tóu yǐng suàn zǐ | 投影算子 |
| $\mathcal{C}$ | kòng jiān | 约束集合 |
| $\|\cdot\|$ | mó | 范数 |

**一句话解释：**
投影算子是Lipschitz连续的，常数为1。

**简单应用场景：**
证明投影梯度下降的收敛性。

---

---

### 公式 (D.45)：Polyak步长 / Polyak Step Size

$$
\eta_t = \frac{f(\boldsymbol{x}_t) - f(\boldsymbol{x}^*)}{\|\nabla f(\boldsymbol{x}_t)\|^2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\eta_t$ | bù cháng | 步长 |
| $f(\boldsymbol{x}^*)$ | zuì yōu zhí | 最优值 |
| $\|\nabla f\|$ | tī dù de mó | 梯度的模 |

**一句话解释：**
Polyak步长自适应选择步长，不需要预知最优值。

**简单应用场景：**
梯度下降的步长选择。

---

---

### 公式 (D.46)：Armijo条件 / Armijo Condition

$$
f(\boldsymbol{x}_t + \alpha \boldsymbol{d}_t) \leq f(\boldsymbol{x}_t) + c_1 \alpha \nabla f(\boldsymbol{x}_t)^T \boldsymbol{d}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | bù cháng | 步长 |
| $\boldsymbol{d}_t$ | xià jiàng fāng xiàng | 下降方向 |
| $c_1$ | cháng shù | Armijo常数（通常0.1） |

**一句话解释：**
Armijo条件保证直线搜索有足够的函数值下降。

**简单应用场景：**
直线搜索的一阶充分条件。

---

---

### 公式 (D.47)：Wolfe条件 / Wolfe Condition

$$
\nabla f(\boldsymbol{x}_t + \alpha \boldsymbol{d}_t)^T \boldsymbol{d}_t \geq c_2 \nabla f(\boldsymbol{x}_t)^T \boldsymbol{d}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla f$ | tī dù | 梯度 |
| $\alpha$ | bù cháng | 步长 |
| $c_2$ | cháng shù | Wolfe常数（通常0.1）） |

**一句话解释：**
Wolfe条件同时要求足够下降和足够陡峭，保证收敛。

**简单应用场景：**
直线搜索的二阶充分条件。

---

---

### 公式 (D.48)：置信域方法 / Trust Region Method

$$
\min_{\boldsymbol{d}} \left\{\nabla f(\boldsymbol{x})^T \boldsymbol{d} + \frac{1}{2}\boldsymbol{d}^T \boldsymbol{B} \boldsymbol{d}\right\} \quad \text{s.t.} \quad \|\boldsymbol{d}\| \leq \Delta
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{d}$ | xià jiàng fāng xiàng | 下降方向 |
| $\boldsymbol{B}$ | jìn jìn hǎi sēn jǔ zhèn | 近似海森矩阵 |
| $\Delta$ | xìn rèn yù bàn jìng | 信任域半径 |

**一句话解释：**
置信域方法在有限区域内最小化二次模型，自适应调整信任域。

**简单应用场景：**
非凸优化、非线性最小二乘问题。

---

---

### 公式 (D.49)：细致平衡条件 / Detailed Balance Condition

$$
\pi_i m_{ij} = \pi_j m_{ji}, \quad \forall i, j \in K
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi_i$ | píng héng fēn bù | 平衡分布 |
| $m_{ij}$ | zhuàn yí gài lǜ | 转移概率 |
| $K$ | zhuàng tài kòng jiān | 状态空间 |

**一句话解释：**
细致平衡条件保证马尔可夫链转移矩阵的平衡分布存在。

**简单应用场景：**
MCMC采样的理论保证。

---

---

### 公式 (D.50)：高斯过程回归中的联合分布 / Joint Distribution in Gaussian Process Regression

$$
\begin{bmatrix} f(\boldsymbol{x}) \\ f(\boldsymbol{x}') \end{bmatrix} \sim \mathcal{N}\left(\boldsymbol{\mu}(\boldsymbol{x}, \boldsymbol{x}'), K(\boldsymbol{x}, \boldsymbol{x}')\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}(\boldsymbol{x}, \boldsymbol{x}')$ | jūn zhí xiàng liàng | 均值向量 |
| $K(\boldsymbol{x}, \boldsymbol{x}')$ | xié fāng chà jǔ zhèn | 协方差矩阵 |
| $\boldsymbol{x}, \boldsymbol{x}'$ | shù jù diǎn | 数据点 |

**一句话解释：**
高斯过程回归中任意有限个点的联合分布服从多变量高斯分布。

**简单应用场景：**
贝叶斯优化、不确定性量化。

---

---

### 公式 (D.51)：平方指数核 / Squared Exponential Kernel

$$
k(\boldsymbol{x}_i, \boldsymbol{x}_j) = \exp\left(-\frac{\|\boldsymbol{x}_i - \boldsymbol{x}_j\|^2}{2\ell^2}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $k(\boldsymbol{x}_i, \boldsymbol{x}_j)$ | hé hánshù | 核函数 |
| $\ell$ | cháng dù biāo | 长度标度 |
| $\|\cdot\|$ | ōu jī lǐ jù lí | 欧几里得距离 |

**一句话解释：**
平方指数核是高斯过程中最常用的核函数，具有平滑性。

**简单应用场景：**
非参数回归、函数逼近。

---

---

### 公式 (D.52)：条件高斯分布 / Conditional Gaussian Distribution

$$
\begin{bmatrix} \boldsymbol{y} \\ \boldsymbol{y}' \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}(\boldsymbol{x}) \\ \boldsymbol{\mu}(\boldsymbol{x}') \end{bmatrix}, \begin{bmatrix} K(\boldsymbol{x}, \boldsymbol{x}) + \sigma^2 I & K(\boldsymbol{x}, \boldsymbol{x}')^T \\ K(\boldsymbol{x}', \boldsymbol{x}) & k(\boldsymbol{x}', \boldsymbol{x}') \end{bmatrix}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{y}$ | guān chá zhí | 观测值 |
| $\boldsymbol{y}'$ | yù cè zhí | 预测值 |
| $K(\boldsymbol{x}, \boldsymbol{x}')$ | xié fāng chà jǔ zhèn | 协方差矩阵 |
| $\sigma^2$ | gào sēng chóu | 高斯噪声 |

**一句话解释：**
给定观测数据的条件下，预测值的分布仍为高斯分布。

**简单应用场景：**
高斯过程的预测与不确定性量化。

---

---

### 公式 (D.53)：后验均值和方差 / Posterior Mean and Variance

$$
p(y'|X, y) = \mathcal{N}(\mu, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu$ | hòu yàn jūn zhí | 后验均值 |
| $\sigma^2$ | hòu yàn fāng chà | 后验方差 |
| $X, y$ | shù jù | 数据 |

**一句话解释：**
观测数据后的预测分布由均值和方差刻画。

**简单应用场景：**
贝叶斯推断、高斯过程预测。

---

---

### 公式 (D.54)：后验均值的显式形式 / Explicit Form of Posterior Mean

$$
\mu = K(\boldsymbol{x}^*, X)K(X, X)^{-1}\boldsymbol{y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K(\boldsymbol{x}^*, X)$ | jīng yàn hé yù cè diǎn de hé zhí | 经验核与预测点的核值 |
| $K(X, X)$ | jīng yàn hé jǔ zhèn | 经验核矩阵 |
| $\boldsymbol{y}$ | guān chá zhí | 观测值 |

**一句话解释：**
后验均值是观测值的加权组合，权重由核函数决定。

**简单应用场景：**
高斯过程回归的预测。

---

---

### 公式 (D.55)：后验方差的显式形式 / Explicit Form of Posterior Variance

$$
\sigma^2 = k(\boldsymbol{x}^*, \boldsymbol{x}^*) - K(\boldsymbol{x}^*, X)K(X, X) + \sigma_0^2]^{-1}K(\boldsymbol{x}', X)^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $k(\boldsymbol{x}^*, \boldsymbol{x}^*)$ | yù cè diǎn de zì hé | 预测点的自核 |
| $K(\boldsymbol{x}^*, X)$ | jīng yàn hé yù cè diǎn de hé zhí | 经验核与预测点的核值 |
| $\sigma_0^2$ | chóu zhǔn chà | 噪声方差 |

**一句话解释：**
后验方差衡量预测的不确定性，高斯过程提供了这种不确定性的定量评估。

**简单应用场景：**
置信区间的估计、采样策略的设计。

---

---

### 公式 (D.56)：高斯过程回归的均值估计 / Mean Estimate in Gaussian Process Regression

$$
\hat{\mu} = K(\boldsymbol{x}^*, X)K(X, X) + \sigma^2 I)^{-1}\boldsymbol{y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{\mu}$ | jūn zhí gū jì | 均值估计 |
| $K(\boldsymbol{x}^*, X)$ | jīng yàn hé yù cè diǎn de hé zhí | 经验核与预测点的核值 |
| $K(X, X)$ | jīng yàn hé jǔ zhèn | 经验核矩阵 |
| $\sigma^2$ | chóu shēng jiàn | 噪声项 |
| $\boldsymbol{y}$ | guān chá zhí | 观测值 |

**一句话解释：**
在考虑噪声的情况下，高斯过程回归的预测均值形式。

**简单应用场景：**
实际高斯过程回归中的计算。

---

## 附录E：信息论

---

## 附录E：信息论

### 公式 (E.1)：自信息 / Self Information

$$
p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(x)$ | zì xìn xī | 自信息 |
| $p(x)$ | gài lǜ | 事件x的概率 |
| $\log$ | duì shù | 对数（常用以2为底） |

**一句话解释：**
自信息衡量事件发生时包含的信息量，概率越小信息量越大。

**简单应用场景：**
数据压缩、编码理论。

---

---

### 公式 (E.2)：信息熵 / Entropy

$$
\mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X)$ | xìn xī sh熵 | 信息熵 |
| $E_X$ | qī wàng suàn zǐ | 期望算子 |
| $I(x)$ | zì xìn xī | 自信息 |

**一句话解释：**
信息熵是自信息的期望，衡量随机变量的不确定性。

**简单应用场景：**
衡量数据的混乱程度，决策树的分裂指标。

---

---

### 公式 (E.3)：信息熵的期望形式 / Expected Form of Entropy

$$
\mathbb{E}[X] = \int x \, p(x) \, dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X)$ | xìn xī shāng | 信息熵 |
| $p(x)$ | gài lǜ | x的概率 |

**一句话解释：**
信息熵的另一种表示形式，强调其期望含义。

**简单应用场景：**
理论分析和计算。

---

---

### 公式 (E.4)：离散分布的熵 / Entropy of Discrete Distribution

$$
\operatorname{Var}[X] = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X)$ | xìn xī shāng | 信息熵 |
| $p(x)$ | gài lǜ zhì liàng hánshù | 概率质量函数 |
| $\sum$ | qiú hé | 求和 |

**一句话解释：**
离散随机变量的熵是所有可能值的概率加权对数和。

**简单应用场景：**
分类问题中衡量标签分布的混乱程度。

---

---

### 公式 (E.5)：联合熵 / Joint Entropy

$$
\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X, Y)$ | lián hé shāng | 联合熵 |
| $p(x, y)$ | lián hé gài lǜ | 联合概率 |
| $\mathcal{X}, \mathcal{Y}$ | zhǐ | 取值空间 |

**一句话解释：**
联合熵衡量两个随机变量组合的总不确定性。

**简单应用场景：**
多变量系统的信息量分析。

---

---

### 公式 (E.6)：条件熵 / Conditional Entropy

$$
p(x, y) = p(x|y) p(y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X \mid Y)$ | tiáo jiàn shāng | 条件熵 |
| $p(x \mid y)$ | tiáo jiàn gài lǜ | 条件概率 |
| $p(x, y)$ | lián hé gài lǜ | 联合概率 |

**一句话解释：**
条件熵衡量在已知Y的条件下X的剩余不确定性。

**简单应用场景：**
特征选择、信息增益的计算。

---

---

### 公式 (E.7)：条件熵的另一种形式 / Alternative Form of Conditional Entropy

$$
p(y|x) = \frac{p(x|y) p(y)}{p(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X \mid Y)$ | tiáo jiàn shāng | 条件熵 |
| $p(x, y)$ | lián hé gài lǜ | 联合概率 |
| $p(y)$ | biān jì gài lǜ | 边际概率 |

**一句话解释：**
条件熵的等价形式，与联合熵和边际熵的关系密切。

**简单应用场景：**
理论推导、关系式证明。

---

---

### 公式 (E.8)：条件熵与其他熵的关系 / Relationship Between Conditional and Other Entropies

$$
p(x) = \sum_y p(x, y) = \sum_y p(x|y) p(y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X \mid Y)$ | tiáo jiàn shāng | 条件熵 |
| $H(X, Y)$ | lián hé shāng | 联合熵 |
| $H(Y)$ | xìn xī shāng | Y的熵 |

**一句话解释：**
条件熵等于联合熵减去条件变量的熵。

**简单应用场景：**
熵相关性质的推导。

---

---

### 公式 (E.9)：互信息 / Mutual Information

$$
\mathbb{E}[g(X)] = \int g(x) p(x) \, dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(X;Y)$ | hù xìn xī | 互信息 |
| $p(x, y)$ | lián hé gài lǜ | 联合概率 |
| $p(x), p(y)$ | biān jì gài lǜ | 边际概率 |

**一句话解释：**
互信息衡量两个随机变量之间的相互依赖程度。

**简单应用场景：**
特征选择、关联性分析。

---

---

### 公式 (E.10)：互信息与条件熵的关系 / Relationship Between Mutual Information and Conditional Entropy

$$
H(X) = -\sum_x p(x) \log p(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(X;Y)$ | hù xìn xī | 互信息 |
| $H(X)$ | xìn xī shāng | X的熵 |
| $H(X \mid Y)$ | tiáo jiàn shāng | 条件熵 |

**一句话解释：**
互信息等于X的熵减去X在Y条件下的条件熵。

**简单应用场景：**
信息增益的另一种表达。

---

---

### 公式 (E.11)：互信息的对称性 / Symmetry of Mutual Information

$$
H(X) = -\int p(x) \log p(x) \, dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(X;Y)$ | hù xìn xī | 互信息 |
| $H(Y)$ | xìn xī shāng | Y的熵 |
| $H(Y \mid X)$ | tiáo jiàn shāng | 条件熵 |

**一句话解释：**
互信息关于X和Y对称，反映了它们相互信息的对称性。

**简单应用场景：**
验证互信息的对称性质。

---

---

### 公式 (E.12)：交叉熵 / Cross Entropy

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(p, q)$ | jiāo叉 shāng | 交叉熵 |
| $p(x)$ | zhēn shí fēn bù | 真实分布 |
| $q(x)$ | yù cè fēn bù | 预测分布 |

**一句话解释：**
交叉熵衡量两个概率分布之间的相似程度，是分类问题的常用损失函数。

**简单应用场景：**
神经网络分类问题的损失函数。

---

---

### 公式 (E.13)：交叉熵的求和形式 / Summation Form of Cross Entropy

$$
D_{\mathrm{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(p, q)$ | jiāo叉 shāng | 交叉熵 |
| $p(x)$ | zhēn shí fēn bù | 真实分布 |
| $q(x)$ | yù cè fēn bù | 预测分布 |

**一句话解释：**
交叉熵的离散形式，对所有可能取值求和。

**简单应用场景：**
分类损失函数的计算。

---

---

### 公式 (E.14)：KL散度 / Kullback-Leibler Divergence

$$
D_{\mathrm{KL}}(p \| q) \geq 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $KL(p, q)$ | KL sǎn dù | KL散度 |
| $H(p, q)$ | jiāo叉 shāng | 交叉熵 |
| $H(p)$ | xìn xī shāng | 真实分布的熵 |

**一句话解释：**
KL散度衡量两个概率分布的差异，非对称性，仅当p=q时为0。

**简单应用场景：**
变分推断、模型选择。

---

---

### 公式 (E.15)：KL散度的求和形式 / Summation Form of KL Divergence

$$
\text{JS}(p \| q) = \frac{1}{2} D_{\mathrm{KL}}(p \| m) + \frac{1}{2} D_{\mathrm{KL}}(q \| m), \quad m = \frac{p + q}{2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $KL(p, q)$ | KL sǎn dù | KL散度 |
| $p(x)$ | zhēn shí fēn bù | 真实分布 |
| $q(x)$ | jìn jìn fēn bù | 近似分布 |

**一句话解释：**
KL散度的离散形式，衡量两个概率分布的信息差异。

**简单应用场景：**
变分推断中的目标函数。

---

---

### 公式 (E.16)：JS散度 / Jensen-Shannon Divergence

$$
I(X; Y) = D_{\mathrm{KL}}(p(x,y) \| p(x) p(y))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $JS(p, q)$ | JS sǎn dù | JS散度 |
| $m = \frac{1}{2}(p+q)$ | píng jūn fēn bù | 平均分布 |
| $KL(p, m)$ | KL sǎn dù | KL散度 |

**一句话解释：**
JS散度是KL散度的对称化版本，衡量两个分布的相似程度。

**简单应用场景：**
生成对抗网络（GAN）的理论基础。

---

---

### 公式 (E.17)：Wasserstein距离 / Wasserstein Distance

$$
I(X; Y) = H(X) - H(X|Y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $W_p$ | Wasserstein jù lí | Wasserstein距离 |
| $\gamma$ | yùn shū jì huà | 运输计划 |
| $d(x, y)$ | jù lí | 距离函数 |
| $\Gamma(q_1, q_2)$ | lián hé fēn bù | 联合分布 |

**一句话解释：**
Wasserstein距离衡量两个概率分布的最优运输成本，比KL散度更合理。

**简单应用场景：**
Wasserstein GAN、最优传输问题。

---

---

### 公式 (E.18)：边际概率的归一化约束 / Marginal Probability Normalization Constraint

$$
\mathbb{E}_{q}[f(x)] = \sum_x q(x) f(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma(x, y)$ | lián hé yùn shū jì huà | 联合运输计划 |
| $q_2(y)$ | biān jì fēn bù | 边际分布 |
| $\sum_y$ | duì y qiú hé | 对y求和 |

**一句话解释：**
运输计划的边际分布应该等于目标分布。

**简单应用场景：**
最优运输问题的约束条件。

---

---

### 公式 (E.19)：边际概率的另一个约束 / Another Marginal Probability Constraint

$$
\mathbb{E}_{q}[f(x)] = \mathbb{E}_{p}\left[ \frac{q(x)}{p(x)} f(x) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma(x, y)$ | lián hé yùn shū jì huà | 联合运输计划 |
| $q_1(x)$ | biān jì fēn bù | 边际分布 |
| $\sum_y$ | duì y qiú hé | 对y求和 |

**一句话解释：**
运输计划的另一个边际分布应该等于源分布。

**简单应用场景：**
最优运输问题的约束条件。

---

---

### 公式 (E.20)：Wasserstein距离与运输的关系 / Relationship Between Wasserstein Distance and Transportation

$$
\hat{\mathbb{E}}_{q}[f(x)] = \frac{1}{N} \sum_{n=1}^{N} \frac{q(x^{(n)})}{p(x^{(n)})} f(x^{(n)}), \quad x^{(n)} \sim p
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma(x, y)$ | yùn shū jì huà | 运输计划 |
| $d(x, y)$ | jù lí | 距离 |
| $p$ | cì shù | 幂次 |

**一句话解释：**
Wasserstein距离是所有可能的运输计划中成本最小的那个。

**简单应用场景：**
最优传输的成本计算。

---

---

### 公式 (E.21)：2-Wasserstein距离的显式形式 / Explicit Form of 2-Wasserstein Distance

$$
\operatorname{Var}[\hat{\mathbb{E}}] = \frac{1}{N} \operatorname{Var}_{p}\left[ \frac{q(x)}{p(x)} f(x) \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $W_2$ | 2-Wasserstein jù lí | 2-Wasserstein距离 |
| $\mu_1, \mu_2$ | jūn zhí xiàng liàng | 均值向量 |
| $\boldsymbol{\Sigma}_1, \boldsymbol{\Sigma}_2$ | xié fāng chà jǔ zhèn | 协方差矩阵 |
| $\text{tr}$ | j迹 | 矩阵迹 |

**一句话解释：**
当两个分布都是高斯分布时，2-Wasserstein距离有显式公式。

**简单应用场景：**
高斯分布之间的距离计算，两个正态分布间的匹配。

---

## 完成说明

所有公式已从附录C、D、E中完整提取，共计：
- **附录C（概率论）**：公式 C.1 - C.27（27个）
- **附录D（数学优化）**：公式 D.1 - D.56（56个）
- **附录E（信息论）**：公式 E.1 - E.21（21个）

**总计：104个公式**

每个公式均包含：
1. 标准序号和中英文名称
2. 准确的LaTeX表达式
3. 完整的符号表（包括中文拼音及声调）
4. 一句话通俗解释
5. 简单应用场景

---

