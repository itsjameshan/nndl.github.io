# 《神经网络与深度学习》全书公式提取

> 本文档从邱锡鹏著《神经网络与深度学习》一书中提取了从公式 (2.1) 到公式 (E.21) 的所有编号公式。
> 每个公式包含：LaTeX表达式、符号表（含汉语拼音读法）、一句话解释、简单应用场景。

---

## 第2章：机器学习概述

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

### 公式 (2.3)：标签的条件概率 / Conditional Probability of Label

$$
p(y|\boldsymbol{x}) = f_c(\boldsymbol{x};\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y\|\boldsymbol{x})$ | gài lǜ | 给定输入x下标签y的条件概率 |
| $f_c$ | hánshù | 条件概率函数 |
| $\boldsymbol{\theta}$ | xī tǎ | 模型参数 |

**一句话解释：**
在给定输入特征的条件下，标签的条件概率可以由参数化函数表示。

**简单应用场景：**
在垃圾邮件分类中，计算给定邮件内容下该邮件是垃圾邮件的概率。

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

### 公式 (2.6)：线性模型 / Linear Model

$$
f(\boldsymbol{x};\boldsymbol{\omega}) = \boldsymbol{\omega}^T \boldsymbol{x} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x};\boldsymbol{\omega})$ | hánshù | 线性模型 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |
| $b$ | piàn zhì | 偏置项 |
| $\boldsymbol{x}$ | xiàng liàng | 输入特征向量 |
| $\boldsymbol{\omega}^T$ | zhuǎnzhì | 权重向量的转置 |

**一句话解释：**
线性模型是输入特征的线性组合加上偏置项。

**简单应用场景：**
在房价预测中，用房屋特征的线性组合预测房价。

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

### 公式 (2.8)：多层非线性模型 / Multi-layer Nonlinear Model

$$
\phi_k(\boldsymbol{x}) = h(\boldsymbol{\omega}_k^T \boldsymbol{\phi}(\boldsymbol{x}) + b_k), \quad \forall 1 \le k \le K
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\phi_k(\boldsymbol{x})$ | fài | 第k层的输出特征 |
| $h(\cdot)$ | hánshù | 非线性激活函数 |
| $\boldsymbol{\omega}_k$ | ōmǐ gā | 第k层的权重向量 |
| $b_k$ | piàn zhì | 第k层的偏置项 |
| $K$ | dà xiě kāi | 层数 |

**一句话解释：**
多层非线性模型通过多次应用非线性变换逐步提取更抽象的特征。

**简单应用场景：**
深度神经网络中，多层隐层通过非线性激活函数学习数据的分层表示。

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

### 公式 (2.10)：经验误差与真实条件概率 / Empirical Error and True Conditional Probability

$$
|f_r(\boldsymbol{x}, \boldsymbol{\theta}^*) - p_r(y|\boldsymbol{x})| < \varepsilon, \quad \forall(\boldsymbol{x}, y) \in \mathcal{X} \times \mathcal{Y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_r(\boldsymbol{x}, \boldsymbol{\theta}^*)$ | hánshù | 学习到的条件概率 |
| $p_r(y\|\boldsymbol{x})$ | gài lǜ | 真实的条件概率 |
| $\varepsilon$ | yì pǔ xī lóng | 很小的正数 |
| $\mathcal{X} \times \mathcal{Y}$ | chéngenjī | 样本空间 |

**一句话解释：**
要求学到的条件概率与真实条件概率的差异足够小。

**简单应用场景：**
概率分类模型应该学到与真实数据分布相近的条件概率。

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

### 公式 (2.15)：交叉熵损失函数概率表示 / Cross-Entropy Loss Function - Probability Form

$$
p(y \mid \boldsymbol{x};\boldsymbol{\theta}) = f_c(\boldsymbol{x};\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y\|\boldsymbol{x};\boldsymbol{\theta})$ | gài lǜ | 给定输入和参数下标签的条件概率 |
| $f_c(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 条件概率函数 |

**一句话解释：**
交叉熵损失基于模型学到的条件概率分布。

**简单应用场景：**
在神经网络中使用softmax输出作为类别概率。

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

### 公式 (2.21)：Hinge损失的绝对值形式 / Hinge Loss - Absolute Value Form

$$
\mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta})) = [1 - yf(\boldsymbol{x};\boldsymbol{\theta})]_+
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $[\cdot]_+$ | jǔ | hinge函数，定义为$\max(0, \cdot)$ |
| $y$ | wài | 标签（取值为-1或1） |
| $f(\boldsymbol{x};\boldsymbol{\theta})$ | hánshù | 模型的预测输出 |

**一句话解释：**
使用下标+表示取正部分，与公式(2.20)等价。

**简单应用场景：**
SVM中的标准损失函数表示方法。

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

### 公式 (2.25)：带正则化项的结构风险 / Regularized Risk

$$
\mathcal{R}_{\mathcal{D}}^{struct}(\boldsymbol{\theta}) = \mathcal{R}_{\mathcal{D}}^{emp}(\boldsymbol{\theta}) + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_{\mathcal{D}}^{struct}$ | jié gòu fēng xiǎn | 结构风险 |
| $\mathcal{R}_{\mathcal{D}}^{emp}$ | jīng yàn fēng xiǎn | 经验风险 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\|\boldsymbol{\theta}\|^2$ | fàn shù | 参数的L2范数平方 |

**一句话解释：**
结构风险由经验风险和L2正则化项组成，通过惩罚模型复杂度防止过拟合。

**简单应用场景：**
岭回归中的目标函数。

---

### 公式 (2.26)：完整的正则化优化目标 / Complete Regularized Objective

$$
\mathcal{R}_{\mathcal{D}}^{struct}(\boldsymbol{\theta}) = \arg \min_{\boldsymbol{\theta}} \frac{1}{N} \sum_{n=1}^N \mathcal{L}(y^{(n)}, f(\boldsymbol{x}^{(n)};\boldsymbol{\theta})) + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_{\mathcal{D}}^{struct}$ | jié gòu fēng xiǎn | 结构风险 |
| $N$ | ēn | 训练样本总数 |
| $\mathcal{L}$ | sǔnshī | 损失函数 |
| $\lambda$ | lán mǔ dá | 正则化系数 |
| $\boldsymbol{\theta}$ | xī tǎ | 模型参数 |

**一句话解释：**
完整的优化目标包含平均经验损失和正则化惩罚项。

**简单应用场景：**
实际机器学习中最常见的训练目标函数。

---

### 公式 (2.27)：梯度下降法的参数更新 / Gradient Descent Parameter Update

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \frac{\partial \mathcal{R}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}_{t+1}$ | xī tǎ | 第t+1次迭代的参数 |
| $\boldsymbol{\theta}_t$ | xī tǎ | 第t次迭代的参数 |
| $\alpha$ | āl fǎ | 学习率 |
| $\frac{\partial \mathcal{R}}{\partial \boldsymbol{\theta}}$ | piān dǎo | 风险函数对参数的梯度 |

**一句话解释：**
梯度下降法沿着负梯度方向更新参数，步长由学习率控制。

**简单应用场景：**
神经网络训练中最基本的优化算法。

---

### 公式 (2.28)：随机梯度下降的参数更新 / Stochastic Gradient Descent Update

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \alpha \frac{1}{N} \sum_{n=1}^N \frac{\partial \mathcal{L}(y^{(n)}, f(\boldsymbol{x}^{(n)};\boldsymbol{\theta}))}{\partial \boldsymbol{\theta}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}_t$ | xī tǎ | 第t次迭代的参数 |
| $\boldsymbol{\theta}_{t-1}$ | xī tǎ | 第t-1次迭代的参数 |
| $\alpha$ | āl fǎ | 学习率 |
| $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}$ | piān dǎo | 损失函数对参数的梯度 |
| $N$ | ēn | 批次大小 |

**一句话解释：**
随机梯度下降使用样本的平均梯度更新参数，比完全梯度下降更高效。

**简单应用场景：**
深度学习中的标准训练算法。

---

### 公式 (2.29)：小批量梯度下降 / Mini-Batch Gradient Descent

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \frac{1}{S} \sum_{(\boldsymbol{x},y) \in \mathcal{S}_t} \frac{\partial \mathcal{L}(y, f(\boldsymbol{x};\boldsymbol{\theta}))}{\partial \boldsymbol{\theta}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}_{t+1}$ | xī tǎ | 第t+1次迭代的参数 |
| $\boldsymbol{\theta}_t$ | xī tǎ | 第t次迭代的参数 |
| $\alpha$ | āl fǎ | 学习率 |
| $\mathcal{S}_t$ | píngjiàn | 第t个小批量 |
| $S$ | ēs | 小批量大小 |

**一句话解释：**
小批量梯度下降使用一小批样本的平均梯度更新参数，平衡效率和稳定性。

**简单应用场景：**
现代深度学习框架中最常用的优化方法。

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

### 公式 (2.32)：扩展的特征向量 / Extended Feature Vector

$$
\tilde{\boldsymbol{x}} = \boldsymbol{x} \oplus \mathbf{1} = \begin{bmatrix} x \\ x_1 \\ \vdots \\ x_D \\ 1 \end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{\boldsymbol{x}}$ | xiàng liàng | 扩展后的特征向量 |
| $\boldsymbol{x}$ | xiàng liàng | 原始特征向量 |
| $\oplus$ | zhí jiē hé | 直接和运算 |
| $\mathbf{1}$ | yī | 值为1的列向量 |

**一句话解释：**
在原始特征向量后面追加1，用于隐含表示偏置项。

**简单应用场景：**
增广特征向量的构造方法。

---

### 公式 (2.33)：扩展的权重向量 / Extended Weight Vector

$$
\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} \oplus b = \begin{bmatrix} \boldsymbol{\omega} \\ b \end{bmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tilde{\boldsymbol{\omega}}$ | ōmǐ gā | 扩展后的权重向量 |
| $\boldsymbol{\omega}$ | ōmǐ gā | 原始权重向量 |
| $\oplus$ | zhí jiē hé | 直接和运算 |
| $b$ | piàn zhì | 偏置项 |

**一句话解释：**
在原始权重向量后追加偏置项，对应扩展特征向量。

**简单应用场景：**
增广权重向量的构造方法。

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

### 公式 (2.42)：梯度下降的迭代更新（最小均方） / Gradient Descent Update - Least Mean Squares

$$
\boldsymbol{\omega} \leftarrow \boldsymbol{\omega} + \alpha(\boldsymbol{y} - \boldsymbol{X}^T\boldsymbol{\omega})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}$ | ōmǐ gā | 权重向量 |
| $\alpha$ | āl fǎ | 学习率 |
| $\boldsymbol{y}$ | xiàng liàng | 标签向量 |
| $\boldsymbol{X}$ | jǔ zhèn | 特征矩阵 |

**一句话解释：**
最小均方法通过沿梯度反方向更新权重，其中梯度与预测误差成正比。

**简单应用场景：**
自适应信号处理中的经典算法。

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

### 公式 (2.47)：线性回归的概率分布 / Probabilistic Distribution of Linear Regression

$$
p(y|\boldsymbol{x}; \boldsymbol{\omega}, \sigma) = \mathcal{N}(y; \boldsymbol{\omega}^\top \boldsymbol{x}, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y\|\boldsymbol{x}; \boldsymbol{\omega}, \sigma)$ | gài lǜ | 给定输入和参数的输出概率分布 |
| $\mathcal{N}$ | zhèng tài | 高斯/正态分布 |
| $\boldsymbol{\omega}^\top \boldsymbol{x}$ | jūn zhī | 均值/期望 |
| $\sigma^2$ | fāng chā | 方差 |
| $\sigma$ | xī gé mǎ | 标准差 |

**一句话解释：**
线性回归的输出服从以线性预测值为均值、标准差为σ的高斯分布。

**简单应用场景：**
在贝叶斯线性回归中，为参数估计提供概率框架，允许量化预测的不确定性。

---

### 公式 (2.48)：高斯分布的概率密度函数 / Gaussian Distribution Probability Density Function

$$
= \frac{1}{\sqrt{2\pi\sigma}} \exp\left( -\frac{(y - \boldsymbol{\omega}^\top \boldsymbol{x})^2}{2\sigma^2} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp$ | zhǐ shù | 指数函数 |
| $(y - \boldsymbol{\omega}^\top \boldsymbol{x})^2$ | píng fāng | 平方偏差 |
| $2\pi\sigma$ | pài | 圆周率相关归一化常数 |

**一句话解释：**
高斯分布的概率密度函数表示输出值在均值附近的概率。

**简单应用场景：**
计算给定参数下观测数据的似然函数，用于最大似然估计或贝叶斯推断。

---

### 公式 (2.49)：似然函数（连乘形式）/ Likelihood Function (Product Form)

$$
p(\boldsymbol{y}|\mathbf{X}; \boldsymbol{\omega}, \sigma) = \prod_{n=1}^{N} \mu(y^{(n)}|\boldsymbol{x}^{(n)}; \boldsymbol{\omega}, \sigma)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{y}\|\mathbf{X}; \boldsymbol{\omega}, \sigma)$ | gài lǜ | 整个数据集的似然函数 |
| $\prod$ | chéng jī | 连乘积 |
| $N$ | dà xiě N | 样本数 |
| $\mu$ | miū | 单个样本的概率分布 |
| $y^{(n)}$ | dì n ge | 第n个样本的输出 |
| $\boldsymbol{x}^{(n)}$ | dì n ge | 第n个样本的输入 |

**一句话解释：**
整个数据集的似然函数是所有单个样本概率的乘积。

**简单应用场景：**
最大似然估计中，通过最大化该函数来学习最优的模型参数$\boldsymbol{\omega}$和$\sigma$。

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

### 公式 (2.52)：最大似然估计的参数 / Maximum Likelihood Estimation Parameters

$$
\boldsymbol{\omega}^{ML} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \boldsymbol{y}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^{ML}$ | zuì dà sì rán | 最大似然估计的权重 |
| $\mathbf{X}$ | dà xiě X | 设计矩阵（包含所有输入样本） |
| $\mathbf{X}^\top$ | zhuan zhì | 矩阵的转置 |
| $(\mathbf{X}^\top \mathbf{X})^{-1}$ | nì | 矩阵求逆 |
| $\boldsymbol{y}$ | duō wéi xiàng liàng | 所有输出的向量 |

**一句话解释：**
线性回归的最大似然估计解析解，通过矩阵运算直接求得最优权重。

**简单应用场景：**
小规模线性回归问题中快速求解最优参数，无需迭代优化。

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

### 公式 (2.54)：后验分布 / Posterior Distribution

$$
p(\boldsymbol{w}|\mathbf{X}, \boldsymbol{y}, v, \sigma) = \frac{p(\boldsymbol{w}, \boldsymbol{y}|\mathbf{X}, v, \sigma)}{\sum_{\boldsymbol{w}} p(\boldsymbol{w}, \boldsymbol{y}|\mathbf{X}, v, \sigma)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{w}\|\mathbf{X}, \boldsymbol{y}, v, \sigma)$ | hòu yàn | 后验分布 |
| $p(\boldsymbol{w}, \boldsymbol{y}\|\mathbf{X}, v, \sigma)$ | lián hé gài lǜ | 联合概率 |
| $\sum_{\boldsymbol{w}}$ | biàn lì | 对所有权重求和（归一化） |

**一句话解释：**
根据观测数据更新权重的分布，结合先验和似然函数。

**简单应用场景：**
贝叶斯推断中计算权重的后验分布，用于不确定性量化。

---

### 公式 (2.55)：后验分布的简化形式 / Posterior Distribution Simplified

$$
\propto p(\boldsymbol{y}|\mathbf{X}, \boldsymbol{w}; \sigma) p(\boldsymbol{w}; v)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\propto$ | zhèng bǐ yú | 正比于 |
| $p(\boldsymbol{y}\|\mathbf{X}, \boldsymbol{w}; \sigma)$ | sì rán hánshù | 似然函数 |
| $p(\boldsymbol{w}; v)$ | xiān yàn | 先验分布 |

**一句话解释：**
后验分布正比于似然函数与先验分布的乘积，遵循贝叶斯定理。

**简单应用场景：**
贝叶斯线性回归中的核心计算，后验 ∝ 似然 × 先验。

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
| $p(\boldsymbol{y}\|\mathbf{X}, \boldsymbol{\omega}; \sigma)$ | sì rán hánshù | 似然函数 |
| $p(\boldsymbol{\omega}; v)$ | xiān yàn | 先验分布 |

**一句话解释：**
在贝叶斯框架中，找使后验分布最大的权重估计值。

**简单应用场景：**
含正则化的线性回归中，综合考虑数据拟合和先验约束的参数估计。

---

### 公式 (2.57)：对数后验 / Log Posterior

$$
\log p(\boldsymbol{w}|\mathbf{X}, \boldsymbol{y}, v, \sigma) \propto \log p(\boldsymbol{y}|\mathbf{X}, \boldsymbol{\omega}; \sigma) + \log p(\boldsymbol{\omega}; v)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log$ | duì shù | 自然对数 |
| $p(\boldsymbol{w}\|\mathbf{X}, \boldsymbol{y}, v, \sigma)$ | hòu yàn | 后验分布 |
| $p(\boldsymbol{y}\|\mathbf{X}, \boldsymbol{\omega}; \sigma)$ | sì rán hánshù | 似然函数 |
| $p(\boldsymbol{\omega}; v)$ | xiān yàn | 先验分布 |

**一句话解释：**
对数后验等于对数似然加上对数先验，便于计算和优化。

**简单应用场景：**
贝叶斯参数估计中使用对数域进行数值计算，提高稳定性。

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


## 第3章：线性模型

### 公式 (3.1)：线性组合 / Linear Combination

$$
f(\boldsymbol{x}; \boldsymbol{\omega}) = \omega_1 x_1 + \omega_2 x_2 + \cdots + \omega_D x_D + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | xiàngliàng | 输入特征向量 |
| $\omega_i$ | ōu mǐ gā xià biāo | 权重 |
| $b$ | bì | 偏置项 |
| $D$ | dà xiě dì | 特征维度 |

**一句话解释：**
线性组合是将输入特征的加权和加上偏置项。

**简单应用场景：**
用于房价预测，其中特征为房屋面积和房龄，权重分别代表每单位面积和房龄对房价的影响。

---

### 公式 (3.2)：线性模型紧凑表示 / Compact Linear Model

$$
f(\boldsymbol{x}; \boldsymbol{\omega}) = \boldsymbol{\omega}^T \boldsymbol{x} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^T$ | ōu mǐ gā zhuǎnzhì | 权重向量的转置 |
| $\boldsymbol{x}$ | xiàngliàng | 输入特征向量 |
| $b$ | bì | 偏置项 |

**一句话解释：**
线性模型可以用向量形式紧凑地表示为权重向量与特征向量的点积加上偏置项。

**简单应用场景：**
在矩阵运算和计算机实现中，这种向量形式便于批量计算多个样本的预测结果。

---

### 公式 (3.3)：判别函数结合决策函数 / Discriminant Function with Decision Function

$$
y = g(f(\boldsymbol{x}; \boldsymbol{\omega}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | 预测输出 |
| $g$ | jī | 决策函数 |
| $f(\boldsymbol{x}; \boldsymbol{\omega})$ | ài fǎ | 判别函数的值 |

**一句话解释：**
最终的预测输出是通过决策函数处理判别函数的值得到的。

**简单应用场景：**
在分类问题中，判别函数产生连续值，决策函数将其转换为离散的类别标签。

---

### 公式 (3.4)：符号函数定义 / Sign Function Definition

$$
g(f(\boldsymbol{x}; \boldsymbol{\omega})) = \text{sgn}(f(\boldsymbol{x}; \boldsymbol{\omega}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g$ | jī | 决策函数 |
| $\text{sgn}$ | fúhào hánshù | 符号函数 |
| $f(\boldsymbol{x}; \boldsymbol{\omega})$ | ài fǎ | 判别函数 |

**一句话解释：**
符号函数将判别函数的值映射到正负两个类别。

**简单应用场景：**
在二分类任务中，符号函数根据判别函数值的正负返回+1或-1标签。

---

### 公式 (3.5)：符号函数的分段表示 / Piecewise Definition of Sign Function

$$
g(f(\boldsymbol{x}; \boldsymbol{\omega})) = \begin{cases} +1 & \text{if } f(\boldsymbol{x}; \boldsymbol{\omega}) \geq 0 \\ -1 & \text{if } f(\boldsymbol{x}; \boldsymbol{\omega}) < 0 \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $+1$ | jiā yī | 正类标签 |
| $-1$ | fù yī | 负类标签 |
| $f(\boldsymbol{x}; \boldsymbol{\omega})$ | ài fǎ | 判别函数值 |

**一句话解释：**
符号函数根据判别函数值是否非负返回对应的类别标签。

**简单应用场景：**
在二分类问题中作为最基本的决策规则，判别函数值为0的样本位于决策边界上。

---

### 公式 (3.6)：二分类有符号距离 / Signed Distance in Binary Classification

$$
y = \frac{f(\boldsymbol{x}; \boldsymbol{\omega})}{\|\boldsymbol{\omega}\|}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | 有符号距离 |
| $f(\boldsymbol{x}; \boldsymbol{\omega})$ | ài fǎ | 判别函数值 |
| $\|\boldsymbol{\omega}\|$ | fànshù mó | 权重向量的范数 |

**一句话解释：**
有符号距离是判别函数值除以权重向量的范数，表示样本到决策边界的距离。

**简单应用场景：**
在支持向量机中用于衡量样本与决策边界的关系，正值表示在正侧，负值表示在负侧。

---

### 公式 (3.7)：二分类判别条件之一 / Binary Classification Condition 1

$$
f(\boldsymbol{x}^{(n)}; \boldsymbol{\omega}^*) > 0 \quad \text{if } y^{(n)} = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(\boldsymbol{x}^{(n)}; \boldsymbol{\omega}^*)$ | ài fǎ | 第n个样本的判别函数值 |
| $\boldsymbol{\omega}^*$ | zuì yōu quán zhòng | 最优权重向量 |
| $y^{(n)}$ | dì n gè yàngběn de lèibiē | 第n个样本的真实标签 |

**一句话解释：**
如果样本属于正类，其判别函数值应大于0。

**简单应用场景：**
在训练线性分类器时，要求所有正类样本的判别函数值都为正数。

---

### 公式 (3.8)：所有样本的判别条件 / Classification Condition for All Samples

$$
y^{(n)} f(\boldsymbol{x}^{(n)}; \boldsymbol{\omega}^*) > 0, \quad \forall n \in [1, N]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | dì n gè yàngběn de lèibiē | 第n个样本的真实标签 |
| $f(\boldsymbol{x}^{(n)}; \boldsymbol{\omega}^*)$ | ài fǎ | 第n个样本的判别函数值 |
| $N$ | dà xiě ài | 样本总数 |
| $\forall$ | duìyú suǒyǒu | 对于所有 |

**一句话解释：**
所有样本的真实标签与其判别函数值的乘积都必须为正，表示分类完全正确。

**简单应用场景：**
在感知机学习中作为收敛的判断条件，当所有样本都满足该条件时分类完成。

---

### 公式 (3.9)：二分类0-1损失函数 / Binary Classification 0-1 Loss Function

$$
\mathcal{L}_{01}(y, f(\boldsymbol{x}; \boldsymbol{\omega})) = \mathbb{I}(yf(\boldsymbol{x}; \boldsymbol{\omega}) < 0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{01}$ | sǔnshī hánshù | 0-1损失函数 |
| $y$ | wài | 真实标签 |
| $f(\boldsymbol{x}; \boldsymbol{\omega})$ | ài fǎ | 判别函数值 |
| $\mathbb{I}$ | zhǐshìhánshù | 指示函数 |

**一句话解释：**
当样本被错误分类时损失为1，否则为0。

**简单应用场景：**
作为分类准确率的直接衡量标准，但由于不可微性在实际优化中很少直接使用。

---

### 公式 (3.10)：多分类的一对其余方法 / One-vs-Rest Multi-class Classification

$$
f_c(\boldsymbol{x}, \boldsymbol{\omega}_c) = \boldsymbol{\omega}_c^T \boldsymbol{x} + b_c, \quad c \in \{1, \ldots, C\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_c$ | dì c gè pànbiē hánshù | 第c个类别的判别函数 |
| $\boldsymbol{\omega}_c$ | dì c gè quán zhòng xiàngliàng | 第c个类别的权重向量 |
| $c$ | lèibiē | 类别标签 |
| $C$ | lèibiē zǒngshù | 类别总数 |

**一句话解释：**
为每一个类别都训练一个判别函数，将多分类问题转化为多个二分类问题。

**简单应用场景：**
在手写数字识别中，为每个数字类别（0-9）各训练一个判别函数，共10个模型。

---

### 公式 (3.11)：多分类argmax预测 / Argmax Decision for Multi-class

$$
y = \arg\max_{c=1}^{C} f_c(\boldsymbol{x}, \boldsymbol{\omega}_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y$ | wài | 预测类别 |
| $\arg\max$ | zuì dà zhí duìyìng de cānshù | 返回最大值对应的参数 |
| $f_c$ | pànbiē hánshù | 第c个类别的判别函数值 |
| $C$ | lèibiē zǒngshù | 类别总数 |

**一句话解释：**
预测的类别是所有判别函数中输出值最大的那个类别。

**简单应用场景：**
在图像分类任务中，选择神经网络输出概率最高的类别作为最终预测结果。

---

### 公式 (3.12)：Logistic回归的条件概率模型 / Logistic Regression Conditional Probability Model

$$
p(y = 1|\boldsymbol{x}) = g(f(\boldsymbol{x}; \boldsymbol{\omega}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = 1\|\boldsymbol{x})$ | tiáojiàn gàilǜ | 给定输入x样本属于正类的条件概率 |
| $g$ | jī | 激活函数 |
| $f(\boldsymbol{x}; \boldsymbol{\omega})$ | ài fǎ | 判别函数值 |

**一句话解释：**
Logistic回归用激活函数将判别函数的输出转换为[0,1]区间的概率值。

**简单应用场景：**
在医学诊断中，用于估计患者患有某种疾病的概率。

---

### 公式 (3.13)：Logistic激活函数 / Logistic Activation Function

$$
p(y = 1|\boldsymbol{x}) = \sigma(\boldsymbol{\omega}^T \boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | Sigmoid函数 |
| $\boldsymbol{\omega}^T \boldsymbol{x}$ | ài fǎ | 线性组合 |

**一句话解释：**
Sigmoid函数是Logistic回归中常用的激活函数，将实数映射到(0,1)区间。

**简单应用场景：**
作为逻辑回归和神经网络中最常用的激活函数之一。

---

### 公式 (3.14)：Sigmoid函数标准定义 / Standard Sigmoid Function

$$
\sigma(\cdot) = \frac{1}{1 + \exp(-\boldsymbol{\omega}^T\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ hánshù | Sigmoid函数 |
| $\exp$ | zhǐ shù hánshù | 指数函数 |
| $\boldsymbol{\omega}^T\boldsymbol{x}$ | ài fǎ | 线性组合 |

**一句话解释：**
Sigmoid函数将任意实数映射到(0,1)区间，是一条S形曲线。

**简单应用场景：**
在神经网络中作为隐层和输出层的激活函数，实现非线性变换。

---

### 公式 (3.15)：负类概率 / Negative Class Probability

$$
p(y = 0|\boldsymbol{x}) = 1 - p(y = 1|\boldsymbol{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = 0\|\boldsymbol{x})$ | tiáojiàn gàilǜ | 给定x样本属于负类的条件概率 |
| $p(y = 1\|\boldsymbol{x})$ | tiáojiàn gàilǜ | 给定x样本属于正类的条件概率 |

**一句话解释：**
负类概率等于1减去正类概率，利用概率的互补性质。

**简单应用场景：**
在二分类问题中，一旦计算出正类概率，负类概率可直接得出。

---

### 公式 (3.16)：负类概率的代数表示 / Algebraic Form of Negative Class Probability

$$
p(y = 0|\boldsymbol{x}) = \frac{\exp(-\boldsymbol{\omega}^T \boldsymbol{x})}{1 + \exp(-\boldsymbol{\omega}^T \boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y = 0\|\boldsymbol{x})$ | tiáojiàn gàilǜ | 负类的条件概率 |
| $\exp$ | zhǐ shù hánshù | 指数函数 |
| $\boldsymbol{\omega}^T \boldsymbol{x}$ | ài fǎ | 线性组合 |

**一句话解释：**
负类概率也可以用指数函数形式表示，便于数学推导。

**简单应用场景：**
在推导Logistic回归的梯度时，这种形式便于化简。

---

### 公式 (3.17)：Logit变换（对数几率） / Logit Transformation (Log-odds)

$$
\boldsymbol{\omega}^T \boldsymbol{x} = \log \frac{p(y = 1|\boldsymbol{x})}{1 - p(y = 1|\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}^T \boldsymbol{x}$ | ài fǎ | 线性组合 |
| $\log$ | duìshù hánshù | 自然对数 |
| $p(y = 1\|\boldsymbol{x})$ | tiáojiàn gàilǜ | 正类条件概率 |

**一句话解释：**
Logit变换是Sigmoid函数的反函数，用于关联线性组合与概率。

**简单应用场景：**
在医学研究中，用对数几率模型研究风险因素与疾病的关联。

---

## 补充说明

以上为Chapter 3中从公式(3.1)到(3.17)的全部提取内容。这些公式涵盖了线性模型的核心内容，包括：

1. **线性模型基础**（3.1-3.2）：线性组合及其紧凑表示
2. **二分类线性模型**（3.3-3.9）：判别函数、决策函数、符号函数、损失函数
3. **多分类方法**（3.10-3.11）：一对其余方法和argmax决策
4. **Logistic回归**（3.12-3.17）：概率模型、Sigmoid函数、Logit变换

所有符号表均为自包含式，公式编号按照书籍中的原始编号。


---

### 公式 (3.18)：几率 / Odds Ratio

$$
\text{logit}(p) = \log\frac{p(y=1|\boldsymbol{x})}{p(y=0|\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y=1\|\boldsymbol{x})$ | gài lǜ | 样本为正例的后验概率 |
| $p(y=0\|\boldsymbol{x})$ | gài lǜ | 样本为反例的后验概率 |
| logit | luó jī tè | 对数几率函数 |

**一句话解释：**
对数几率是正反例后验概率的比值的对数，衡量样本倾向于正例的程度。

**简单应用场景：**
在逻辑回归中用于建立输入特征与二分类标签的非线性关系。

---

### 公式 (3.19)：逻辑回归模型 / Logistic Regression Model

$$
p(y=1|\boldsymbol{x}) = \sigma(\boldsymbol{w}^T\boldsymbol{x} + b)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma(\cdot)$ | xī gé mǎ | Sigmoid激活函数 |
| $\boldsymbol{w}$ | quán zhòng | 权重向量 |
| $b$ | piān zhì | 偏置项 |
| $\boldsymbol{x}$ | shū rù | 输入特征向量 |

**一句话解释：**
使用Sigmoid函数将线性组合映射到(0,1)区间，得到正例的概率。

**简单应用场景：**
二分类问题中预测样本属于正例的概率。

---

### 公式 (3.20)：风险函数 / Risk Function

$$
\mathcal{R}(\boldsymbol{w}) = \sum_{n=1}^{N} \log(1 + \exp(-y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(\boldsymbol{w})$ | fēng xiǎn hánshù | 风险函数 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |
| $N$ | yàng běn shù | 样本总数 |

**一句话解释：**
对所有样本的对数损失求和，衡量模型预测错误的总体成本。

**简单应用场景：**
逻辑回归的目标函数，需要通过优化最小化。

---

### 公式 (3.21)：标准化风险函数 / Normalized Risk Function

$$
\mathcal{R}(\boldsymbol{w}) = \frac{1}{N}\sum_{n=1}^{N} \log(1 + \exp(-y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{1}{N}$ | píng jūn | 平均因子 |
| $N$ | yàng běn shù | 样本总数 |

**一句话解释：**
将风险函数除以样本总数，得到平均每个样本的损失。

**简单应用场景：**
使目标函数与样本数量无关，便于不同规模数据集间的比较。

---

### 公式 (3.22)：对数损失函数 / Log Loss Function

$$
\mathcal{L}^{(n)} = \log(1 + \exp(-y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}^{(n)}$ | sǔn shī hánshù | 第n个样本的损失函数 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |

**一句话解释：**
单个样本的对数损失，衡量该样本预测的错误程度。

**简单应用场景：**
逻辑回归中评估单个样本预测的准确性。

---

### 公式 (3.23)：凸函数性质 / Convexity Property

$$
\nabla^2 \mathcal{R}(\boldsymbol{w}) = \frac{1}{N}\sum_{n=1}^{N} \sigma(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)(1-\sigma(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b))\boldsymbol{x}^{(n)}(\boldsymbol{x}^{(n)})^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla^2$ | hēi sī yàn jī | Hessian矩阵（二阶导数） |
| $\sigma(\cdot)$ | xī gé mǎ | Sigmoid激活函数 |
| $\boldsymbol{x}^{(n)}$ | shū rù tè zhēng | 第n个样本的输入特征 |

**一句话解释：**
风险函数的Hessian矩阵为正半定，说明目标函数是凸函数，有唯一全局最优解。

**简单应用场景：**
理论分析，保证梯度下降能找到全局最优的权重参数。

---

### 公式 (3.24)：梯度计算 / Gradient Computation

$$
\frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w}} = \frac{1}{N}\sum_{n=1}^{N} (\sigma(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) - y^{(n)})\boldsymbol{x}^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \boldsymbol{w}}$ | piàn dǎo shù | 偏导数 |
| $\sigma(\cdot)$ | xī gé mǎ | Sigmoid激活函数 |
| $y^{(n)}$ | zhēn shí biāo qiān | 第n个样本的真实标签 |

**一句话解释：**
目标函数对权重的偏导数，反映了预测值与真实标签的误差加权特征。

**简单应用场景：**
梯度下降法更新权重参数时使用。

---

### 公式 (3.25)：权重更新 / Weight Update

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āl fǎ | 学习率 |
| $\frac{\partial \mathcal{R}}{\partial \boldsymbol{w}}$ | piàn dǎo shù | 梯度 |
| $\leftarrow$ | fù zhí | 赋值操作 |

**一句话解释：**
按照负梯度方向更新权重，步长由学习率控制。

**简单应用场景：**
梯度下降法的迭代更新规则。

---

### 公式 (3.26)：多分类Softmax回归 / Softmax Regression

$$
p(y=c|\boldsymbol{x}) = \frac{\exp(\boldsymbol{w}_c^T\boldsymbol{x} + b_c)}{\sum_{k=1}^{K} \exp(\boldsymbol{w}_k^T\boldsymbol{x} + b_k)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y=c\|\boldsymbol{x})$ | gài lǜ | 样本属于第c类的概率 |
| $\boldsymbol{w}_c$ | quán zhòng | 第c类的权重向量 |
| $b_c$ | piān zhì | 第c类的偏置项 |
| $K$ | lèi bié shù | 类别总数 |

**一句话解释：**
将多个线性组合通过softmax函数归一化，得到K个类别的概率分布。

**简单应用场景：**
多分类问题中的概率预测。

---

### 公式 (3.27)：多分类风险函数 / Softmax Risk Function

$$
\mathcal{R}(\boldsymbol{W}) = -\sum_{n=1}^{N} \sum_{k=1}^{K} y_k^{(n)} \log p(y=k|\boldsymbol{x}^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(\boldsymbol{W})$ | fēng xiǎn hánshù | 多分类风险函数 |
| $y_k^{(n)}$ | biāo qiān | 第n个样本第k类的真实标签（one-hot编码） |
| $K$ | lèi bié shù | 类别总数 |
| $N$ | yàng běn shù | 样本总数 |

**一句话解释：**
对所有样本的交叉熵损失求和，衡量多分类模型的预测性能。

**简单应用场景：**
Softmax回归的目标函数，需要最小化。

---

### 公式 (3.28)：随机梯度下降 / Stochastic Gradient Descent

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \frac{\partial \mathcal{L}^{(n)}(\boldsymbol{w})}{\partial \boldsymbol{w}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āl fǎ | 学习率 |
| $\mathcal{L}^{(n)}(\boldsymbol{w})$ | dān gè yàng běn sǔn shī | 单个样本的损失函数 |
| $\frac{\partial}{\partial \boldsymbol{w}}$ | piàn dǎo shù | 偏导数 |

**一句话解释：**
基于单个样本的损失函数梯度进行权重更新，使用频率高但梯度方差大。

**简单应用场景：**
大规模数据集的在线学习和实时更新。

---

### 公式 (3.29)：二分类决策函数（向量形式）/ Binary Classification Decision Function

$$
\hat{y}^{(n)} = \text{sign}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}^{(n)}$ | yù cè | 第n个样本的预测标签 |
| $\text{sign}(\cdot)$ | fú hào hánshù | 符号函数 |
| $\boldsymbol{w}$ | quán zhòng | 权重向量 |
| $b$ | piān zhì | 偏置项 |

**一句话解释：**
根据线性组合的符号决定预测类别，正为第一类，负为第二类。

**简单应用场景：**
二分类问题的最终预测。

---

### 公式 (3.30)：支持向量机目标函数 / Support Vector Machine Objective

$$
\min_{\boldsymbol{w},b} \frac{1}{2}\|\boldsymbol{w}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{1}{2}\|\boldsymbol{w}\|^2$ | quán zhòng de píngfāng mó | 权重平方模 |
| $\min$ | zuì xiǎo huà | 最小化 |

**一句话解释：**
最小化权重的模长平方，以最大化分类间隔（margin）。

**简单应用场景：**
支持向量机的基本目标，在约束条件下求解。

---

### 公式 (3.31)：支持向量机约束条件 / SVM Constraints

$$
y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) \geq 1, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | zhēn shí biāo qiān | 第n个样本的真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b$ | xiàn xìng zǔ hé | 线性组合 |
| $\geq 1$ | dà yú děng yú | 大于等于 |

**一句话解释：**
所有样本到决策超平面的距离至少为1，保证分类边界的安全间隔。

**简单应用场景：**
硬间隔支持向量机的线性可分约束。

---

### 公式 (3.32)：支持向量机对偶问题 / SVM Dual Problem

$$
y = \min_{n} y^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | biāo qiān | 第n个样本的标签值 |
| $\min_n$ | zuì xiǎo zhí | 最小标签值 |

**一句话解释：**
取所有样本标签的最小值，用于参数化问题。

**简单应用场景：**
支持向量机的参数化过程。

---

### 公式 (3.33)：间隔最大化问题 / Margin Maximization

$$
\max_{\boldsymbol{w},b} \gamma
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gān mǎ | 间隔宽度 |
| $\max$ | zuì dà huà | 最大化 |

**一句话解释：**
最大化正负样本之间的决策边界间隔，提高模型的泛化能力。

**简单应用场景：**
支持向量机的核心目标。

---

### 公式 (3.34)：间隔约束 / Margin Constraints

$$
\frac{y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)}{\|\boldsymbol{w}\|} \geq \gamma, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gān mǎ | 间隔宽度 |
| $\|\boldsymbol{w}\|$ | quán zhòng de mó | 权重向量的模 |
| $y^{(n)}$ | zhēn shí biāo qiān | 第n个样本的真实标签 |

**一句话解释：**
每个样本到决策超平面的规范化距离至少为γ，保证最小分类间隔。

**简单应用场景：**
支持向量机的间隔约束条件。

---

### 公式 (3.35)：标准化间隔问题 / Normalized Margin Problem

$$
\max_{\boldsymbol{w},b} \gamma
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gān mǎ | 间隔宽度 |

**一句话解释：**
在规范化约束下最大化间隔，等价于最小化权重的模长。

**简单应用场景：**
支持向量机问题的等价表述。

---

### 公式 (3.36)：标准化形式 / Canonical Form

$$
\min_{\boldsymbol{w},b} \frac{1}{\|\boldsymbol{w}\|^2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{1}{\|\boldsymbol{w}\|^2}$ | quán zhòng mó de dào shù | 权重模的倒数 |
| $\min$ | zuì xiǎo huà | 最小化 |

**一句话解释：**
通过最小化权重模的倒数来最大化间隔，等价于最小化权重的平方模。

**简单应用场景：**
支持向量机的等价优化目标。

---

### 公式 (3.37)：支持向量机标准形式 / Standard SVM Formulation

$$
\min_{\boldsymbol{w},b} \frac{1}{2}\|\boldsymbol{w}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{1}{2}$ | èr fēn zhī yī | 常数因子 |
| $\|\boldsymbol{w}\|^2$ | quán zhòng píngfāng mó | 权重平方模 |

**一句话解释：**
支持向量机的标准目标函数形式，乘以1/2便于求导。

**简单应用场景：**
支持向量机的优化问题。

---

### 公式 (3.38)：支持向量机约束形式 / SVM with Constraints

$$
\text{s.t.} \quad y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) \geq 1, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| s.t. | suǒ yǐ | 约束条件 |
| $y^{(n)}$ | zhēn shí biāo qiān | 第n个样本的真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
支持向量机的等式约束，要求所有样本都在正确一侧且间隔至少为1。

**简单应用场景：**
硬间隔支持向量机的完整问题陈述。

---

### 公式 (3.39)：Softmax回归风险函数 / Softmax Risk Function

$$
\mathcal{R}(\boldsymbol{W}) = -\frac{1}{N}\sum_{n=1}^{N} \sum_{k=1}^{K} y_k^{(n)} \log \hat{y}_k^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(\boldsymbol{W})$ | fēng xiǎn hánshù | 风险函数 |
| $y_k^{(n)}$ | zhēn shí biāo qiān | 第n个样本第k类的真实标签 |
| $\hat{y}_k^{(n)}$ | yù cè gài lǜ | 第n个样本第k类的预测概率 |
| $K$ | lèi bié shù | 类别总数 |

**一句话解释：**
多分类的交叉熵损失的平均值，衡量预测概率与真实标签的差异。

**简单应用场景：**
Softmax回归的目标函数，通过最小化来训练分类模型。

---

### 公式 (3.40)：梯度计算 / Gradient Calculation

$$
\frac{\partial \mathcal{L}^{(n)}(\boldsymbol{W})}{\partial \boldsymbol{W}} = -(\boldsymbol{y}^{(n)} - \hat{\boldsymbol{y}}^{(n)})(\boldsymbol{x}^{(n)})^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}^{(n)}$ | sǔn shī | 第n个样本的损失函数 |
| $\boldsymbol{y}^{(n)}$ | zhēn shí | 第n个样本的真实标签向量 |
| $\hat{\boldsymbol{y}}^{(n)}$ | yù cè | 第n个样本的预测概率向量 |

**一句话解释：**
损失函数对权重矩阵的偏导数，由预测误差与输入特征的外积组成。

**简单应用场景：**
梯度下降法更新权重矩阵时使用。

---

### 公式 (3.41)：权重更新规则 / Weight Update Rule

$$
\boldsymbol{W} \leftarrow \boldsymbol{W} - \alpha \frac{\partial \mathcal{L}^{(n)}(\boldsymbol{W})}{\partial \boldsymbol{W}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | āl fǎ | 学习率 |
| $\mathcal{L}^{(n)}$ | sǔn shī | 第n个样本的损失函数 |
| $\leftarrow$ | fù zhí | 赋值操作 |

**一句话解释：**
按照负梯度方向更新权重矩阵，步长由学习率决定。

**简单应用场景：**
Softmax回归的参数更新。

---

### 公式 (3.42)：Softmax函数 / Softmax Activation

$$
\text{softmax}(\boldsymbol{z})_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{softmax}(\boldsymbol{z})_k$ | ruǎn zuì dà zhí hánshù | Softmax函数第k个输出 |
| $z_k$ | xiàn xìng zǔ hé | 第k个类别的线性组合 |
| $K$ | lèi bié shù | 类别总数 |

**一句话解释：**
将K个线性组合值通过指数和归一化，转化为概率分布。

**简单应用场景：**
多分类神经网络的输出层激活函数。

---

### 公式 (3.43)：Softmax梯度 / Softmax Gradient

$$
\frac{\partial \text{softmax}(\boldsymbol{z})_k}{\partial z_j} = \begin{cases}
\text{softmax}(\boldsymbol{z})_k(1 - \text{softmax}(\boldsymbol{z})_k) & \text{if } k=j \\
-\text{softmax}(\boldsymbol{z})_k \text{softmax}(\boldsymbol{z})_j & \text{if } k \neq j
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial z_j}$ | piàn dǎo shù | 对第j个输入的偏导数 |
| $\text{softmax}(\boldsymbol{z})_k$ | ruǎn zuì dà | 第k个输出 |

**一句话解释：**
Softmax函数的偏导数，展现了自身项和交叉项的不同性质。

**简单应用场景：**
反向传播算法中计算梯度时使用。

---

### 公式 (3.44)：交叉熵损失 / Cross Entropy Loss

$$
\mathcal{L}^{(n)} = -\sum_{k=1}^{K} y_k^{(n)} \log \hat{y}_k^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}^{(n)}$ | sǔn shī | 第n个样本的损失 |
| $y_k^{(n)}$ | zhēn shí biāo qiān | 真实标签one-hot编码的第k维 |
| $\hat{y}_k^{(n)}$ | yù cè gài lǜ | 预测的第k类概率 |
| $K$ | lèi bié shù | 类别总数 |

**一句话解释：**
衡量预测概率分布与真实标签分布的差异，是多分类的标准损失函数。

**简单应用场景：**
Softmax回归模型的损失函数。

---

### 公式 (3.45)：交叉熵梯度 / Cross Entropy Gradient

$$
\frac{\partial \mathcal{L}^{(n)}}{\partial z_k} = \hat{y}_k^{(n)} - y_k^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z_k$ | xiàn xìng zǔ hé | 第k个类别的线性组合 |
| $\hat{y}_k^{(n)}$ | yù cè gài lǜ | 预测概率 |
| $y_k^{(n)}$ | zhēn shí biāo qiān | 真实标签 |

**一句话解释：**
交叉熵对线性组合的偏导数等于预测与真实标签的差，形式优雅。

**简单应用场景：**
反向传播中直接计算输出层的梯度。

---

### 公式 (3.46)：最小间隔 / Minimum Margin

$$
y = \min_n y^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |
| $\min_n$ | zuì xiǎo | 最小值 |

**一句话解释：**
取所有样本预测值的最小值，代表模型的最小间隔。

**简单应用场景：**
支持向量机的参数化过程中定义最小间隔。

---

### 公式 (3.47)：感知机约束 / Perceptron Constraint

$$
\max_{\boldsymbol{w},b} \gamma
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gān mǎ | 间隔宽度 |

**一句话解释：**
最大化分类间隔以提高模型的鲁棒性和泛化能力。

**简单应用场景：**
支持向量机的优化目标。

---

### 公式 (3.48)：间隔约束（标准化） / Normalized Margin Constraints

$$
\frac{y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b)}{\|\boldsymbol{w}\|} \geq \gamma
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma$ | gān mǎ | 间隔宽度 |
| $\|\boldsymbol{w}\|$ | quán zhòng mó | 权重向量的范数 |
| $y^{(n)}$ | zhēn shí biāo qiān | 真实标签 |

**一句话解释：**
每个样本到决策超平面的规范化距离至少为γ。

**简单应用场景：**
支持向量机的间隔定义。

---

### 公式 (3.49)：标准化问题转化 / Problem Reformulation

$$
\|\boldsymbol{w}\| = \frac{1}{\gamma}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{w}\|$ | quán zhòng mó | 权重向量的范数 |
| $\gamma$ | gān mǎ | 间隔宽度 |

**一句话解释：**
在标准化约束下，权重的模与间隔成反比例关系。

**简单应用场景：**
支持向量机的约束转化。

---

### 公式 (3.50)：对偶形式表述 / Dual Form Expression

$$
\min_{\boldsymbol{w},b} \frac{1}{2}\|\boldsymbol{w}\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{1}{2}\|\boldsymbol{w}\|^2$ | quán zhòng píngfāng mó | 权重平方模的一半 |

**一句话解释：**
支持向量机的标准形式目标函数，系数1/2用于简化求导。

**简单应用场景：**
支持向量机的优化问题。

---

### 公式 (3.51)：约束条件 / Constraints

$$
\text{s.t.} \quad y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) \geq 1, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| s.t. | suǒ yǐ | 约束条件 |
| $y^{(n)}$ | zhēn shí biāo qiān | 第n个样本的真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b$ | yù cè zhí | 第n个样本的预测值 |

**一句话解释：**
所有样本的预测值与标签的乘积不小于1，确保正确分类且保有足够间隔。

**简单应用场景：**
硬间隔支持向量机的约束条件。

---

### 公式 (3.52)：拉格朗日函数 / Lagrangian Function

$$
\mathcal{L}(\boldsymbol{w}, b, \boldsymbol{\lambda}) = \frac{1}{2}\|\boldsymbol{w}\|^2 - \sum_{n=1}^{N} \lambda_n(y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) - 1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | lā gé lǎng rì hánshù | 拉格朗日函数 |
| $\boldsymbol{\lambda}$ | lán mǔ dá | 拉格朗日乘数向量 |
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |

**一句话解释：**
将约束条件并入目标函数，形成拉格朗日对偶问题的基础。

**简单应用场景：**
支持向量机的对偶问题推导。

---

### 公式 (3.53)：KKT条件 / KKT Conditions

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{w}} = \boldsymbol{w} - \sum_{n=1}^{N} \lambda_n y^{(n)}\boldsymbol{x}^{(n)} = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial \boldsymbol{w}}$ | piàn dǎo shù | 关于权重的偏导数 |
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |

**一句话解释：**
在最优点处，拉格朗日函数对权重的偏导数为零。

**简单应用场景：**
支持向量机的最优性条件。

---

### 公式 (3.54)：偏置求解条件 / Bias Solution

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{n=1}^{N} \lambda_n y^{(n)} = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial b}$ | piàn dǎo shù | 关于偏置的偏导数 |
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |

**一句话解释：**
在最优点处，所有乘数与标签乘积的和为零。

**简单应用场景：**
支持向量机中确定偏置项的条件。

---

### 公式 (3.55)：权重表示 / Weight Representation

$$
\boldsymbol{w} = \sum_{n=1}^{N} \lambda_n y^{(n)}\boldsymbol{x}^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}$ | quán zhòng | 权重向量 |
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |

**一句话解释：**
最优权重可以表示为所有训练样本的加权线性组合，权重为λn。

**简单应用场景：**
支持向量机的对偶形式推导，只有支持向量对应的λn非零。

---

### 公式 (3.56)：对偶问题 / Dual Problem

$$
\max_{\boldsymbol{\lambda}} \sum_{n=1}^{N} \lambda_n - \frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N} \lambda_n \lambda_m y^{(n)} y^{(m)} \boldsymbol{x}^{(n)T} \boldsymbol{x}^{(m)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\lambda}$ | lán mǔ dá | 拉格朗日乘数向量 |
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |

**一句话解释：**
通过拉格朗日对偶变换得到的问题，只依赖于样本的内积。

**简单应用场景：**
支持向量机的对偶形式，便于使用核技巧。

---

### 公式 (3.57)：对偶约束 / Dual Constraints

$$
\lambda_n \geq 0, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $\geq 0$ | dà yú děng yú líng | 大于等于零 |

**一句话解释：**
所有拉格朗日乘数必须非负，这是支持向量机对偶问题的基本约束。

**简单应用场景：**
支持向量机对偶形式的可行域约束。

---

### 公式 (3.58)：核函数定义 / Kernel Function Definition

$$
k(\boldsymbol{x}, \boldsymbol{z}) = \phi(\boldsymbol{x})^T \phi(\boldsymbol{z})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $k(\boldsymbol{x}, \boldsymbol{z})$ | hé hánshù | 核函数 |
| $\phi(\boldsymbol{x})$ | tè zhēng yìng shè | 特征映射函数 |
| $\phi(\boldsymbol{z})$ | tè zhēng yìng shè | 特征映射函数 |

**一句话解释：**
核函数计算高维特征空间中两个样本的内积，无需显式计算特征映射。

**简单应用场景：**
支持向量机中处理非线性问题的关键技巧。

---

### 公式 (3.59)：多项式核 / Polynomial Kernel

$$
k(\boldsymbol{x}, \boldsymbol{z}) = (1 + \boldsymbol{x}^T\boldsymbol{z})^p
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $k(\boldsymbol{x}, \boldsymbol{z})$ | hé hánshù | 多项式核函数 |
| $p$ | pì | 多项式的次数 |
| $\boldsymbol{x}^T\boldsymbol{z}$ | nèi jī | 向量内积 |

**一句话解释：**
多项式核将原始特征空间映射到p阶多项式特征空间，实现非线性分类。

**简单应用场景：**
支持向量机处理多项式特征的核函数。

---

### 公式 (3.60)：高斯核（RBF核）/ Gaussian Kernel (RBF Kernel)

$$
k(\boldsymbol{x}, \boldsymbol{z}) = \exp\left(-\frac{\|\boldsymbol{x} - \boldsymbol{z}\|^2}{2\sigma^2}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $k(\boldsymbol{x}, \boldsymbol{z})$ | hé hánshù | 高斯核函数 |
| $\|\boldsymbol{x} - \boldsymbol{z}\|^2$ | ōu jī lǐ jù lí píngfāng | 欧几里得距离平方 |
| $\sigma$ | xī gé mǎ | 高斯核的宽度参数 |

**一句话解释：**
高斯核基于样本之间的欧氏距离，将原始空间映射到无穷维特征空间。

**简单应用场景：**
支持向量机处理非线性问题的最常用核函数。

---

### 公式 (3.61)：支持向量机预测 / SVM Prediction

$$
\hat{y} = \text{sign}\left(\sum_{n \in S} \lambda_n y^{(n)} k(\boldsymbol{x}^{(n)}, \boldsymbol{x}) + b\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{y}$ | yù cè | 预测标签 |
| $S$ | zhī chí xiàng liàng jí | 支持向量集合 |
| $\lambda_n$ | lán mǔ dá | 第n个支持向量的系数 |
| $k(\cdot, \cdot)$ | hé hánshù | 核函数 |

**一句话解释：**
预测基于支持向量与新样本的核函数相似度的加权和，无需计算权重向量。

**简单应用场景：**
使用核技巧的支持向量机的预测函数。

---

### 公式 (3.62)：软间隔问题 / Soft Margin Problem

$$
\min_{\boldsymbol{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_{n=1}^{N} \xi_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\xi}$ | xī | 松弛变量向量 |
| $\xi_n$ | xī | 第n个样本的松弛变量 |
| $C$ | cháng shù | 正则化参数（惩罚因子） |

**一句话解释：**
允许部分样本违反间隔约束，通过松弛变量和惩罚参数C平衡鲁棒性和准确性。

**简单应用场景：**
处理线性不可分数据的支持向量机。

---

### 公式 (3.63)：软间隔约束 / Soft Margin Constraints

$$
y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) \geq 1 - \xi_n, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\xi_n$ | xī | 第n个样本的松弛变量 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |
| $\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
允许样本到决策超平面的距离小于1，但使用松弛变量控制违反程度。

**简单应用场景：**
软间隔支持向量机的约束条件。

---

### 公式 (3.64)：松弛变量约束 / Slack Variable Constraints

$$
\xi_n \geq 0, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\xi_n$ | xī | 第n个样本的松弛变量 |
| $\geq 0$ | dà yú děng yú líng | 大于等于零 |

**一句话解释：**
松弛变量必须非负，衡量样本对约束的违反程度。

**简单应用场景：**
软间隔支持向量机的可行域约束。

---

### 公式 (3.65)：Hinge损失函数 / Hinge Loss Function

$$
\mathcal{L}_{\text{hinge}} = \max(0, 1 - y(\boldsymbol{w}^T\boldsymbol{x} + b))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{hinge}}$ | hinge sǔn shī | Hinge损失函数 |
| $y$ | biāo qiān | 真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
当预测值与标签乘积不足1时产生损失，否则损失为0，形成支持向量机的自然损失函数。

**简单应用场景：**
支持向量机的损失函数，与松弛变量等价。

---

### 公式 (3.66)：感知机（Perceptron）/ Perceptron

$$
\mathcal{L}_{\text{perceptron}} = \max(0, -y(\boldsymbol{w}^T\boldsymbol{x} + b))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{perceptron}}$ | gǎn zhī jī sǔn shī | 感知机损失函数 |
| $y$ | biāo qiān | 真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
当预测值与标签同号但未能充分分离时产生损失，比Hinge损失更宽松。

**简单应用场景：**
感知机算法的损失函数。

---

### 公式 (3.67)：指数损失（ExponentialLoss）/ Exponential Loss

$$
\mathcal{L}_{\text{exp}} = \exp(-y(\boldsymbol{w}^T\boldsymbol{x} + b))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{exp}}$ | zhǐ shù sǔn shī | 指数损失函数 |
| $y$ | biāo qiān | 真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
对错误分类给予指数级的惩罚，错误越大惩罚越大，用于Adaboost等集成学习。

**简单应用场景：**
集成学习中的损失函数。

---

### 公式 (3.68)：平方损失 / Squared Loss

$$
\mathcal{L}_{\text{squared}} = (1 - y(\boldsymbol{w}^T\boldsymbol{x} + b))^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{squared}}$ | píngfāng sǔn shī | 平方损失函数 |
| $y$ | biāo qiān | 真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
约束违反的平方作为损失，对大的违反给予更大的惩罚。

**简单应用场景：**
支持向量回归和某些分类任务。

---

### 公式 (3.69)：逻辑回归损失函数 / Logistic Regression Loss

$$
\mathcal{L}_{\text{logistic}} = \log(1 + \exp(-y(\boldsymbol{w}^T\boldsymbol{x} + b)))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{\text{logistic}}$ | luó jī sì huí guī sǔn shī | 逻辑回归损失函数 |
| $y$ | biāo qiān | 真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
使用对数几率作为损失函数，是逻辑回归的标准损失。

**简单应用场景：**
逻辑回归的目标函数。

---

### 公式 (3.70)：0-1损失 / 0-1 Loss

$$
\mathcal{L}_{0-1} = \begin{cases}
0 & \text{if } y(\boldsymbol{w}^T\boldsymbol{x} + b) > 0 \\
1 & \text{otherwise}
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}_{0-1}$ | líng yī sǔn shī | 0-1损失函数 |
| $y$ | biāo qiān | 真实标签 |
| $\boldsymbol{w}^T\boldsymbol{x} + b$ | yù cè zhí | 预测值 |

**一句话解释：**
只要分类正确就没有损失，错误就是最大损失，直接衡量分类错误率。

**简单应用场景：**
分类问题的最终评估指标。

---

### 公式 (3.71)：拉格朗日函数（软间隔）/ Lagrangian for Soft Margin

$$
\mathcal{L}(\boldsymbol{w}, b, \boldsymbol{\xi}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = \frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_{n=1}^{N} \xi_n - \sum_{n=1}^{N} \lambda_n(y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) - 1 + \xi_n) - \sum_{n=1}^{N} \mu_n \xi_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | lā gé lǎng rì hánshù | 拉格朗日函数 |
| $\boldsymbol{\lambda}$ | lán mǔ dá | 间隔约束的乘数 |
| $\boldsymbol{\mu}$ | mǔ | 非负约束的乘数 |
| $\xi_n$ | xī | 松弛变量 |

**一句话解释：**
软间隔支持向量机的拉格朗日函数，包含间隔约束和非负约束的乘数。

**简单应用场景：**
软间隔支持向量机的对偶问题推导。

---

### 公式 (3.72)：KKT条件（权重）/ KKT for Weight

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{w}} = \boldsymbol{w} - \sum_{n=1}^{N} \lambda_n y^{(n)}\boldsymbol{x}^{(n)} = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial \boldsymbol{w}}$ | piàn dǎo shù | 关于权重的偏导数 |
| $\boldsymbol{\lambda}$ | lán mǔ dá | 乘数向量 |

**一句话解释：**
在最优点处，权重等于所有样本的加权和（权重为λn乘以标签乘以特征）。

**简单应用场景：**
软间隔支持向量机的最优性条件。

---

### 公式 (3.73)：KKT条件（偏置）/ KKT for Bias

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{n=1}^{N} \lambda_n y^{(n)} = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial b}$ | piàn dǎo shù | 关于偏置的偏导数 |
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |

**一句话解释：**
在最优点处，所有乘数与标签乘积的和为零。

**简单应用场景：**
确定偏置项的条件。

---

### 公式 (3.74)：KKT条件（松弛变量）/ KKT for Slack Variable

$$
\frac{\partial \mathcal{L}}{\partial \xi_n} = C - \lambda_n - \mu_n = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial \xi_n}$ | piàn dǎo shù | 关于松弛变量的偏导数 |
| $C$ | zhèng zé huà | 正则化参数 |
| $\lambda_n$ | lán mǔ dá | 第n个间隔约束的乘数 |
| $\mu_n$ | mǔ | 第n个非负约束的乘数 |

**一句话解释：**
在最优点处，正则化参数等于两个乘数之和。

**简单应用场景：**
软间隔支持向量机的互补松弛条件。

---

### 公式 (3.75)：互补松弛条件1 / Complementary Slackness 1

$$
\lambda_n (y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) - 1 + \xi_n) = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |
| $\xi_n$ | xī | 第n个样本的松弛变量 |

**一句话解释：**
只有当约束不活跃（边界）或乘数非零，两者不能同时成立。

**简单应用场景：**
支持向量机中识别支持向量。

---

### 公式 (3.76)：互补松弛条件2 / Complementary Slackness 2

$$
\mu_n \xi_n = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mu_n$ | mǔ | 第n个非负约束的乘数 |
| $\xi_n$ | xī | 第n个样本的松弛变量 |

**一句话解释：**
松弛变量非零时乘数为零，反之亦然。

**简单应用场景：**
支持向量机的互补松弛条件。

---

### 公式 (3.77)：互补松弛条件3 / Complementary Slackness 3

$$
\lambda_n \geq 0, \quad \mu_n \geq 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 第n个间隔约束的乘数 |
| $\mu_n$ | mǔ | 第n个非负约束的乘数 |

**一句话解释：**
所有拉格朗日乘数必须非负。

**简单应用场景：**
支持向量机对偶问题的可行域约束。

---

### 公式 (3.78)：对偶问题（软间隔）/ Dual Problem (Soft Margin)

$$
\max_{\boldsymbol{\lambda}} \sum_{n=1}^{N} \lambda_n - \frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{N} \lambda_n \lambda_m y^{(n)} y^{(m)} \boldsymbol{x}^{(n)T}\boldsymbol{x}^{(m)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\lambda}$ | lán mǔ dá | 乘数向量 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |
| $\boldsymbol{x}^{(n)T}\boldsymbol{x}^{(m)}$ | nèi jī | 样本内积 |

**一句话解释：**
软间隔支持向量机的对偶形式，目标函数与硬间隔相同。

**简单应用场景：**
支持向量机的对偶问题求解。

---

### 公式 (3.79)：对偶约束（软间隔）/ Dual Constraints (Soft Margin)

$$
0 \leq \lambda_n \leq C, \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $C$ | zhèng zé huà | 正则化参数（上界） |

**一句话解释：**
软间隔的对偶约束是乘数有上界C，比硬间隔的非负约束更严格。

**简单应用场景：**
软间隔支持向量机的可行域。

---

### 公式 (3.80)：支持向量分类 / Support Vector Classification

$$
\sum_{n=1}^{N} \lambda_n y^{(n)} = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $y^{(n)}$ | biāo qiān | 第n个样本的标签 |

**一句话解释：**
所有支持向量的乘数与标签乘积的加权和为零。

**简单应用场景：**
支持向量机的基本约束。

---

### 公式 (3.81)：支持向量鉴别 / Support Vector Identification

$$
0 < \lambda_n < C \Rightarrow \xi_n = 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $\xi_n$ | xī | 第n个样本的松弛变量 |
| $C$ | zhèng zé huà | 正则化参数 |

**一句话解释：**
当乘数严格介于0和C之间时，松弛变量为0，样本恰好在间隔边界上。

**简单应用场景：**
识别边界支持向量。

---

### 公式 (3.82)：支持向量鉴别（继续）/ Support Vector Identification (Cont.)

$$
\lambda_n = C \Rightarrow y^{(n)}(\boldsymbol{w}^T\boldsymbol{x}^{(n)} + b) = 1 - \xi_n \leq 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda_n$ | lán mǔ dá | 第n个样本的乘数 |
| $\xi_n$ | xī | 第n个样本的松弛变量 |
| $C$ | zhèng zé huà | 正则化参数 |

**一句话解释：**
当乘数等于C时，样本可能在间隔内或被错误分类。

**简单应用场景：**
识别间隔内或边界外的支持向量。

---

### 公式 (3.83)：最小间隔定义 / Minimum Margin Definition

$$
y = \min_n y^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y^{(n)}$ | biāo qiān | 第n个样本的预测值 |
| $\min_n$ | zuì xiǎo | 最小值 |

**一句话解释：**
取所有样本预测值的最小值作为整体的最小间隔。

**简单应用场景：**
支持向量机的间隔定义。

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

### 公式 (3.85)：硬间隔目标函数 / Hard Margin Objective

$$
\max_{\boldsymbol{w},b} \gamma
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


### 公式 3.87：支持向量机 - 标准形式 / Support Vector Machine - Standard Form

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

### 公式 3.88：支持向量机 - 拉格朗日对偶形式 / Support Vector Machine - Lagrangian Dual Form

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

### 公式 3.89：支持向量机 - 权重向量优化条件 / Support Vector Machine - Weight Optimization

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

### 公式 3.90：支持向量机 - 偏置项优化条件 / Support Vector Machine - Bias Optimization

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

### 公式 3.91：支持向量机 - 对偶问题目标函数 / Support Vector Machine - Dual Problem Objective

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

### 公式 3.92：支持向量机 - 偏置项估计 / Support Vector Machine - Bias Estimation

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

### 公式 3.93：支持向量机 - 决策函数 / Support Vector Machine - Decision Function

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

### 公式 3.94：支持向量机 - 对偶形式决策函数 / Support Vector Machine - Decision Function in Dual Form

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

### 公式 3.95：核函数支持向量机 - 标准形式 / Kernel SVM - Standard Form

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

### 公式 3.96：核函数支持向量机 - 对偶形式 / Kernel SVM - Dual Form

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

### 公式 3.97：多项式核函数 / Polynomial Kernel Function

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

### 公式 3.98：多项式核函数 - 显式映射形式 / Polynomial Kernel - Explicit Feature Mapping

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

### 公式 3.99：软间隔支持向量机 - 优化问题 / Soft-Margin SVM - Optimization Problem

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

### 公式 3.100：软间隔支持向量机 - Hinge损失形式 / Soft-Margin SVM - Hinge Loss Form

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

### 公式 3.101：Logistic回归 - 交叉熵损失函数 / Logistic Regression - Cross-Entropy Loss

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

### 公式 3.102：Logistic回归 - 交叉熵损失函数（等价形式） / Logistic Regression - Cross-Entropy Loss (Equivalent Form)

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

### 公式 3.103：Logistic回归 - 简化损失函数 / Logistic Regression - Simplified Loss

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

### 公式 3.104：Logistic回归 - 对数形式 / Logistic Regression - Logarithmic Form

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

### 公式 3.105：感知损失 / Perceptron Loss

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

### 公式 3.106：软间隔支持向量机 - Hinge损失 / Soft-Margin SVM - Hinge Loss

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

### 公式 3.107：平方损失函数 / Squared Loss Function

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

### 公式 3.108：平方损失函数 - 展开形式1 / Squared Loss - Expanded Form 1

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

### 公式 3.109：平方损失函数 - 展开形式2 / Squared Loss - Expanded Form 2

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

### 公式 (4.2)：向量化神经元计算 / Vectorized Neuron Calculation

$$
z = \boldsymbol{\omega}^T \boldsymbol{x} + b
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \boldsymbol{\omega} | ōu mǐ gā | 权重向量 |
| \boldsymbol{x} | x | 输入向量 |
| z | zī | 神经元净输入 |
| b | bī | 偏置项 |

**一句话解释：**
使用向量表示法，将权重向量与输入向量的点积加上偏置项来计算神经元的输入。

**简单应用场景：**
在编程和数值计算中，使用向量化形式可以利用BLAS等优化库提升计算效率。

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

### 公式 (4.4)：Logistic函数 / Logistic Activation Function

$$
\sigma(x) = \frac{1}{1 + \exp(-x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \sigma | xī gé mǎ | Logistic激活函数 |
| x | x | 输入 |
| \exp | ē | 指数函数 |

**一句话解释：**
Logistic函数将任意实数映射到(0,1)区间，常用于二分类问题的输出层。

**简单应用场景：**
在逻辑回归和神经网络的二分类输出层中，用于将预测值映射为概率。

---

### 公式 (4.5)：Tanh函数 / Hyperbolic Tangent Function

$$
\tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \tanh | tán huo | 双曲正切函数 |
| x | x | 输入 |
| \exp | ē | 指数函数 |

**一句话解释：**
Tanh函数将任意实数映射到(-1,1)区间，是Logistic函数的平移和缩放版本。

**简单应用场景：**
在隐层中常用的激活函数，其输出以0为中心，有助于加快收敛。

---

### 公式 (4.6)：Tanh与Logistic的关系 / Relationship between Tanh and Logistic

$$
\tanh(x) = 2\sigma(2x) - 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \tanh | tán huo | 双曲正切函数 |
| \sigma | xī gé mǎ | Logistic函数 |
| x | x | 输入 |

**一句话解释：**
Tanh函数可以通过Logistic函数进行线性变换得到，揭示了两个函数间的联系。

**简单应用场景：**
理论分析中用于证明Tanh和Logistic函数的等价性和相互转化。

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
| \sigma | xī gé mǎ | Logistic激活函数 |
| \sigma' | xī gé mǎ dǎo shù | Logistic函数的导数 |

**一句话解释：**
在x=0处对Logistic函数进行一阶泰勒展开，用于构造Hard-Logistic函数的近似。

**简单应用场景：**
为了获得计算效率更高的激活函数，可以用分段线性函数近似Logistic函数。

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
| \text{hard-logistic} | hèn luó ji | Hard-Logistic激活函数 |
| x | x | 输入 |
| g_l | g xiā biāo l | Logistic函数的一阶近似 |

**一句话解释：**
通过截断Logistic函数的线性近似，使其输出在(-∞,0)和(1,+∞)处达到饱和。

**简单应用场景：**
在需要快速计算且对精度要求不极高的场景中使用。

---

### 公式 (4.10)：Hard-Logistic的等价形式1 / Hard-Logistic Equivalent Form 1

$$
= \max\left(\min(g_l(x), 1), 0\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \max | zuì dà | 最大值函数 |
| \min | zuì xiǎo | 最小值函数 |
| g_l | g xiā biāo l | Logistic函数的一阶近似 |

**一句话解释：**
Hard-Logistic函数也可以表示为先取下界后取上界的形式。

**简单应用场景：**
编程实现时可以直接使用min和max函数。

---

### 公式 (4.11)：Hard-Logistic的等价形式2 / Hard-Logistic Explicit Form

$$
= \max\left(\min(0.25x + 0.5, 1), 0\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \max | zuì dà | 最大值函数 |
| \min | zuì xiǎo | 最小值函数 |
| x | x | 输入 |

**一句话解释：**
将Hard-Logistic函数完全展开为显式的分段线性形式。

**简单应用场景：**
直接计算Hard-Logistic函数值。

---

### 公式 (4.12)：Tanh函数的一阶泰勒展开 / First-Order Taylor Expansion of Tanh

$$
g_t(x) = \tanh(0) + x \cdot \tanh'(0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| g_t | g xiā biāo t | Tanh函数的近似 |
| x | x | 输入 |
| \tanh | tán huo | 双曲正切函数 |
| \tanh' | tán huo dǎo shù | Tanh函数的导数 |

**一句话解释：**
在x=0处对Tanh函数进行一阶泰勒展开，用于构造Hard-Tanh函数。

**简单应用场景：**
为Tanh函数创建更高效的近似版本。

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

### 公式 (4.14)：Hard-Tanh分段函数 / Hard-Tanh Function

$$
\text{hard-tanh}(x) = \max\left(\min(g_t(x), 1), -1\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{hard-tanh} | hèn tán | Hard-Tanh激活函数 |
| x | x | 输入 |
| g_t | g xiā biāo t | Tanh函数的一阶近似 |
| \max | zuì dà | 最大值函数 |
| \min | zuì xiǎo | 最小值函数 |

**一句话解释：**
Hard-Tanh函数通过限制Tanh函数线性近似的输出范围在[-1,1]。

**简单应用场景：**
在需要快速激活函数的应用中替代Tanh。

---

### 公式 (4.15)：Hard-Tanh的等价形式 / Hard-Tanh Equivalent Form

$$
= \max\left(\min(x, 1), -1\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \max | zuì dà | 最大值函数 |
| \min | zuì xiǎo | 最小值函数 |
| x | x | 输入 |

**一句话解释：**
由于Tanh的一阶近似就是x，Hard-Tanh简化为直接截断x的值到[-1,1]范围内。

**简单应用场景：**
编程实现中的显式计算形式。

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
| \text{ReLU} | ruì lú | 修正线性单元激活函数 |
| x | x | 输入 |

**一句话解释：**
ReLU函数对负值输出0，对正值保持原值，是现代深度学习中最常用的激活函数。

**简单应用场景：**
几乎所有现代深度神经网络的隐层都使用ReLU或其变体作为激活函数。

---

### 公式 (4.17)：ReLU的等价形式 / ReLU Equivalent Form

$$
= \max(0, x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \max | zuì dà | 最大值函数 |
| x | x | 输入 |

**一句话解释：**
ReLU可以简洁地表示为输入和0的最大值。

**简单应用场景：**
编程和数学计算中的标准表示。

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
| \text{LeakyReLU} | lì kì ruì lú | 带泄漏的修正线性单元 |
| x | x | 输入 |
| \gamma | gā mǎ | 负值部分的斜率系数 |

**一句话解释：**
Leaky ReLU允许负值部分有一个小的非零斜率γ，解决ReLU中负值完全变为0的问题。

**简单应用场景：**
在ReLU因神经元死亡导致性能下降的情况下使用。

---

### 公式 (4.19)：Leaky ReLU等价形式 / Leaky ReLU Equivalent Form

$$
= \max(0, x) + \gamma \min(0, x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \max | zuì dà | 最大值函数 |
| \min | zuì xiǎo | 最小值函数 |
| x | x | 输入 |
| \gamma | gā mǎ | 负值部分的斜率系数 |

**一句话解释：**
将Leaky ReLU表示为正部分和负部分的加权和。

**简单应用场景：**
便于数值计算和自动微分框架中的实现。

---

### 公式 (4.20)：Leaky ReLU的特殊情况 / Leaky ReLU Special Case

$$
\text{LeakyReLU}(x) = \max(x, \gamma x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{LeakyReLU} | lì kì ruì lú | 带泄漏的修正线性单元 |
| x | x | 输入 |
| \gamma | gā mǎ | 负值部分的斜率系数 |
| \max | zuì dà | 最大值函数 |

**一句话解释：**
Leaky ReLU也可以表示为x和γx的最大值，当x为负时选择γx。

**简单应用场景：**
代码实现中的快速计算形式。

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
| \text{PReLU} | pēi rù shù huà ruì lú | 参数化修正线性单元 |
| x | x | 输入 |
| \gamma | gā mǎ | 可学习的斜率参数 |

**一句话解释：**
PReLU是Leaky ReLU的推广，其中斜率参数γ不再固定，而是在训练过程中学习。

**简单应用场景：**
当固定的Leaky ReLU斜率不能满足需求时，使用PReLU让模型自己学习最优的斜率。

---

### 公式 (4.22)：PReLU等价形式 / PReLU Equivalent Form

$$
= \max(0, x) + \gamma \min(0, x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \max | zuì dà | 最大值函数 |
| \min | zuì xiǎo | 最小值函数 |
| x | x | 输入 |
| \gamma | gā mǎ | 可学习的斜率参数 |

**一句话解释：**
PReLU的数学形式与Leaky ReLU相同，但参数γ是可学习的。

**简单应用场景：**
用于反向传播中对γ的梯度计算。

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
| \text{ELU} | ē ēl yóu | 指数线性单元 |
| x | x | 输入 |
| \gamma | gā mǎ | 小于0时的参数 |
| \exp | ē | 指数函数 |

**一句话解释：**
ELU在正值处为线性，在负值处为指数函数，提供光滑的非线性激活。

**简单应用场景：**
需要更平滑的负值部分且想降低计算复杂度时的折中方案。

---

### 公式 (4.24)：ELU等价形式 / ELU Equivalent Form

$$
= \max(0, x) + \min(0, \gamma(\exp(x) - 1))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \max | zuì dà | 最大值函数 |
| \min | zuì xiǎo | 最小值函数 |
| x | x | 输入 |
| \gamma | gā mǎ | 小于0时的参数 |
| \exp | ē | 指数函数 |

**一句话解释：**
将ELU分解为正部分和负部分的组合形式。

**简单应用场景：**
便于自动微分框架的实现。

---

### 公式 (4.25)：Softplus函数 / Softplus Activation

$$
\text{Softplus}(x) = \log(1 + \exp(x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{Softplus} | ruǎn jiā | 软加函数 |
| x | x | 输入 |
| \log | duì shù | 自然对数 |
| \exp | ē | 指数函数 |

**一句话解释：**
Softplus函数是ReLU的光滑近似，整个定义域上都可导。

**简单应用场景：**
在需要处处可导且希望避免ReLU硬截断的场景中使用。

---

### 公式 (4.26)：Swish函数 / Swish Activation

$$
\text{swish}(x) = x\sigma(\beta x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{swish} | swish | 自门控激活函数 |
| x | x | 输入 |
| \sigma | xī gé mǎ | Logistic函数 |
| \beta | bèi tǎ | 可学习或固定的参数 |

**一句话解释：**
Swish函数将输入与其Logistic变换的乘积作为输出，实现自门控机制。

**简单应用场景：**
在Google的EfficientNet等现代架构中作为激活函数使用。

---

### 公式 (4.27)：GELU函数 / Gaussian Error Linear Unit

$$
\text{GELU}(x) = xP(X \leq x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{GELU} | jiāo ē ēl yóu | 高斯误差线性单元 |
| x | x | 输入 |
| P | gài lǜ | 概率 |
| X | dā | 高斯分布随机变量 |

**一句话解释：**
GELU函数将输入乘以其在标准正态分布中的累积分布函数值。

**简单应用场景：**
在Transformer模型（如BERT）中广泛使用。

---

### 公式 (4.28)：GELU的Tanh近似 / GELU Tanh Approximation

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{GELU} | jiāo ē ēl yóu | 高斯误差线性单元 |
| x | x | 输入 |
| \tanh | tán huo | 双曲正切函数 |
| \sqrt{\frac{2}{\pi}} | píng fāng gēn | 常数系数 |

**一句话解释：**
使用Tanh函数对GELU的累积分布函数进行多项式近似。

**简单应用场景：**
在计算GELU时提供更快的近似版本。

---

### 公式 (4.29)：GELU的Sigmoid近似 / GELU Sigmoid Approximation

$$
\text{GELU}(x) \approx x\sigma(1.702x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{GELU} | jiāo ē ēl yóu | 高斯误差线性单元 |
| x | x | 输入 |
| \sigma | xī gé mǎ | Logistic函数 |

**一句话解释：**
GELU的另一种近似形式，使用Logistic函数实现。

**简单应用场景：**
计算效率更高，但精度略低于Tanh近似。

---

### 公式 (4.30)：Maxout单元的输入计算 / Maxout Unit Input

$$
z_k = \boldsymbol{\omega}_k^T \boldsymbol{x} + b_k
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| z_k | zī xiā biāo k | 第k个分支的输入 |
| \boldsymbol{\omega}_k | ōu mǐ gā xiā biāo k | 第k个分支的权重向量 |
| \boldsymbol{x} | x | 输入向量 |
| b_k | bī xiā biāo k | 第k个分支的偏置 |

**一句话解释：**
Maxout单元由K个独立的线性变换组成，每个分支计算一个加权和加偏置。

**简单应用场景：**
Maxout单元的基本计算步骤，为之后的取最大值操作准备输入。

---

### 公式 (4.31)：Maxout单元 / Maxout Unit

$$
\text{maxout}(x) = \max_{k \in [1,K]} (z_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \text{maxout} | mǎ kè shù chū | 最大输出单元 |
| x | x | 输入向量 |
| z_k | zī xiā biāo k | 第k个分支的输出 |
| K | dà xiě k | 分支数量 |
| \max | zuì dà | 最大值函数 |

**一句话解释：**
Maxout单元计算K个不同线性变换中的最大值，是ReLU的推广。

**简单应用场景：**
在深度学习中用作灵活的激活函数，可以学习分段线性的激活模式。

---

### 公式 (4.32)：前馈网络信息传递 / Feedforward Network Forward Pass

$$
z^{(l)} = \boldsymbol{W}^{(l)} a^{(l-1)} + \boldsymbol{b}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| z^{(l)} | zī shàng l | 第l层的净输入 |
| \boldsymbol{W}^{(l)} | dà xiě w shàng l | 第l层的权重矩阵 |
| a^{(l-1)} | ā shàng l jiǎn 1 | 第l-1层的激活输出 |
| \boldsymbol{b}^{(l)} | bī shàng l | 第l层的偏置向量 |

**一句话解释：**
前馈网络第l层通过对前一层激活的线性变换（加权和加偏置）计算净输入。

**简单应用场景：**
神经网络前向传播中每一层的基本计算。

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

### 公式 (4.34)：前馈网络的合并形式 / Feedforward Network Combined Form

$$
z^{(l)} = \boldsymbol{W}^{(l)} f_l(z^{(l-1)}) + \boldsymbol{b}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| z^{(l)} | zī shàng l | 第l层的净输入 |
| \boldsymbol{W}^{(l)} | dà xiě w shàng l | 第l层的权重矩阵 |
| f_l | f xiā biāo l | 第l层的激活函数 |
| z^{(l-1)} | zī shàng l jiǎn 1 | 第l-1层的净输入 |
| \boldsymbol{b}^{(l)} | bī shàng l | 第l层的偏置向量 |

**一句话解释：**
将激活函数直接代入网络传递公式，表示第l层的输入如何从第l-1层的输入通过激活函数和线性变换得到。

**简单应用场景：**
紧凑的网络层计算表示。

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
| \boldsymbol{W}^{(l)} | dà xiě w shàng l | 第l层的权重矩阵 |
| a^{(l-1)} | ā shàng l jiǎn 1 | 第l-1层的激活输出 |
| \boldsymbol{b}^{(l)} | bī shàng l | 第l层的偏置向量 |

**一句话解释：**
前馈网络第l层的激活输出是通过对前一层激活的线性变换结果应用激活函数得到的。

**简单应用场景：**
神经网络前向传播的标准链式计算形式，也是反向传播的基础。


## 第4章：前馈神经网络

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

### 公式 (4.40)：Logistic 分类回归 / Logistic Classification Regression

$$
p(\boldsymbol{y} = 1| \boldsymbol{x}) = \boldsymbol{a}^{(L)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{y} = 1| \boldsymbol{x})$ | tiáo jiàn gài lǜ | 条件概率 |
| $\boldsymbol{a}^{(L)}$ | shū chū céng huó huà zhí | 输出层激活值 |

**一句话解释：**
输出层激活值表示样本属于正类的条件概率。

**简单应用场景：**
二分类任务中，网络输出值代表样本为正类的概率。

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

### 公式 (4.51)：计算偏导数 ∂z^(l)/∂w^(l)_{ij} / Computing ∂z^(l)/∂w^(l)_{ij}

$$
\frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{w}_{ij}^{(l)}} = \left[ \frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{w}_{i1}^{(l)}} \cdots \frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{w}_{iM_l}^{(l)}} \cdots \frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{w}_{i1}^{(l)}} \right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \boldsymbol{z}^{(l)}}{\partial \boldsymbol{w}_{ij}^{(l)}}$ | piàn dǎo shù jù zhèn | 偏导数矩阵 |

**一句话解释：**
计算线性组合对权重矩阵每个元素的导数。

**简单应用场景：**
反向传播中的中间梯度计算。

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

### 公式 (4.54)：Hadamard乘积定义 / Hadamard Product Definition

$$
\delta^{(l)} \triangleq f_l'(\boldsymbol{a}^{(l-1)}) \in \mathbb{R}^{1 \times M_l}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta^{(l)}$ | dé ěr tǎ xiàng liàng | 误差项向量 |
| $f_l'$ | huó huà hánshù dǎo shù | 激活函数导数 |
| $\boldsymbol{a}^{(l-1)}$ | qián yī céng huó huà zhí | 前一层激活值 |

**一句话解释：**
定义误差项为激活函数导数与上层误差的Hadamard乘积。

**简单应用场景：**
反向传播中误差项的计算。

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

### 公式 (4.58)：激活函数与z的关系 / Relationship between Activation and z

$$
\frac{\partial \boldsymbol{a}^{(l)}}{\partial \boldsymbol{z}^{(l)}} = f_l'(\boldsymbol{z}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{a}^{(l)}$ | huó huà zhí | 激活值 |
| $f_l'(\boldsymbol{z}^{(l)})$ | huó huà hánshù dǎo shù | 激活函数导数 |

**一句话解释：**
激活值对线性组合的导数就是激活函数的导数。

**简单应用场景：**
反向传播中链式法则的应用。

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

### 公式 (4.64)：损失对权重的梯度 / Gradient of Loss to Weights

$$
\frac{\partial \mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})}{\partial \boldsymbol{w}_{ij}^{(l)}} = f_l'(\boldsymbol{z}^{(l)}) \boldsymbol{a}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{w}_{ij}^{(l)}$ | quán zhòng | 权重 |
| $f_l'(\boldsymbol{z}^{(l)})$ | huó huà dǎo shù | 激活导数 |

**一句话解释：**
权重梯度等于当前层误差与上层激活值的乘积。

**简单应用场景：**
反向传播中权重梯度的计算。

---

### 公式 (4.65)：权重梯度向量形式 / Vector Form of Weight Gradient

$$
= \left[ 0, \cdots, \delta_j^{(l)}, \cdots, 0 \right] [\boldsymbol{a}^{(l)}, \boldsymbol{a}^{(l)}, \cdots, \boldsymbol{a}_M^{(l)}]^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta_j^{(l)}$ | dé ěr tǎ xiàng liàng | 误差向量 |
| $\boldsymbol{a}^{(l)}$ | huó huà zhí | 激活值 |

**一句话解释：**
权重梯度可表示为向量形式。

**简单应用场景：**
矩阵化反向传播计算。

---

### 公式 (4.66)：权重梯度简化 / Simplified Weight Gradient

$$
= \delta_j^{(l)} \boldsymbol{a}^{(l-1)T}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\delta_j^{(l)}$ | dé ěr tǎ | 误差项 |
| $\boldsymbol{a}^{(l-1)T}$ | qián yī céng huó huà zhí zhuǎnzhì | 前一层激活值转置 |

**一句话解释：**
权重梯度等于误差与上层激活值的外积。

**简单应用场景：**
反向传播中权重梯度矩阵的计算。

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

### 公式 (4.75)：∂f(x;w,b)/∂w 通过链式法则 / Chain Rule for ∂f/∂w

$$
\frac{\partial f(x;w,b)}{\partial w} = \frac{\partial f(x;w,b) \partial h_2 \partial h_3 \partial h_4 \partial h_5 \partial h_1}{\partial h_4 \partial h_3 \partial h_4 \partial h_3 \partial h_2 \partial h_1 \partial w}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_i$ | zhōng jiān biàn liàng | 中间变量 |
| $\frac{\partial}{\partial w}$ | piàn dǎo | 偏导数 |

**一句话解释：**
权重导数通过链式法则逐步计算。

**简单应用场景：**
自动微分的计算过程。

---

### 公式 (4.76)：∂f(x;w,b)/∂b 通过链式法则 / Chain Rule for ∂f/∂b

$$
\frac{\partial f(x;w,b)}{\partial b} = \frac{\partial f(x;w,b) \partial h_2 \partial h_3 \partial h_4 \partial h_3 \partial h_2}{\partial h_4 \partial h_3 \partial h_4 \partial h_3 \partial h_2 \partial b}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_i$ | zhōng jiān biàn liàng | 中间变量 |
| $b$ | piàn zhì | 偏置 |

**一句话解释：**
偏置导数通过链式法则逐步计算。

**简单应用场景：**
自动微分对偏置的计算。

---

### 公式 (4.77)：∂f(x;w,b)/∂w 具体值计算 / Specific Value Computation for ∂f/∂w

$$
\frac{\partial f(x;w,b)}{\partial w}\bigg|_{x=1,w=0,b=0} = \frac{\partial f(x;w,b) \partial h_2 \partial h_3 \partial h_4 \partial h_3 \partial h_2 \partial h_1}{\partial h_4 \partial h_3 \partial h_4 \partial h_3 \partial h_2 \partial h_1 \partial w}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x, w, b$ | biàn liàng zhí | 变量值 |

**一句话解释：**
将具体数值代入链式法则计算导数。

**简单应用场景：**
自动微分的具体数值计算。

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


## 第5章：卷积神经网络

### 公式 (5.1)：一维卷积 / 1D Convolution

$$
y_t = 1 \times x_t + 1/2 \times x_{t-1} + 1/4 \times x_{t-2}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $y_t$ | yī | 输出信号在时刻 t 的值 |
| $x_t$ | xī | 输入信号在时刻 t 的值 |

**一句话解释：**
当前输出是当前和过去多个输入信号的加权和，权重分别为 1, 0.5, 0.25。

**简单应用场景：**
时间序列数据的平滑处理或信号滤波。

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

### 公式 (5.6)：移动平均卷积的尺度变换 / Scaled Convolution with Moving Average

$$
x^{(t)} = x(t+1) + x(t-1) - 2x(t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x^{(t)}$ | xī | 差分结果 |
| $x(t)$ | xī | 时刻 t 的输入值 |

**一句话解释：**
计算信号的二阶差分，用于边缘检测或二阶导数估计。

**简单应用场景：**
图像处理中的边缘检测或信号的二阶导数计算。

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

### 公式 (5.12)：宽卷积 / Wide Convolution

$$
\boldsymbol{W} \otimes_{\Delta} \boldsymbol{W} \otimes \boldsymbol{X}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{W}$ | w | 卷积核 |
| $\otimes_{\Delta}$ | ōu | 宽卷积符号 |
| $\boldsymbol{X}$ | xī | 输入 |

**一句话解释：**
宽卷积是指在输入周围添加零填充的卷积方式。

**简单应用场景：**
需要保持或增加输出尺寸的卷积操作。

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

### 公式 (5.16)：卷积梯度的另一形式 / Alternative Form of Convolution Gradient

$$
= \sum_{i=1}^{M-U+1} \sum_{j=1}^{N-V+1} x_{i+u-1,j+v-1} \frac{\partial f(Y)}{\partial y_{ij}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_{ij}$ | xī | 输入元素 |
| $\frac{\partial f(Y)}{\partial y_{ij}}$ | piàn dǎo shù | 损失关于输出的梯度 |

**一句话解释：**
通过输入与输出梯度的卷积计算权重梯度。

**简单应用场景：**
CNN 反向传播算法。

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

### 公式 (5.18)：输入梯度 / Input Gradient

$$
\frac{\partial f(Y)}{\partial x_{ij}} = \sum_{i=1}^{M-U+1} \sum_{j=1}^{N-V+1} \frac{\partial y_{ij}}{\partial x_{ij}} \frac{\partial f(Y)}{\partial y_{ij}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_{ij}$ | xī | 输入元素 |
| $y_{ij}$ | yī | 输出元素 |
| $\frac{\partial f(Y)}{\partial y_{ij}}$ | piàn dǎo shù | 损失关于输出的梯度 |

**一句话解释：**
计算损失函数关于输入的梯度。

**简单应用场景：**
反向传播中传递梯度到前一层。

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

### 公式 (5.22)：全连接层卷积表示 / Fully Connected Layer as Convolution

$$
\boldsymbol{z}^{(l)} = \boldsymbol{w}^{(l)} \otimes \boldsymbol{x}^{(l-1)} + \boldsymbol{b}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}^{(l)}$ | z | 第 l 层的卷积输出 |
| $\boldsymbol{w}^{(l)}$ | w | 第 l 层的卷积核 |
| $\boldsymbol{x}^{(l-1)}$ | xī | 第 l-1 层的输出 |
| $\boldsymbol{b}^{(l)}$ | b | 第 l 层的偏置 |

**一句话解释：**
全连接层可以用卷积层表示，其中卷积核大小等于输入大小。

**简单应用场景：**
统一卷积和全连接层的表示方法。

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

### 公式 (5.28)：卷积层反向传播的导数公式 / Backpropagation Gradient for Convolution

$$
\boldsymbol{z}^{(l)} = \sum_{d=1}^{D} \boldsymbol{W}^{(l,d)} \otimes \boldsymbol{x}^{(l-1)} + \boldsymbol{b}^{(l)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}^{(l)}$ | z | 第 l 层卷积输出 |
| $\boldsymbol{W}^{(l,d)}$ | w | 第 l 层第 d 个输入通道的卷积核 |
| $\boldsymbol{x}^{(l-1)}$ | xī | 第 l-1 层输出 |

**一句话解释：**
卷积层的前向传播，对每个输入通道进行卷积并求和。

**简单应用场景：**
CNN 的卷积层计算。

---

### 公式 (5.29)：卷积层参数梯度 / Parameter Gradient for Convolution

$$
\frac{\partial \mathcal{C}}{\partial \boldsymbol{W}^{(l,d)}} = \frac{\partial \mathcal{C}}{\partial \boldsymbol{Z}^{(l)}} \otimes \boldsymbol{x}^{(l-1d)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{C}$ | dà xiě c | 损失函数 |
| $\boldsymbol{W}^{(l,d)}$ | w | 卷积核 |
| $\boldsymbol{Z}^{(l)}$ | z | 层输出 |

**一句话解释：**
卷积层权重的梯度等于损失关于输出的梯度与对应输入通道的互相关。

**简单应用场景：**
CNN 反向传播权重更新。

---

### 公式 (5.30)：隐藏层梯度计算 / Hidden Layer Gradient Calculation

$$
= \boldsymbol{\delta}^{(l)} \otimes \boldsymbol{x}^{(l-1d)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\delta}^{(l)}$ | dé ěr tǎ | 第 l 层的误差项 |
| $\boldsymbol{x}^{(l-1d)}$ | xī | 第 l-1 层第 d 个通道的输出 |

**一句话解释：**
卷积层权重梯度由误差项与输入的互相关组成。

**简单应用场景：**
反向传播算法中的梯度计算。

---

### 公式 (5.31)：偏置梯度 / Bias Gradient

$$
\frac{\partial \mathcal{C}}{\partial \boldsymbol{b}^{(l)}} = \sum_{i,j} \boldsymbol{\delta}^{(l)}_{i,j}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{C}$ | dà xiě c | 损失函数 |
| $\boldsymbol{b}^{(l)}$ | b | 第 l 层偏置 |
| $\boldsymbol{\delta}^{(l)}$ | dé ěr tǎ | 误差项 |

**一句话解释：**
偏置的梯度是所有误差项的求和。

**简单应用场景：**
CNN 反向传播中偏置的更新。

---

### 公式 (5.32)：上一层误差项（卷积层情形）/ Error Term for Previous Layer

$$
\boldsymbol{\delta}^{(l)} \triangleq \frac{\partial \mathcal{C}}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\delta}^{(l)}$ | dé ěr tǎ | 第 l 层误差项 |
| $\mathcal{C}$ | dà xiě c | 损失函数 |
| $\boldsymbol{z}^{(l)}$ | z | 第 l 层的卷积输出 |

**一句话解释：**
定义卷积层的误差项为损失函数关于卷积输出的梯度。

**简单应用场景：**
反向传播算法中的误差项定义。

---

### 公式 (5.33)：误差项链式法则 / Error Term Chain Rule

$$
= \frac{\partial x^{(l)} \partial z^{(l+1)}}{\partial z^{(l)} \partial x^{(l)}} \frac{\partial \mathcal{C}}{\partial z^{(l+1)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial}$ | piàn dǎo shù | 偏导数 |
| $\mathcal{C}$ | dà xiě c | 损失函数 |

**一句话解释：**
通过链式法则计算当前层的误差项。

**简单应用场景：**
反向传播梯度流传递。

---

### 公式 (5.34)：激活函数导数项 / Activation Function Derivative

$$
= f'(\boldsymbol{z}^{(l)}) \otimes \boldsymbol{\delta}^{(l+1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f'$ | f dǎo shù | 激活函数导数 |
| $\boldsymbol{z}^{(l)}$ | z | 第 l 层输出 |
| $\boldsymbol{\delta}^{(l+1)}$ | dé ěr tǎ | 第 l+1 层误差项 |

**一句话解释：**
激活函数的导数与下一层误差项的逐点乘积。

**简单应用场景：**
CNN 反向传播的激活层梯度计算。

---

### 公式 (5.35)：多卷积核的误差项 / Error Term for Multiple Kernels

$$
\boldsymbol{z}^{(l+1)} = \sum_{d=1}^{D} \boldsymbol{W}^{(l+1,d)} \otimes \boldsymbol{x}^{(l)} + \boldsymbol{b}^{(l+1)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}^{(l+1)}$ | z | 第 l+1 层输出 |
| $\boldsymbol{W}^{(l+1,d)}$ | w | 第 l+1 层第 d 个卷积核 |
| $\boldsymbol{x}^{(l)}$ | xī | 第 l 层输出 |

**一句话解释：**
多卷积核情形下的前向传播公式。

**简单应用场景：**
CNN 卷积层多通道输出。

---

### 公式 (5.36)：池化层误差项 / Pooling Layer Error Term

$$
\boldsymbol{\delta}^{(l)} \triangleq \frac{\partial \mathcal{C}}{\partial \boldsymbol{z}^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\delta}^{(l)}$ | dé ěr tǎ | 第 l 层误差项 |
| $\mathcal{C}$ | dà xiě c | 损失函数 |
| $\boldsymbol{z}^{(l)}$ | z | 第 l 层输出 |

**一句话解释：**
池化层的误差项定义为损失关于该层输出的梯度。

**简单应用场景：**
池化层反向传播。

---

### 公式 (5.37)：池化层梯度计算 / Pooling Layer Gradient

$$
= \frac{\partial x^{(l)}}{\partial z^{(l)}} \frac{\partial \mathcal{C}}{\partial x^{(l)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x^{(l)}$ | xī | 第 l 层输入 |
| $z^{(l)}$ | z | 第 l 层输出 |
| $\mathcal{C}$ | dà xiě c | 损失函数 |

**一句话解释：**
池化层梯度是输入导数与上一层梯度的乘积。

**简单应用场景：**
池化层反向传播梯度计算。

---

### 公式 (5.38)：卷积层的完整反向传播 / Complete Backpropagation for Convolution

$$
= f_l'(\boldsymbol{z}^{(l)}) \otimes \sum_{p=1}^{P} \left(\text{rot180}(\boldsymbol{W}^{(l+1,p)}) \otimes \frac{\partial \mathcal{C}}{\partial z^{(l+1)p}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_l'$ | f dǎo shù | 激活函数导数 |
| rot180 | luó tài bǎi bā shí | 180 度旋转 |
| $\boldsymbol{W}^{(l+1,p)}$ | w | 第 l+1 层第 p 个卷积核 |

**一句话解释：**
卷积层的反向梯度通过旋转卷积核与下层梯度的互相关得到。

**简单应用场景：**
CNN 反向传播的完整计算。

---

### 公式 (5.39)：等价形式 / Equivalent Form

$$
= f_l'(\boldsymbol{z}^{(l)}) \otimes \sum_{p=1}^{P} \left(\text{rot180}(\boldsymbol{W}^{(l+1,p)}) \otimes \boldsymbol{\delta}^{(l+1)p}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_l'$ | f dǎi shù | 激活导数 |
| rot180 | luó tài bǎi bā shí | 旋转 |
| $\boldsymbol{\delta}^{(l+1)p}$ | dé ěr tǎ | 误差项 |

**一句话解释：**
用误差项表示的等价反向传播公式。

**简单应用场景：**
CNN 反向传播算法的实现。

---

### 公式 (5.40)：LeNet-5 的链接表 / LeNet-5 Link Table

$$
\boldsymbol{Y}^p = f\left(\sum_{d=1}^{D} \boldsymbol{W}^{p,d} \otimes \boldsymbol{X}^d + \boldsymbol{b}^p\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{Y}^p$ | yī | 第 p 个输出特征图 |
| $f$ | f | 激活函数 |
| $D$ | dé | 输入通道数 |
| $\boldsymbol{W}^{p,d}$ | w | 卷积核 |

**一句话解释：**
LeNet-5 中某些层的连接只使用特定的输入通道来减少参数。

**简单应用场景：**
LeNet-5 等经典 CNN 的卷积层设计。

---

### 公式 (5.42)：仿射变换 / Affine Transformation

$$
\boldsymbol{z} = \boldsymbol{W} \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | z | 变换后的输出 |
| $\boldsymbol{W}$ | w | 变换矩阵 |
| $\boldsymbol{x}$ | xī | 输入 |

**一句话解释：**
通过矩阵乘法进行线性变换。

**简单应用场景：**
转置卷积的基础变换。

---

### 公式 (5.43)：逆变换 / Inverse Transformation

$$
\boldsymbol{x} = \boldsymbol{W}^T \boldsymbol{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | xī | 恢复的输入 |
| $\boldsymbol{W}^T$ | w | 变换矩阵的转置 |
| $\boldsymbol{z}$ | z | 输出 |

**一句话解释：**
仿射变换的逆变换通过转置矩阵进行。

**简单应用场景：**
转置卷积的实现。

---

### 公式 (5.44)：三维卷积 / 3D Convolution

$$
\boldsymbol{z} = \boldsymbol{w} \otimes \boldsymbol{x}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | z | 卷积输出 |
| $\boldsymbol{w}$ | w | 三维卷积核 |
| $\boldsymbol{x}$ | xī | 三维输入 |
| $\otimes$ | ōu | 卷积操作符 |

**一句话解释：**
对三维数据（如视频）进行卷积操作。

**简单应用场景：**
视频处理或三维医学影像分析。

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


## 第6章：循环神经网络

## 第6章：循环神经网络

### 公式 (6.1)：延时神经网络 / Time Delay Neural Network (TDNN)

$$
\boldsymbol{h}_t^{(l)} = f(\boldsymbol{h}_t^{(l-1)}, \boldsymbol{h}_{t-1}^{(l)}, \ldots, \boldsymbol{h}_{t-K}^{(l)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t^{(l)}$ | hóu huàn jìzhuàng | 第 $l$ 层在时刻 $t$ 的隐藏状态 |
| $f(\cdot)$ | fúnshù | 非线性激活函数 |
| $K$ | dàixiànghuà | 记忆深度（延时步数） |
| $M_l$ | céngdù | 第 $l$ 层神经元的数量 |

**一句话解释：**
延时神经网络通过利用历史时刻的隐藏状态信息来增强网络的记忆能力。

**简单应用场景：**
语音识别中利用过去多个时刻的特征来预测当前时刻的音素。

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

### 公式 (6.13)：隐藏层嵌套变换 / Nested Hidden Layer Transformation

$$
s_t' = f(ACAs_{t-1} + Bx_t + b)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t'$ | xīnshēng zhuàngtài | 新生成的隐藏状态 |
| $A, C$ | quánzhòng | 权重矩阵 |
| $B$ | quánzhòng | 输入权重矩阵 |
| $b$ | piānzhì | 偏置向量 |
| $f(\cdot)$ | hánshu | 非线性激活函数 |

**一句话解释：**
通过嵌套矩阵变换来实现更复杂的隐藏状态转移。

**简单应用场景：**
构造具有多个线性变换层的深层循环神经网络。

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

### 公式 (6.18)：分离的隐藏和输出状态矩阵表示 / Separated Hidden and Output States Matrix Form

$$
\begin{bmatrix} s_t' \\ y_t' \end{bmatrix} = f\left(\begin{bmatrix} AC & 0 \\ A'C & 0 \end{bmatrix} \begin{bmatrix} s_{t-1} \\ y_{t-1} \end{bmatrix} + \begin{bmatrix} B \\ B' \end{bmatrix} x_t + \begin{bmatrix} b \\ b' \end{bmatrix}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s_t'$ | yǐncáng zhuàngtài | 新生成的隐藏状态 |
| $y_t'$ | shūchū zhuàngtài | 输出隐藏状态 |
| $A, A', B, B', C$ | quánzhòng | 权重矩阵 |
| $b, b'$ | piānzhì | 偏置向量 |
| $f(\cdot)$ | hánshu | 非线性激活函数 |

**一句话解释：**
使用分块矩阵表示隐藏层和输出层的联合状态转移和激活。

**简单应用场景：**
统一表示循环神经网络中的多个不同类型的状态。

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

### 公式 (6.33)：指示向量记号 / Indicator Vector Notation

$$\mathbb{I}_i(x) = [0, \cdots, x, \cdots, 0]$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{I}_i(x)$ | zhǐ shì xiàng liàng | 指示向量，第i位为x |
| $x$ | xiào | 标量值 |

**一句话解释：**
指示向量在第i个位置为x，其余位置为0。

**简单应用场景：**
简化BPTT梯度计算表达式。

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

### 公式 (6.35)：误差项链式法则 / Error Term Chain Rule

$$\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{z}_k} = \frac{\partial \boldsymbol{h}_k}{\partial \boldsymbol{z}_k} \frac{\partial \boldsymbol{z}_{k+1}}{\partial \boldsymbol{h}_k} \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{z}_{k+1}}$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_k$ | héi kè | 第k时刻的隐状态 |
| $\boldsymbol{z}_k$ | zhī kè | 第k时刻的净输入 |

**一句话解释：**
使用链式法则将误差项从k+1时刻反向传播到k时刻。

**简单应用场景：**
时间反向传播（BPTT）的递推关系。

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

### 公式 (6.45)：参数偏导元素乘法 / Parameter Derivative Hadamard

$$\frac{\partial \boldsymbol{h}_{t+1}}{\partial u_{ij}} = \left(\mathbb{I}_i([\boldsymbol{h}_t]_j) + \frac{\partial \boldsymbol{h}_t^T}{\partial u_{ij}}\boldsymbol{U}^T\right) \odot (f'(\boldsymbol{z}_{t+1}))^T$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\odot$ | yuánsù chéng jī | 元素乘积（Hadamard乘积） |
| $f'(\boldsymbol{z}_{t+1})$ | dǎo shù | 激活函数导数向量 |

**一句话解释：**
使用元素乘积表示RTRL的梯度前向传播。

**简单应用场景：**
实时循环学习中的梯度计算。

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

### 公式 (6.49)：跳连残差 / Skip Connection Residual

$$\boldsymbol{h}_t = \boldsymbol{h}_{t-1} + g(\boldsymbol{x}_t; \boldsymbol{\theta})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_t$ | héi tī | 第t时刻的隐状态 |
| $g(\cdot)$ | gē | 非线性变换函数 |
| $\boldsymbol{x}_t$ | xiào tī | 第t时刻的输入 |
| $\boldsymbol{\theta}$ | tā tǎ | 参数 |

**一句话解释：**
通过跳连使隐状态和上一时刻为线性关系，缓解梯度消失问题。

**简单应用场景：**
改进循环网络的长期依赖学习能力。

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

### 公式 (6.79)：双向递归神经网络前向 / Bidirectional Recursive Forward

$$\overleftarrow{\boldsymbol{h}}_i = f(\overleftarrow{\boldsymbol{h}}_{\text{left}(i)}, \overleftarrow{\boldsymbol{h}}_{\text{right}(i)})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\overleftarrow{\boldsymbol{h}}_i$ | zuǒ xiàng gé zhuàng tài | 前向隐状态 |
| $f(\cdot)$ | jī huà hánshù | 变换函数 |
| $\text{left}(i), \text{right}(i)$ | zuǒ zǐ, yòu zǐ | 左右子节点 |

**一句话解释：**
双向递归神经网络的前向传播方向，节点由其左右子节点的表示合成。

**简单应用场景：**
树结构上的前向信息传播。

---

### 公式 (6.80)：双向递归神经网络反向 / Bidirectional Recursive Backward

$$\overrightarrow{\boldsymbol{h}}_i = f(\overrightarrow{\boldsymbol{h}}_{\text{parent}(i)})$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\overrightarrow{\boldsymbol{h}}_i$ | yòu xiàng gé zhuàng tài | 反向隐状态 |
| $f(\cdot)$ | jī huà hánshù | 变换函数 |
| $\text{parent}(i)$ | fù jié diǎn | 父节点 |

**一句话解释：**
双向递归神经网络的反向传播，节点表示由其父节点的信息更新。

**简单应用场景：**
树结构上的反向信息传播。

---

### 公式 (6.81)：双向递归神经网络输出 / Bidirectional Recursive Output

$$\boldsymbol{h}_i = \overleftarrow{\boldsymbol{h}}_i \oplus \overrightarrow{\boldsymbol{h}}_i$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{h}_i$ | gé zhuàng tài | 最终隐状态 |
| $\overleftarrow{\boldsymbol{h}}_i$ | zuǒ xiàng | 前向隐状态 |
| $\overrightarrow{\boldsymbol{h}}_i$ | yòu xiàng | 反向隐状态 |
| $\oplus$ | pīn jiē | 向量拼接 |

**一句话解释：**
双向递归神经网络的最终节点表示由前向和反向信息拼接而成，包含树的完整结构信息。

**简单应用场景：**
双向递归神经网络的输出，利用完整的树结构信息。

---

## 总结

本文档提取了《神经网络与深度学习》第六章（循环神经网络）第二部分（页面130-148）的所有60个编号公式，从(6.22)到(6.81)。

主要涵盖内容：
- **序列建模** (6.22-6.27)：序列到类别、序列标注、编码器-解码器模式
- **参数学习** (6.28-6.48)：BPTT、RTRL算法及梯度计算
- **长程依赖问题** (6.49-6.50)：跳连改进思想
- **LSTM网络** (6.51-6.64)：长短期记忆、门控机制、变体设计
- **GRU网络** (6.65-6.69)：简化的门控循环单元
- **深层循环网络** (6.70)：堆叠RNN
- **双向循环网络** (6.71-6.73)：双向信息融合
- **递归神经网络** (6.74-6.81)：树结构扩展和双向递归

每个公式都包含符号表、简洁说明和实际应用场景，便于深度学习从业者的快速参考和理解。

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

### 公式 (7.10)：三角形循环学习率 / Triangular Cyclic Learning Rate

$$
m = \left\lfloor \frac{t}{2\Delta T} \right\rfloor
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $m$ | xún huán cì shù | 循环次数 |
| $t$ | tī dài cì shù | 当前迭代次数 |
| $\Delta T$ | xún huán zhōu qī | 循环周期 |
| $\lfloor \cdot \rfloor$ | xiàng xià qū zhěng | 向下取整函数 |

**一句话解释：**
计算当前迭代属于第几个完整的学习率循环周期。

**简单应用场景：**
在循环学习率策略中，确定当前位置所在的循环周期。

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

### 公式 (7.13)：带热重启的循环学习率 / Cyclic Learning Rate with Warm Restarts

$$
\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{\pi t_m}{T_m}\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_t$ | xué xí lǜ | 第t次迭代的学习率 |
| $\alpha_{\min}$ | zuì xiǎo xué xí lǜ | 最小学习率 |
| $\alpha_{\max}$ | zuì dà xué xí lǜ | 最大学习率 |
| $t_m$ | xún huán nèi tī dài cì shù | 当前循环周期内的迭代次数 |
| $T_m$ | xún huán zhōu qī | 循环周期长度 |

**一句话解释：**
使用余弦函数在每个循环周期内平滑调整学习率，模拟热重启效果。

**简单应用场景：**
在长期训练中使用SGDR(随机梯度下降与热重启)策略，周期性地重启优化器。

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

### 公式 (7.22)：Nesterov加速梯度-中间变量 / NAG Intermediate Variable

$$
\boldsymbol{\theta} = \boldsymbol{\theta}_{t-1} + \rho\Delta\boldsymbol{\theta}_{t-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | cān shù | 中间参数值 |
| $\boldsymbol{\theta}_{t-1}$ | cān shù | 第t-1次迭代的参数 |
| $\rho$ | rù | 动量系数 |
| $\Delta\boldsymbol{\theta}_{t-1}$ | cān shù gèng xīn liàng | 第t-1次迭代的参数更新量 |

**一句话解释：**
Nesterov加速梯度先沿着动量方向向前看一步，在该位置计算梯度。

**简单应用场景：**
相比普通动量法，NAG能更好地预测下一步的位置并据此调整更新方向。

---

### 公式 (7.23)：Nesterov加速梯度-梯度计算 / NAG Gradient Calculation

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta} - \alpha\boldsymbol{g}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}_t$ | cān shù | 第t次迭代的参数 |
| $\boldsymbol{\theta}$ | zhōng jiān cān shù | 中间参数值 |
| $\alpha$ | āl fǎ | 学习率 |
| $\boldsymbol{g}_t$ | tī dù | 在中间位置处计算的梯度 |

**一句话解释：**
在中间位置处计算梯度，然后沿负梯度方向更新参数。

**简单应用场景：**
实现Nesterov加速梯度算法的参数更新步骤。

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

### 公式 (7.42)：权重方差要求 / Weight Variance Requirement

$$
\text{var}(w^{(l)}_i) = \frac{\text{var}(a^{(l)})}{M_{l-1}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{var}(w^{(l)}_i)$ | quán zhòng fāng chā | 权重方差 |
| $\text{var}(a^{(l)})$ | huó huà zhí fāng chā | 激活值方差 |
| $M_{l-1}$ | qián yī céng shén jīng yuán gè shù | 前一层神经元个数 |

**一句话解释：**
为了保持激活值方差在层间传递时不变，权重方差应与前一层神经元个数成反比。

**简单应用场景：**
Xavier初始化中用于计算每一层权重的初始化范围。

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



## 第7章 优化算法 (Chapter 7: Optimization Algorithms)

### 公式 (7.32)：优化算法小结 - 梯度更新 / Summary of Optimization - Gradient Update

$$
\Delta\boldsymbol{\theta}_t = -\frac{\alpha_t}{\sqrt{G_t + \varepsilon}} \boldsymbol{M}_t
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Delta\boldsymbol{\theta}_t$ | dé-ěl-tà shèta t | 第 $t$ 步的参数更新量 |
| $\alpha_t$ | ā-ěr-fā-ā t | 第 $t$ 步的学习率 |
| $G_t$ | jī-dé t | 梯度平方的累积值（用于自适应学习率） |
| $\varepsilon$ | yī-pū-sī-lóng | 防止分母为零的小常数 |
| $\boldsymbol{M}_t$ | ěm t | 第 $t$ 步的一阶矩（动量项）或梯度 |

**一句话解释：**
通过自适应学习率和动量项来更新参数，其中分母 $\sqrt{G_t + \varepsilon}$ 实现了不同参数的自适应缩放。

**简单应用场景：**
在实际训练神经网络时，根据不同参数的梯度幅度自动调整学习率，避免某些参数更新过快或过慢。

---

### 公式 (7.33)：优化算法小结 - 缩放系数 / Summary of Optimization - Scaling Coefficient

$$
G_t = \psi(g_1, \cdots, g_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G_t$ | jī-dé t | 梯度平方累积项（如 Adam 中的二阶矩） |
| $\psi$ | pū-sī | 累积函数（可以是平方、指数移动平均等） |
| $g_1, \cdots, g_t$ | jī-lì yī zhì t | 从第1步到第 $t$ 步的梯度序列 |

**一句话解释：**
$G_t$ 是对历史梯度进行累积或统计处理后的结果，用于计算自适应的学习率。

**简单应用场景：**
在 AdaGrad 算法中，$G_t$ 是梯度平方的累积和；在 Adam 中，$G_t$ 是梯度平方的指数移动平均。

---

### 公式 (7.34)：优化算法小结 - 矩估计 / Summary of Optimization - Moment Estimation

$$
\boldsymbol{M}_t = \varphi(g_1, \cdots, g_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{M}_t$ | ěm t | 梯度的矩估计（一阶矩或其他统计量） |
| $\varphi$ | fī | 矩计算函数（可以是梯度本身、指数移动平均等） |
| $g_1, \cdots, g_t$ | jī-lì yī zhì t | 从第1步到第 $t$ 步的梯度序列 |

**一句话解释：**
$\boldsymbol{M}_t$ 是梯度序列的某种统计处理结果，通常用作动量项来平滑参数更新。

**简单应用场景：**
在 Momentum 方法中，$\boldsymbol{M}_t$ 是梯度的指数移动平均；在 Adam 中，$\boldsymbol{M}_t$ 是梯度的一阶矩估计。

---

### 公式 (7.45) 至 (7.∞)：

No additional formulas found in pages 172-185 beyond formula 7.44. The chapter concludes with formula 7.34 as a summary formula, and the subsequent pages contain parameter initialization section (7.3) which does not include numbered formulas (7.45) onwards in this range.

---

## 第9章 无监督学习 (

## 第8章：注意力机制与外部记忆

---

### 公式 (8.1)：注意力分布 / Attention Distribution

$$
\alpha_n = p(z = n | \boldsymbol{X}, \boldsymbol{q}) = \frac{\exp(s(\boldsymbol{x}_n, \boldsymbol{q}))}{\sum_{j=1}^{N} \exp(s(\boldsymbol{x}_j, \boldsymbol{q}))}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_n$ | ālf fǎ | 第n个位置的注意力权重 |
| $\boldsymbol{X}$ | dà xiě X | 输入信息矩阵 |
| $\boldsymbol{q}$ | xiǎo xiě q | 查询向量 |
| $s(\boldsymbol{x}_n, \boldsymbol{q})$ | s | 注意力打分函数 |
| $N$ | dà xiě N | 输入信息的总数 |
| $\exp$ | zhǐ shù | 指数函数 |

**一句话解释：**
注意力分布是通过对各个输入位置的打分进行softmax归一化得到的概率分布，用来指示模型应该关注哪个位置。

**简单应用场景：**
在机器翻译中，当翻译当前词语时，计算出对源语言句子中每个词的注意力分布，权重高的词语对翻译贡献更大。

---

### 公式 (8.2)：加性模型 / Additive Model

$$
s(\boldsymbol{x}, \boldsymbol{q}) = \boldsymbol{v}^\top \tanh(\boldsymbol{Wx} + \boldsymbol{Uq})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(\boldsymbol{x}, \boldsymbol{q})$ | s | 打分函数值 |
| $\boldsymbol{v}$ | xiǎo xiě v | 权重向量 |
| $\boldsymbol{W}$ | dà xiě W | 输入权重矩阵 |
| $\boldsymbol{U}$ | dà xiě U | 查询权重矩阵 |
| $\boldsymbol{x}$ | xiǎo xiě x | 输入向量 |
| $\boldsymbol{q}$ | xiǎo xiě q | 查询向量 |
| $\tanh$ | shuāng qū xiàn hánshù | 双曲正切激活函数 |

**一句话解释：**
加性模型通过将输入和查询进行线性变换后相加，再通过tanh激活和权重向量投影来计算注意力打分。

**简单应用场景：**
在神经机器翻译的Bahdanau注意力机制中使用，可以有效地捕捉输入和查询之间的非线性关系。

---

### 公式 (8.3)：点积模型 / Dot-Product Model

$$
s(\boldsymbol{x}, \boldsymbol{q}) = \boldsymbol{x}^\top \boldsymbol{q}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(\boldsymbol{x}, \boldsymbol{q})$ | s | 打分函数值 |
| $\boldsymbol{x}$ | xiǎo xiě x | 输入向量 |
| $\boldsymbol{q}$ | xiǎo xiě q | 查询向量 |
| $^\top$ | zhuǎnzhì | 转置操作 |

**一句话解释：**
点积模型直接计算输入向量和查询向量的内积作为相似度打分，计算效率高。

**简单应用场景：**
在多头注意力机制中广泛使用，是Transformer模型的核心打分机制。

---

### 公式 (8.4)：缩放点积模型 / Scaled Dot-Product Model

$$
s(\boldsymbol{x}, \boldsymbol{q}) = \frac{\boldsymbol{x}^\top \boldsymbol{q}}{\sqrt{D}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(\boldsymbol{x}, \boldsymbol{q})$ | s | 打分函数值 |
| $\boldsymbol{x}$ | xiǎo xiě x | 输入向量 |
| $\boldsymbol{q}$ | xiǎo xiě q | 查询向量 |
| $D$ | dà xiě D | 向量维度 |
| $\sqrt{D}$ | gēnghào D | 维度的平方根 |

**一句话解释：**
在点积模型基础上加入维度的平方根作为缩放因子，防止点积值过大导致softmax梯度消失。

**简单应用场景：**
标准Transformer模型中的自注意力层使用这个打分函数，是现代深度学习的基础构件。

---

### 公式 (8.5)：双线性模型 / Bilinear Model

$$
s(\boldsymbol{x}, \boldsymbol{q}) = \boldsymbol{x}^\top \boldsymbol{Wq}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $s(\boldsymbol{x}, \boldsymbol{q})$ | s | 打分函数值 |
| $\boldsymbol{x}$ | xiǎo xiě x | 输入向量 |
| $\boldsymbol{W}$ | dà xiě W | 权重矩阵 |
| $\boldsymbol{q}$ | xiǎo xiě q | 查询向量 |
| $^\top$ | zhuǎnzhì | 转置操作 |

**一句话解释：**
双线性模型在查询向量上应用一个权重矩阵变换，然后与输入向量进行点积，提供了更灵活的参数化方式。

**简单应用场景：**
在关系抽取、信息检索等需要学习输入和查询之间复杂关系的任务中使用。

---

### 公式 (8.6)：加权求和 / Weighted Sum

$$
\text{att}(\boldsymbol{X}, \boldsymbol{q}) = \sum_{n=1}^{N} \alpha_n \boldsymbol{x}_n
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{att}(\boldsymbol{X}, \boldsymbol{q})$ | ā tì | 注意力加权输出 |
| $\alpha_n$ | ālf fǎ | 第n个位置的注意力权重 |
| $\boldsymbol{x}_n$ | xiǎo xiě x | 第n个输入向量 |
| $N$ | dà xiě N | 输入总数 |

**一句话解释：**
注意力输出是所有输入信息的加权和，权重由注意力分布决定，高权重的输入对输出贡献更大。

**简单应用场景：**
在序列模型中，用于根据注意力权重对源序列的隐层状态进行加权汇聚，生成上下文向量。

---

### 公式 (8.7)：软性注意力机制期望 / Expectation in Soft Attention Mechanism

$$
\text{att}(\boldsymbol{X}, \boldsymbol{q}) = \mathbb{E}_{z \sim p(z|q)} [\boldsymbol{x}_z]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{att}(\boldsymbol{X}, \boldsymbol{q})$ | ā tì | 注意力加权输出 |
| $\mathbb{E}$ | qī wàng | 期望操作符 |
| $z$ | xiǎo xiě z | 位置随机变量 |
| $p(z\|q)$ | gài lǜ | 给定查询q时位置z的概率分布 |
| $\boldsymbol{x}_z$ | xiǎo xiě x | 位置z处的向量 |

**一句话解释：**
软性注意力机制可以理解为在注意力分布上求期望，即对所有位置向量的概率加权平均。

**简单应用场景：**
提供软性注意力机制的概率论解释，说明注意力输出是对所有输入的随机加权组合。

---



## 注意力机制 (Attention Mechanism)

### 公式 (8.8)：硬性注意力 / Hard Attention

$$
\text{att}(X, q) = x_a
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $X$ | X | 输入序列 |
| $q$ | q | 查询向量 |
| $x_a$ | x zǐ | 选中的输入向量 |
| $a$ | ā | 注意位置索引 |

**一句话解释：**
硬性注意力从输入序列中选择单一最相关的向量，其中 $a = \arg\max_n \alpha_n$。

**简单应用场景：**
在视觉问答任务中，从图像的多个区域中选择最相关的单个区域进行处理。

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

### 公式 (8.28)：多跳记忆读取 / Multi-Hop Memory Reading

$$
r^{(k)} = \sum_{n=1}^{N} \text{softmax}\left((a_n^{(j)})^T q^{(k)}\right) c_n^{(k)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r^{(k)}$ | dì k gè tiào tiào de dú qǔ xiàng liàng | 第 k 跳的读取向量 |
| $a_n^{(j)}$ | dì k tiào tiào dì j gè jì yì de dì zhǐ | 第 k 跳第 j 个记忆的地址向量 |
| $c_n^{(k)}$ | dì k gè tiào tiào dì n gè jì yì de nèi rónɡ | 第 k 跳第 n 个记忆的内容向量 |

**一句话解释：**
在多跳推理的第 k 跳，根据更新后的查询从记忆中读取新的内容向量。

**简单应用场景：**
复杂推理任务中，模型需要多次访问记忆库来综合多个事实。

---

## 神经图灵机 (Neural Turing Machine)

### 公式 (8.29)：内容寻址 / Content Addressing

$$
a_{t,n} = \text{softmax}\left(s(m_n, q_t)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $a_{t,n}$ | dì t gè shí jiānbù dì n gè jì yì de zhù yì lì | 第 t 时刻第 n 个记忆位置的注意力 |
| $m_n$ | dì n gè jì yì nèi rónɡ | 第 n 个记忆单元的内容 |
| $q_t$ | dì t gè shí jiānbù de chá xun xiàng liàng | 第 t 时刻的查询向量 |
| $s(\cdot)$ | xiāng hóngrónɡ xìng dǎ fēn | 相容性打分函数 |

**一句话解释：**
计算查询向量与所有记忆单元内容的相似度，通过Softmax得到内容寻址的权重分布。

**简单应用场景：**
神经图灵机在复制、排序等算法任务中，根据输入内容定位和访问相关记忆位置。

---

## 注释说明

- 所有向量使用黑体表示（如 $\boldsymbol{x}$）
- 矩阵使用大写字母表示
- 下标表示序列位置或元素索引
- Softmax函数进行概率归一化
- 注意力机制是多个公式的核心组件

## 第9章：无监督学习 / Unsupervised Learning

### 公式 (9.1)：主成分分析投影 / PCA Projection

$$
z^{(n)} = \boldsymbol{\omega}^T \mathbf{x}^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z^{(n)}$ | zī | 第n个样本的投影值 |
| $\boldsymbol{\omega}$ | ōu mǐ gā | 投影方向向量（单位向量） |
| $\mathbf{x}^{(n)}$ | xiāng | 第n个输入样本 |

**一句话解释：**
主成分分析通过投影方向向量ω与样本数据的内积，将高维数据投影到一维空间。

**简单应用场景：**
将高维图像数据投影到主要方向，用于数据可视化和降维。

---

### 公式 (9.2)：PCA目标函数（逐点形式） / PCA Objective Function (Per-Sample)

$$
\sigma(X; \boldsymbol{\omega}) = \frac{1}{N} \sum_{n=1}^{N} (\boldsymbol{\omega}^T \mathbf{x}^{(n)} - \boldsymbol{\omega}^T \overline{\mathbf{x}})^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | 方差 |
| $X$ | xiāng | 数据矩阵 |
| $N$ | ēn | 样本数量 |
| $\overline{\mathbf{x}}$ | xiāng píng jūn | 样本均值向量 |

**一句话解释：**
PCA的目标是最大化投影数据的方差，使得信息损失最少。

**简单应用场景：**
在回归或分类任务中使用PCA进行特征提取，减少计算复杂度。

---

### 公式 (9.3)：PCA方差（矩阵形式） / PCA Variance (Matrix Form)

$$
\sigma(X; \boldsymbol{\omega}) = \frac{1}{N} (\boldsymbol{\omega}^T X - \boldsymbol{\omega}^T \overline{\mathbf{x}} (\boldsymbol{\omega}^T - \boldsymbol{\omega}^T X)^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $X$ | xiāng | 数据矩阵 |
| $\boldsymbol{\omega}$ | ōu mǐ gā | 投影方向 |

**一句话解释：**
使用矩阵表示PCA的目标函数，便于计算和优化。

**简单应用场景：**
在矩阵运算库中实现PCA算法。

---

### 公式 (9.4)：协方差矩阵形式 / Covariance Matrix Form

$$
\sigma(X; \boldsymbol{\omega}) = \boldsymbol{\omega}^T \boldsymbol{\Sigma} \boldsymbol{\omega}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\Sigma}$ | xī gé mǎ | 协方差矩阵 |
| $\boldsymbol{\omega}$ | ōu mǐ gā | 投影方向 |

**一句话解释：**
PCA通过最大化投影数据的方差，等价于最大化$\boldsymbol{\omega}^T \boldsymbol{\Sigma} \boldsymbol{\omega}$。

**简单应用场景：**
使用协方差矩阵的特征值分解来求解PCA。

---

### 公式 (9.5)：PCA约束优化问题 / PCA Constrained Optimization

$$
\max_{\boldsymbol{\omega}} \boldsymbol{\omega}^T \boldsymbol{\Sigma} \boldsymbol{\omega} + \lambda (1 - \boldsymbol{\omega}^T \boldsymbol{\omega})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\omega}$ | ōu mǐ gā | 投影方向向量 |
| $\boldsymbol{\Sigma}$ | xī gé mǎ | 协方差矩阵 |
| $\lambda$ | lā mǔ dá | 拉格朗日乘数 |

**一句话解释：**
在单位向量约束（$\boldsymbol{\omega}^T \boldsymbol{\omega} = 1$）下，最大化投影方差。

**简单应用场景：**
使用拉格朗日乘数法求解PCA的优化问题。

---

### 公式 (9.6)：拉格朗日条件 / Lagrange Condition

$$
\boldsymbol{\Sigma} \boldsymbol{\omega} = \lambda \boldsymbol{\omega}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\Sigma}$ | xī gé mǎ | 协方差矩阵 |
| $\boldsymbol{\omega}$ | ōu mǐ gā | 投影方向 |
| $\lambda$ | lā mǔ dá | 特征值 |

**一句话解释：**
在极值点，协方差矩阵作用于投影方向向量等于特征值乘以该向量。

**简单应用场景：**
通过求协方差矩阵的特征值和特征向量来获得PCA的解。

---

### 公式 (9.7)：PCA方差与特征值关系 / PCA Variance and Eigenvalue

$$
\sigma(X; \boldsymbol{\omega}) = \boldsymbol{\omega}^T \boldsymbol{\Sigma} \boldsymbol{\omega} = \boldsymbol{\omega}^T \lambda \boldsymbol{\omega} = \lambda
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sigma$ | xī gé mǎ | 方差 |
| $\lambda$ | lā mǔ dá | 特征值 |
| $\boldsymbol{\omega}$ | ōu mǐ gā | 特征向量 |

**一句话解释：**
投影数据的方差等于协方差矩阵对应特征向量的特征值。

**简单应用场景：**
选择最大的k个特征值对应的特征向量作为前k个主成分。

---

### 公式 (9.8)：特征值分解 / Eigenvalue Decomposition

$$
\boldsymbol{\Sigma} \mathbf{W} = \mathbf{W} \text{diag}(\boldsymbol{\lambda})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\Sigma}$ | xī gé mǎ | 协方差矩阵 |
| $\mathbf{W}$ | dà wài | 特征向量矩阵 |
| $\boldsymbol{\lambda}$ | lā mǔ dá | 特征值向量 |

**一句话解释：**
协方差矩阵可以分解为特征向量矩阵与对角化特征值矩阵的乘积。

**简单应用场景：**
使用特征值分解快速计算PCA的主成分。

---

### 公式 (9.9)：稀疏编码（分解形式） / Sparse Coding (Decomposition Form)

$$
\mathbf{x} = \sum_{m=1}^{M} z_m \mathbf{a}_m
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{x}$ | xiāng | 输入数据向量 |
| $z_m$ | zī | 第m个编码系数 |
| $\mathbf{a}_m$ | ā | 第m个字典原子 |
| $M$ | ēm | 字典大小 |

**一句话解释：**
稀疏编码将输入数据表示为少数几个字典原子的加权组合。

**简单应用场景：**
用稀疏编码表示图像块，用于图像去噪或超分辨率重建。

---

### 公式 (9.10)：稀疏编码（矩阵形式） / Sparse Coding (Matrix Form)

$$
\mathbf{x} = \mathbf{A} \mathbf{z}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{x}$ | xiāng | 输入数据向量 |
| $\mathbf{A}$ | ā | 字典矩阵 |
| $\mathbf{z}$ | zī | 稀疏编码向量 |

**一句话解释：**
将输入数据表示为字典矩阵与稀疏系数向量的乘积。

**简单应用场景：**
在矩阵形式下实现稀疏编码的优化算法。

---

### 公式 (9.11)：稀疏编码的学习目标函数 / Sparse Coding Loss Function

$$
\mathcal{L}(\mathbf{A}, \mathbf{Z}) = \sum_{n=1}^{N} \left( \left\| \mathbf{x}^{(n)} - \mathbf{A} \mathbf{z}^{(n)} \right\|^2 + \eta \rho(\mathbf{z}^{(n)}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ēl | 损失函数 |
| $\mathbf{A}$ | ā | 字典矩阵 |
| $\mathbf{Z}$ | zī | 编码矩阵 |
| $N$ | ēn | 样本数量 |
| $\eta$ | ē tǎ | 稀疏性权重系数 |
| $\rho$ | ró | 稀疏性函数 |

**一句话解释：**
稀疏编码的学习目标是最小化重构误差和稀疏性惩罚项的和。

**简单应用场景：**
同时学习最优字典和样本的稀疏编码系数。

---

### 公式 (9.12)：L0范数 / L0 Norm

$$
\rho(\mathbf{z}) = \sum_{m=1}^{M} \mathbb{I}(|z_m| > 0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho$ | ró | 稀疏性度量 |
| $\mathbf{z}$ | zī | 编码向量 |
| $\mathbb{I}$ | yī | 示性函数 |
| $M$ | ēm | 向量维度 |

**一句话解释：**
L0范数计算向量中非零元素的个数，直接度量稀疏性。

**简单应用场景：**
在稀疏编码中使用L0范数作为稀疏性约束。

---

### 公式 (9.13)：L1范数 / L1 Norm

$$
\rho(\mathbf{z}) = \sum_{m=1}^{M} |z_m|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho$ | ró | 稀疏性度量 |
| $\mathbf{z}$ | zī | 编码向量 |
| $M$ | ēm | 向量维度 |

**一句话解释：**
L1范数（曼哈顿距离）是L0范数的凸松弛，在优化中更易处理。

**简单应用场景：**
在LASSO回归和稀疏编码中替代L0范数。

---

### 公式 (9.14)：对数范数 / Logarithmic Norm

$$
\rho(\mathbf{z}) = \sum_{m=1}^{M} \log(1 + z_m^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho$ | ró | 稀疏性度量 |
| $\mathbf{z}$ | zī | 编码向量 |
| $M$ | ēm | 向量维度 |

**一句话解释：**
对数范数通过对数函数平滑地惩罚非零系数，介于L0和L1之间。

**简单应用场景：**
在需要平滑稀疏性约束的稀疏编码变种中使用。

---

### 公式 (9.15)：指数范数 / Exponential Norm

$$
\rho(\mathbf{z}) = \sum_{m=1}^{M} -\exp(-z_m^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho$ | ró | 稀疏性度量 |
| $\mathbf{z}$ | zī | 编码向量 |
| $M$ | ēm | 向量维度 |

**一句话解释：**
指数范数使用指数衰减函数惩罚非零系数，促进稀疏性。

**简单应用场景：**
在非凸优化框架中实现更强的稀疏性约束。

---

### 公式 (9.16)：固定字典的稀疏编码优化 / Sparse Coding with Fixed Dictionary

$$
\min_{\mathbf{z}^{(n)}} \left\| \mathbf{x}^{(n)} - \mathbf{A} \mathbf{z}^{(n)} \right\|^2 + \eta \rho(\mathbf{z}^{(n)}), \quad \forall n \in [1, N]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{z}^{(n)}$ | zī | 第n个样本的编码向量 |
| $\mathbf{x}^{(n)}$ | xiāng | 第n个输入样本 |
| $\mathbf{A}$ | ā | 固定的字典矩阵 |
| $\eta$ | ē tǎ | 稀疏性权重 |

**一句话解释：**
在字典A固定的情况下，对每个样本单独优化其稀疏编码系数。

**简单应用场景：**
稀疏编码训练的第一步，固定字典并更新编码系数。

---

### 公式 (9.17)：固定编码的字典学习优化 / Dictionary Learning with Fixed Codes

$$
\min_{\mathbf{A}} \sum_{n=1}^{N} \left\| \mathbf{x}^{(n)} - \mathbf{A} \mathbf{z}^{(n)} \right\|^2 + \frac{1}{2} \lambda \| \mathbf{A} \|_F^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{A}$ | ā | 字典矩阵 |
| $\mathbf{z}^{(n)}$ | zī | 第n个样本的编码向量 |
| $\lambda$ | lā mǔ dá | 正则化系数 |
| $\| \cdot \|_F$ | fú | 弗罗贝尼乌斯范数 |

**一句话解释：**
在编码系数固定的情况下，优化字典矩阵并添加Frobenius范数正则化。

**简单应用场景：**
稀疏编码训练的第二步，固定编码并更新字典。

---

### 公式 (9.18)：自编码器重构误差 / Autoencoder Reconstruction Error

$$
\mathcal{L} = \sum_{n=1}^{N} \| \mathbf{x}^{(n)} - g(f(\mathbf{x}^{(n)})) \|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ēl | 损失函数 |
| $\mathbf{x}^{(n)}$ | xiāng | 第n个输入样本 |
| $f$ | ēf | 编码函数 |
| $g$ | jī | 解码函数 |
| $N$ | ēn | 样本数量 |

**一句话解释：**
自编码器的目标是通过编码器和解码器的组合使重构误差最小。

**简单应用场景：**
训练自编码器进行无监督特征学习和数据降维。

---

### 公式 (9.19)：自编码器重构误差（展开形式） / Autoencoder Loss (Expanded)

$$
\mathcal{L} = \sum_{n=1}^{N} \| \mathbf{x}^{(n)} - f \circ g(\mathbf{x}^{(n)}) \|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ēl | 损失函数 |
| $\circ$ | yuán | 函数复合 |
| $f$ | ēf | 编码器 |
| $g$ | jī | 解码器 |

**一句话解释：**
将自编码器的重构过程表示为编码器和解码器的函数复合。

**简单应用场景：**
在深层自编码器中构建多层编码-解码结构。

---

### 公式 (9.20)：自编码器编码层 / Autoencoder Encoding

$$
\mathbf{z} = f(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{z}$ | zī | 隐层表示 |
| $f$ | ēf | 激活函数 |
| $\mathbf{W}^{(1)}$ | dà wài | 编码权重矩阵 |
| $\mathbf{x}$ | xiāng | 输入 |
| $\mathbf{b}^{(1)}$ | bì | 编码偏置 |

**一句话解释：**
自编码器通过权重矩阵和激活函数将输入映射到隐层表示。

**简单应用场景：**
在自编码器中定义编码器部分的计算。

---

### 公式 (9.21)：自编码器解码层 / Autoencoder Decoding

$$
\mathbf{x}' = f(\mathbf{W}^{(2)} \mathbf{z} + \mathbf{b}^{(2)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{x}'$ | xiāng | 重构输出 |
| $f$ | ēf | 激活函数 |
| $\mathbf{W}^{(2)}$ | dà wài | 解码权重矩阵 |
| $\mathbf{z}$ | zī | 隐层表示 |
| $\mathbf{b}^{(2)}$ | bì | 解码偏置 |

**一句话解释：**
自编码器通过解码层将隐层表示映射回输入空间进行重构。

**简单应用场景：**
在自编码器中定义解码器部分的计算。

---

### 公式 (9.22)：带权重约束的自编码器损失函数 / Tied Weight Autoencoder Loss

$$
\mathcal{L} = \sum_{n=1}^{N} \| \mathbf{x}^{(n)} - \mathbf{x}'^{(n)} \|^2 + \lambda \| \mathbf{W} \|_F^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ēl | 损失函数 |
| $\mathbf{x}^{(n)}$ | xiāng | 第n个输入 |
| $\mathbf{x}'^{(n)}$ | xiāng | 第n个重构输出 |
| $\lambda$ | lā mǔ dá | 权重衰减系数 |
| $\mathbf{W}$ | dà wài | 权重矩阵 |

**一句话解释：**
自编码器的损失函数包含重构误差和权重正则化项，其中编码权重与解码权重共享（$\mathbf{W}^{(2)} = (\mathbf{W}^{(1)})^T$）。

**简单应用场景：**
使用权重共享减少自编码器的参数数量，防止过拟合。

---

### 公式 (9.23)：稀疏自编码器损失函数 / Sparse Autoencoder Loss

$$
\mathcal{L} = \sum_{n=1}^{N} \| \mathbf{x}^{(n)} - \mathbf{x}'^{(n)} \|^2 + \eta \rho(\mathbf{z}) + \lambda \| \mathbf{W} \|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ēl | 损失函数 |
| $\eta$ | ē tǎ | 稀疏性权重系数 |
| $\rho$ | ró | 稀疏性约束函数 |
| $\mathbf{z}$ | zī | 隐层表示 |
| $\lambda$ | lā mǔ dá | 权重衰减系数 |

**一句话解释：**
稀疏自编码器在基础自编码器损失中添加隐层稀疏性约束，促进学习稀疏特征。

**简单应用场景：**
在自编码器中学习稀疏的特征表示，用于特征提取和异常检测。

---

### 公式 (9.24)：稀疏约束的KL散度参数 / Sparsity Target Parameter

$$
\hat{\rho}_j = \frac{1}{N} \sum_{n=1}^{N} z_j^{(n)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{\rho}_j$ | ró | 隐层第j个神经元的平均激活值 |
| $N$ | ēn | 样本数量 |
| $z_j^{(n)}$ | zī | 第n个样本的第j个隐层单元激活值 |

**一句话解释：**
计算隐层每个神经元在所有训练样本上的平均激活值，用于稀疏性约束。

**简单应用场景：**
在稀疏自编码器中监测隐层神经元的平均激活率。

---

### 公式 (9.25)：KL散度稀疏性惩罚 / KL Divergence Sparsity Penalty

$$
\text{KL}(\rho^* \| \hat{\rho}_j) = \rho^* \log \frac{\rho^*}{\hat{\rho}_j} + (1 - \rho^*) \log \frac{1 - \rho^*}{1 - \hat{\rho}_j}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{KL}$ | kāi ēl | 库尔贝克-莱布勒散度 |
| $\rho^*$ | ró | 目标稀疏性参数 |
| $\hat{\rho}_j$ | ró | 实际平均激活值 |

**一句话解释：**
使用KL散度度量实际激活值与目标稀疏参数的差异，作为稀疏性惩罚项。

**简单应用场景：**
在稀疏自编码器中实现精细的稀疏性控制。

---

### 公式 (9.26)：稀疏自编码器完整损失函数 / Sparse Autoencoder Complete Loss

$$
\rho(\mathbf{Z}) = \sum_{j=1}^{k} \text{KL}(\rho^* \| \hat{\rho}_j)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho$ | ró | 稀疏性惩罚函数 |
| $\mathbf{Z}$ | zī | 隐层表示矩阵 |
| $k$ | kāi | 隐层单元数 |
| $\text{KL}$ | kāi ēl | 库尔贝克-莱布勒散度 |

**一句话解释：**
将所有隐层单元的KL散度相加得到总的稀疏性惩罚。

**简单应用场景：**
在稀疏自编码器的损失函数中统计所有隐层单元的稀疏性约束。

---


### 公式 (9.27)：稀疏自编码器 - 稀疏性约束 / Sparse Auto-Encoder - Sparsity Constraint

$$
\mathcal{L} = \sum_{n=1}^{N} \left\| \boldsymbol{x}^{(n)} - \boldsymbol{x}'^{(n)} \right\|^2 + \rho(\mathcal{Z}) + \lambda \left\| \boldsymbol{W} \right\|^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | ěl | 总的损失函数 |
| $\boldsymbol{x}^{(n)}$ | x dì n | 第 $n$ 个输入样本 |
| $\boldsymbol{x}'^{(n)}$ | x pī dì n | 第 $n$ 个样本的重构输出 |
| $\rho(\mathcal{Z})$ | ròu | 稀疏性约束项（惩罚函数） |
| $\mathcal{Z}$ | cā pī | 隐层编码向量的集合 |
| $\lambda$ | lǎm-dá | 正则化系数 |
| $\boldsymbol{W}$ | dá-bǎi-yóu | 权重矩阵 |

**一句话解释：**
稀疏自编码器通过重构损失、稀疏性约束和权重正则化三项的组合来学习稀疏的隐层表示。

**简单应用场景：**
用于图像特征学习，强制隐层只有少数神经元被激活，学到稀疏、有意义的特征表示。

---

### 公式 (9.28)：稀疏性约束的个数约束表示 / Sparsity Constraint - Cardinality Representation

$$
\rho(\boldsymbol{z}) = \sum_{m=1}^{M} \mathbf{1}(|z_m| > 0)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho(\boldsymbol{z})$ | ròu | 稀疏性约束函数（计数型） |
| $\boldsymbol{z}$ | z | 隐层编码向量 |
| $M$ | ěm | 隐层单元个数 |
| $z_m$ | z dì ěm | 隐层第 $m$ 个单元的激活值 |
| $\mathbf{1}(\cdot)$ | yī | 指示函数（真为1，假为0） |

**一句话解释：**
通过计数隐层中激活（非零）的神经元个数来定义稀疏性，实现离散的稀疏约束。

**简单应用场景：**
在需要明确控制激活神经元个数的场景，如 K-sparse autoencoder。

---

### 公式 (9.29)：稀疏性约束的 L1 范数表示 / Sparsity Constraint - L1 Norm Representation

$$
\rho(\boldsymbol{z}) = \sum_{m=1}^{M} |z_m|
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho(\boldsymbol{z})$ | ròu | 稀疏性约束函数（L1 范数） |
| $\boldsymbol{z}$ | z | 隐层编码向量 |
| $M$ | ěm | 隐层单元个数 |
| $z_m$ | z dì ěm | 隐层第 $m$ 个单元的激活值 |

**一句话解释：**
使用 L1 范数作为稀疏性约束，鼓励隐层激活向量中的元素尽可能为零。

**简单应用场景：**
在稀疏编码和压缩感知中广泛使用，作为促进稀疏性的正则化项。

---

### 公式 (9.30)：稀疏性约束的对数范数表示 / Sparsity Constraint - Log Norm Representation

$$
\rho(\boldsymbol{z}) = \sum_{m=1}^{M} \log(1 + z_m^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho(\boldsymbol{z})$ | ròu | 稀疏性约束函数（对数范数） |
| $\boldsymbol{z}$ | z | 隐层编码向量 |
| $M$ | ěm | 隐层单元个数 |
| $z_m$ | z dì ěm | 隐层第 $m$ 个单元的激活值 |

**一句话解释：**
通过对数函数的非线性性质，更温和地惩罚非零激活，在稀疏性和可微性之间取得平衡。

**简单应用场景：**
当需要平滑的稀疏约束且梯度计算相对容易时的选择。

---

### 公式 (9.31)：稀疏性约束的指数表示 / Sparsity Constraint - Exponential Representation

$$
\rho(\boldsymbol{z}) = \sum_{m=1}^{M} -\exp(-z_m^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\rho(\boldsymbol{z})$ | ròu | 稀疏性约束函数（指数型） |
| $\boldsymbol{z}$ | z | 隐层编码向量 |
| $M$ | ěm | 隐层单元个数 |
| $z_m$ | z dì ěm | 隐层第 $m$ 个单元的激活值 |

**一句话解释：**
使用指数函数强烈鼓励激活值接近零，当激活值远离零时，惩罚项快速增长。

**简单应用场景：**
需要强制隐层高度稀疏，使大多数神经元处于不激活状态的场景。

---

## 说明

- **第7章 (7.32-7.34)**：这些是章节最后的优化算法小结公式，用于统一表示不同优化算法的框架。
- **第7章 (7.45及之后)**：无附加公式。第7.5节优化方法小结后，后续章节为参数初始化内容，不包含编号公式(7.45)及以后的内容。
- **第9章 (9.27-9.31)**：这些公式展示了稀疏自编码器中不同类型的稀疏性约束函数的定义方式。



## Formulas 10.1 - 10.23 from Pages 228-232

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

### 公式 (10.9)：集成学习的误差界 / Ensemble Learning Error Bound

$$
\mathcal{R}(J) \geq \mathcal{R}(F) \geq \frac{1}{M} \mathcal{R}(J)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}(J)$ | jī chǔ mó xíng de píng jun1 cuò wù | Average base model error |
| $\mathcal{R}(F)$ | jí chéng mó xíng de cuò wù | Ensemble error |

**一句话解释：**
The ensemble error is bounded above by the average base model error and below by one M-th of it.

**简单应用场景：**
Proving the effectiveness of ensemble methods in reducing error.

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

### 公式 (10.15)：简化形式的损失 / Simplified Loss Form

$$
\mathcal{L}(\alpha_m, f_m(x)) = \sum_{n=1}^{N} w_n^{(n)} \exp\left(-\alpha_m y^{(n)} f_m(x^{(n)})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_n^{(n)}$ | dì $n$ ge yàng běn de quán zhòng | Weight of sample $n$ |
| $\exp(-y^{(n)} F_{m-1}(x^{(n)}))$ | qián $m-1$ cì zuò yòng hou de quán zhòng | Reweighting factor |

**一句话解释：**
The loss can be rewritten with sample weights that depend on previous iterations.

**简单应用场景：**
Focusing AdaBoost on misclassified examples by reweighting samples.

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

### 公式 (10.17)：修正的损失函数 / Modified Loss Function

$$
\mathcal{L}(\alpha_m, f_m(x)) = \sum_{n=1}^{N} w_n^{(n)}\left(1 - \alpha_m y^{(n)} f_m(x^{(n)}) + \frac{1}{2}\alpha_m^2\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_n^{(n)}$ | dì $n$ ge yàng běn de quán zhòng | Sample weight |
| $\alpha_m$ | dì $m$ cì de quán zhòng xì shu | Weight coefficient |
| $y^{(n)} f_m(x^{(n)})$ | yù cè de zhèng que xìng | Prediction correctness |

**一句话解释：**
A quadratic approximation of the exponential loss for optimization.

**简单应用场景：**
Deriving the update rule for weights and classifier weights in AdaBoost.

---

### 公式 (10.18)：损失函数的近似 / Loss Function Approximation

$$
\propto \alpha_m \sum_{n=1}^{N} w_n^{(n)}\mathcal{I}\left(y^{(n)} \neq f_m(x^{(n)})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{I}(y^{(n)} \neq f_m(x^{(n)}))$ | fēn lèi cuò wù zhǐ shì | Misclassification indicator |
| $w_n^{(n)}$ | dì $n$ ge yàng běn de quán zhòng | Sample weight |

**一句话解释：**
The loss is proportional to the weighted misclassification error.

**简单应用场景：**
Showing that AdaBoost minimizes weighted classification error.

---

### 公式 (10.19)：加权误分类率的表示 / Weighted Misclassification Rate

$$
\mathcal{L}(\alpha_m, f_m(x)) = \sum_{y^{(n)}=f_m(x^{(n)})} w_n^{(n)} \exp(-\alpha_m) + \sum_{y^{(n)} \neq f_m(x^{(n)})} w_n^{(n)} \exp(\alpha_m)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w_n^{(n)}$ | dì $n$ ge yàng běn de quán zhòng | Sample weight |
| $\exp(-\alpha_m)$ | zhèng que fēn lèi de sǔn shī | Loss for correct classification |
| $\exp(\alpha_m)$ | cuò wù fēn lèi de sǔn shī | Loss for misclassification |

**一句话解释：**
Separating the loss into correct and incorrect classification terms.

**简单应用场景：**
Computing the total weighted loss in AdaBoost.

---

### 公式 (10.20)：损失的再次近似 / Loss Approximation Again

$$
\propto (1 - \epsilon_m) \exp(-\alpha_m) + \epsilon_m \exp(\alpha_m)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\epsilon_m$ | dì $m$ cì fēn lèi qì de cuò wù lǜ | Weighted error rate of classifier $m$ |
| $1 - \epsilon_m$ | zhèng que lǜ | Accuracy rate |

**一句话解释：**
The weighted loss depends on the weighted error rate of the classifier.

**简单应用场景：**
Deriving optimal weight assignment based on classifier accuracy.

---

### 公式 (10.21)：加权误分类率的定义 / Definition of Weighted Error Rate

$$
\epsilon_m = \frac{\sum_{y^{(n)} \neq f_m(x^{(n)})} w_n^{(n)}}{\sum_n w_n^{(n)}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\epsilon_m$ | jiā quán cuò wù lǜ | Weighted error rate |
| $w_n^{(n)}$ | dì $n$ ge yàng běn de quán zhòng | Sample weight |
| $y^{(n)} \neq f_m(x^{(n)})$ | cuò wù fēn lèi | Misclassified samples |

**一句话解释：**
The weighted error rate is the sum of weights for misclassified samples divided by total weights.

**简单应用场景：**
Computing the error rate for the current base classifier in AdaBoost.

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

### 公式 (10.23)：协同训练中两个视图的约束 / Co-training with Two Views Constraint

$$
\exists f_i, f_j, \quad \forall x \in \mathcal{X}, \quad f_i(x) = f_j(x) = g(x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f_i$ | dì $i$ ge shì tú de fēn lèi qì | Classifier for view $i$ |
| $f_j$ | dì $j$ ge shì tú de fēn lèi qì | Classifier for view $j$ |
| $g(x)$ | zhēn shí biāo qiān hánshù | True labeling function |
| $\mathcal{X}$ | shū jù kōng jiān | Data space |

**一句话解释：**
In co-training, two views (classifiers) should both learn the true concept that explains all examples.

**简单应用场景：**
Training multiple models on different feature subsets in semi-supervised learning.

---

**End of Chapter 10 Formulas (10.1 - 10.23)**

## 第10章：多任务学习、迁移学习与元学习

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

### 公式 (10.26)：域不变性目标 / Domain-Invariant Objective

$$
\mathcal{R}_c(\boldsymbol{\theta}_f) = \mathbb{E}_{(x,y) \sim p_S(x,y)} \mathcal{L}((x,\boldsymbol{\theta}_f), y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_c(\boldsymbol{\theta}_f)$ | ěr cè shàng xiā tà ef | 分类器的风险函数 |
| $\mathbb{E}$ | yī | 期望值 |
| $p_S(x,y)$ | pì èm | 源域的分布 |
| $\mathcal{L}$ | ěl | 损失函数 |

**一句话解释：**
在源域上定义的分类损失函数，计算模型在源域数据上的平均误差。

**简单应用场景：**
在迁移学习中，源域模型的训练损失用作目标域适应的基准。

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

### 公式 (10.29)：带有特征映射的域自适应目标 / Domain Adaptation with Feature Mapping

$$
\mathcal{R}_{\mathcal{T}}(\boldsymbol{\theta}_f, \boldsymbol{\theta}_g) = \mathbb{E}_{(x,y) \sim p_S(x, \boldsymbol{\theta}_f), y)} + \gamma d_g(S, T)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_{\mathcal{T}}$ | ěr tì | 目标域的风险函数 |
| $\boldsymbol{\theta}_g$ | xī tǎ jī | 特征映射参数 |
| $d_g(S, T)$ | dì jī | 源域和目标域之间的距离度量 |
| $\gamma$ | gā mǎ | 权衡参数 |

**一句话解释：**
通过特征映射将源域和目标域特征对齐，同时最小化任务损失和特征差异。

**简单应用场景：**
在迁移学习中通过特征空间对齐来实现从源域到目标域的知识迁移。

---

### 公式 (10.30)：简化的风险分解 / Simplified Risk Decomposition

$$
\mathcal{R}_S(\boldsymbol{\theta}_f, \boldsymbol{\theta}_g) = \mathcal{R}_S(\boldsymbol{\theta}_f, \boldsymbol{\theta}_g) + \gamma d_g(S, T)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{R}_S$ | ěr ès | 源域的风险函数 |

**一句话解释：**
源域风险与特征距离的加权组合，用于优化域自适应模型。

**简单应用场景：**
通过平衡源域性能和域间距离来优化迁移学习模型。

---

### 公式 (10.31)：源域和目标域数据集定义 / Source and Target Domain Data Sets

$$
\mathcal{D}_S = \{(x_i^{(s)}, y_i^{(s)})\}_{i=1}^N \sim p_S(x, y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_S$ | dé ěl ès | 源域数据集 |
| $x_i^{(s)}$ | x shàng biāo ès | 源域第i个样本 |
| $y_i^{(s)}$ | y shàng biāo ès | 源域第i个样本标签 |
| $p_S(x, y)$ | pì ès | 源域分布 |

**一句话解释：**
定义源域数据集，包含N个有标签的样本对，来自源域分布。

**简单应用场景：**
迁移学习中用于初始模型训练的有标签源域数据。

---

### 公式 (10.32)：目标域无标签数据集 / Target Domain Unlabeled Data Set

$$
\mathcal{D}_T = \{x_i^{(u)}\}_{i=1}^M \sim p_T(x, y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}_T$ | dé ěl tì | 目标域数据集 |
| $x_i^{(u)}$ | x shàng biāo u | 目标域第i个无标签样本 |
| $M$ | dà xiě èm | 目标域样本总数 |

**一句话解释：**
定义目标域的无标签数据集，包含M个无标签样本。

**简单应用场景：**
迁移学习中目标域的无标签数据用于域自适应。

---

### 公式 (10.33)：源域标签分离的损失函数 / Source Domain Label Separation Loss

$$
\mathcal{L}_c(\boldsymbol{\theta}_c, \boldsymbol{\theta}_c) = \frac{1}{N} \sum_{i=1}^{N} \log(h_i^{(n)}, \boldsymbol{\theta}_c) + \frac{1}{M} \sum_{n=1}^{M} \log(1 - c(h_n^{(u)}, \boldsymbol{\theta}_c))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_i^{(n)}$ | hǎi shàng biāo n | 第i个有标签样本的特征 |
| $h_n^{(u)}$ | hǎi shàng biāo u | 第n个无标签样本的特征 |
| $c$ | xī | 二分类器 |

**一句话解释：**
分别对有标签和无标签样本进行分类，用于源域标签分离。

**简单应用场景：**
在半监督学习中区分有标签和无标签数据。

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

### 公式 (10.35)：完整的损失函数优化 / Complete Loss Function Optimization

$$
\min_{\boldsymbol{\theta}_f,\boldsymbol{\theta}_c} \sum_{i=1}^{N} \mathcal{L}\left(\left(f(x_i^{(n)}; \boldsymbol{\theta}_f), y_i^{(n)}\right) - \gamma\mathcal{L}_c(\boldsymbol{\theta}_f, \boldsymbol{\theta}_c)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $f(x_i^{(n)}; \boldsymbol{\theta}_f)$ | fā | 特征提取函数 |
| $\gamma$ | gā mǎ | 权衡参数 |

**一句话解释：**
联合优化特征提取器和分类器，平衡监督学习和无监督学习的贡献。

**简单应用场景：**
半监督或迁移学习中的端到端模型优化。

---

### 公式 (10.36)：终身学习中的数据分布分解 / Data Distribution Decomposition in Lifelong Learning

$$
\log p(\boldsymbol{\theta}|\mathcal{D}) = \log p(\boldsymbol{\theta}|\boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) - \log p(\mathcal{D})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{\theta}\|\mathcal{D})$ | pì | 后验分布 |
| $p(\boldsymbol{\theta})$ | pì | 先验分布 |
| $p(\mathcal{D})$ | pì dé ěl | 数据的边际分布 |

**一句话解释：**
贝叶斯推断中后验分布可以分解为似然、先验和证据三部分。

**简单应用场景：**
终身学习中的贝叶斯框架下的参数学习。

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

### 公式 (10.38)：进一步简化 / Further Simplification

$$
= \log p(\mathcal{D}_B|\boldsymbol{\theta}) + \log p(\boldsymbol{\theta}|\mathcal{D}_A) - \log p(\mathcal{D}_B)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{\theta}\|\mathcal{D}_A)$ | pì | 在A任务后的参数分布 |

**一句话解释：**
展示了如何将联合分布进一步简化，使用任务A的后验作为任务B的先验。

**简单应用场景：**
顺序多任务学习中的参数更新。

---

### 公式 (10.39)：Fisher信息矩阵的逆 / Inverse of Fisher Information Matrix

$$
p(\boldsymbol{\theta}|\mathcal{D}_A) = \mathcal{N}(\mathcal{F}_A, F^{-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{N}$ | dà xiě èn | 正态分布 |
| $\mathcal{F}_A$ | dà xiě èf | Fisher信息矩阵 |
| $F^{-1}$ | èf yì cì | Fisher信息矩阵的逆 |

**一句话解释：**
参数的后验分布可以用正态分布近似，其协方差矩阵是Fisher信息矩阵的逆。

**简单应用场景：**
贝叶斯参数估计中用于量化参数的不确定性。

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

### 公式 (10.49)：Fisher信息关于子集的估计 / Fisher Information Estimation on Subset

$$
F^A(\boldsymbol{\theta}) = \frac{1}{\sum_{(x,y) \in \mathcal{D}_A}} \nabla_{\boldsymbol{\theta}} \log p(y|x;\boldsymbol{\theta})(\nabla_{\boldsymbol{\theta}} \log p(y|x;\boldsymbol{\theta}))^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $F^A(\boldsymbol{\theta})$ | èf ā | 任务A的Fisher信息矩阵 |
| $\mathcal{D}_A$ | dé ěl ā | 任务A的数据集 |

**一句话解释：**
在特定任务或数据集上估计Fisher信息矩阵，用于任务特定的参数学习。

**简单应用场景：**
多任务学习中评估各个任务对参数的约束。

---

### 公式 (10.50)：带有Fisher信息惩罚的损失函数 / Loss Function with Fisher Information Penalty

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}_B(\boldsymbol{\theta}) + \sum_{i=1}^{N} \frac{1}{2}\lambda^A_i \cdot (\boldsymbol{\theta}_i - \boldsymbol{\theta}^*_A)^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\lambda^A_i$ | lā mbda ā | 任务A关于第i个参数的Fisher信息 |
| $\boldsymbol{\theta}^*_A$ | xī tǎ ā xīng | 任务A学习后的参数 |

**一句话解释：**
终身学习中的重要权重遗忘（Elastic Weight Consolidation），通过Fisher信息加权来防止遗忘旧任务。

**简单应用场景：**
连续学习多个任务时保持模型对之前任务的性能。

---

### 公式 (10.51)：基于优化器的元学习 / Optimizer-Based Meta-Learning

$$
\boldsymbol{\theta}_{i+1} = \boldsymbol{\theta}_i - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_{i-1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | ālf fá | 学习率 |
| $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ | nā bǐ lā | 损失函数的梯度 |

**一句话解释：**
标准的梯度下降法，表示模型参数沿着损失函数梯度的反方向更新。

**简单应用场景：**
元学习中学习如何优化模型参数。

---

### 公式 (10.52)：优化器加速的元学习 / Meta-Learning with Optimizer Acceleration

$$
\boldsymbol{\theta}_{i+1} = \boldsymbol{\theta}_i + g_i(\nabla \mathcal{L}(\boldsymbol{\theta}_i); \boldsymbol{\phi})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g_i$ | jī | 第i步的优化器函数 |
| $\boldsymbol{\phi}$ | fài | 优化器函数的参数 |

**一句话解释：**
学习一个优化器函数来根据梯度信息更新参数，使得优化过程更高效。

**简单应用场景：**
学习学习（Learning to learn）中的参数优化策略。

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

### 公式 (10.56)：模型无关元学习中的快速适应 / Fast Adaptation in Model-Agnostic Meta-Learning

$$
\boldsymbol{\theta}^m = \boldsymbol{\theta} - \alpha \nabla \mathcal{L}_{m}(f_\boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}^m$ | xī tǎ em | 任务m快速适应后的参数 |
| $\mathcal{L}_{m}$ | ěl em | 任务m的损失函数 |
| $f_\boldsymbol{\theta}$ | èf | 参数化的模型函数 |

**一句话解释：**
通过单次梯度步更新使模型快速适应新任务，这是MAML算法的核心。

**简单应用场景：**
少样本学习中快速学习新分类或回归任务。

---

### 公式 (10.57)：模型无关元学习的目标函数 / MAML Objective Function

$$
\min_{\boldsymbol{\theta}} \sum_{m \sim p(\mathcal{T})} \mathcal{L}_{m}\left(f(\boldsymbol{\theta} - \alpha \nabla \mathcal{L}_m(f_\boldsymbol{\theta}))\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\mathcal{T})$ | pì | 任务分布 |

**一句话解释：**
MAML的元学习目标是找到初始参数，使得在快速适应任意新任务后损失最小。

**简单应用场景：**
实现Few-shot learning，使模型能用少量样本快速学习新任务。

---

### 公式 (10.58)：元优化的梯度更新 / Meta-Optimization Gradient Update

$$
\boldsymbol{\theta} = \boldsymbol{\theta} - \beta \nabla_{\boldsymbol{\theta}} \sum_{m=1}^{M} \mathcal{L}_{m}(f_{\boldsymbol{\theta}^m})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\beta$ | bè tá | 元学习率 |
| $\nabla_{\boldsymbol{\theta}}$ | nā bǐ lā | 关于初始参数的梯度 |

**一句话解释：**
在元学习中，通过梯度下降更新初始参数，使其成为更好的初始化点。

**简单应用场景：**
MAML算法的外层优化步骤。

---

### 公式 (10.59)：展开的元优化 / Unrolled Meta-Optimization

$$
= \boldsymbol{\theta} - \beta \sum_{m=1}^{M} \nabla_{\boldsymbol{\theta}} \mathcal{L}_{m}(f_{\boldsymbol{\theta}^m})\left(I - \alpha \nabla^2_{\boldsymbol{\theta}} \mathcal{L}_m(f_{\boldsymbol{\theta}^m})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I$ | dà xiě ài | 单位矩阵 |
| $\nabla^2$ | nā bǐ lā píng fāng | Hessian矩阵（二阶导数） |

**一句话解释：**
展开元优化的计算过程，显示初始参数梯度如何通过任务特定参数的更新而受到影响。

**简单应用场景：**
MAML中的反向传播计算。

---

## 章节总结

第10章系统地介绍了多任务学习、迁移学习和元学习三个重要的机器学习范式。主要内容包括：

1. **多任务学习（10.24-10.25）**：通过共享表示和加权损失实现多个相关任务的联合学习。

2. **迁移学习（10.26-10.35）**：从源域知识转移到目标域，涵盖域不变性、重要性加权和特征对齐等方法。

3. **终身学习（10.36-10.50）**：处理顺序任务学习中的灾难性遗忘问题，引入Fisher信息矩阵进行参数约束。

4. **元学习（10.51-10.59）**：学习学习算法本身，包括基于优化器的方法（LSTM）和模型无关的元学习（MAML）。

这些学习范式都旨在提高模型的泛化能力和学习效率。


## 11.1 模型表示

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

### 公式 (11.3)：条件独立性1 / Conditional Independence 1

$$
p(x_2|x_1, x_3) = p(x_2|x_1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_2\|x_1, x_3)$ | tiáo jiàn gài lǜ | 给定$x_1$和$x_3$下$x_2$的条件概率 |
| $x_1$ | biàn liàng | 已知变量1 |
| $x_3$ | biàn liàng | 已知变量2 |

**一句话解释：**
在已知$x_1$时，$x_2$独立于$x_3$，即$x_3$对$x_2$没有影响。

**简单应用场景：**
简化贝叶斯网络中的概率计算，减少需要考虑的变量数。

---

### 公式 (11.4)：条件独立性2 / Conditional Independence 2

$$
p(x_3|x_1, x_2) = p(x_3|x_1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_3\|x_1, x_2)$ | tiáo jiàn gài lǜ | 给定$x_1$和$x_2$下$x_3$的条件概率 |
| $x_1$ | biàn liàng | 已知变量 |
| $x_2$ | biàn liàng | 冗余变量 |

**一句话解释：**
在已知$x_1$时，$x_3$独立于$x_2$，这表示图中的条件独立结构。

**简单应用场景：**
在有向无环图(DAG)中表示父节点与其他变量的独立关系。

---

### 公式 (11.5)：条件独立性3 / Conditional Independence 3

$$
p(x_4|x_1, x_2, x_3) = p(x_4|x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_4\|x_1, x_2, x_3)$ | tiáo jiàn gài lǜ | 多变量条件下的条件概率 |
| $x_4$ | biàn liàng | 目标变量 |
| $x_1, x_2, x_3$ | yǐ zhī biàn liàng | 条件变量 |

**一句话解释：**
在已知$x_2$和$x_3$时，$x_4$独立于$x_1$，显示了图形结构中的独立性。

**简单应用场景：**
证明在概率图中某些变量之间的条件独立关系。

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
| $p(x_i\|...)$ | tiáo jiàn gài lǜ | 条件概率因子 |

**一句话解释：**
联合概率分解为一系列条件概率的乘积，反映了链式结构。

**简单应用场景：**
在链式贝叶斯网络中计算完整的概率分布。

---

### 公式 (11.7)：联合概率的因子分解2 / Factored Joint Probability 2

$$
p(\boldsymbol{x}) = p(x_1)p(x_2|x_1)p(x_3|x_1)p(x_4|x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率分布 |
| $p(x_i\|...)$ | tiáo jiàn gài lǜ | 条件概率 |

**一句话解释：**
利用条件独立性简化因子分解，$x_2$和$x_3$都仅依赖$x_1$。

**简单应用场景：**
表示具有不同结构的贝叶斯网络中的概率分解。

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

### 公式 (11.9)：条件独立与特后变量 / Conditional Independence with Posterior

$$
x_k \perp x_{\pi_k^c}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_k$ | jié diǎn | 当前节点变量 |
| $\perp$ | dú lì | 独立于 |
| $x_{\pi_k^c}$ | qī tā biàn liàng | 非父节点的其他变量 |

**一句话解释：**
在有向图中，一个节点与非其父节点的所有节点条件独立，当给定其父节点时。

**简单应用场景：**
确定贝叶斯网络中的马尔可夫毯(Markov Blanket)。

---

## 11.1.2 常见的有向图模型

### 公式 (11.10)：Sigmoid信念网络 / Sigmoid Belief Network

$$
p(x_k = 1|x_{\neg k}, \theta) = \sigma\left(\theta_0 + \sum_{\ell \neq k} \theta_\ell x_\ell\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_k=1\|...)$ | tiáo jiàn gài lǜ | 变量$x_k$取1的条件概率 |
| $\sigma(\cdot)$ | shì tǐ hánshù | Sigmoid逻辑函数 |
| $\theta_0$ | piàn chì | 偏置参数 |
| $\theta_\ell$ | quán zhòng | 权重参数 |
| $x_\ell$ | biàn liàng | 其他变量 |

**一句话解释：**
使用Sigmoid函数将其他变量的加权和映射到[0,1]作为二值变量的概率。

**简单应用场景：**
在Sigmoid信念网络中定义二值变量的条件概率分布。

---

### 公式 (11.11)：朴素贝叶斯分类1 / Naive Bayes Classification 1

$$
p(y|\boldsymbol{x};\theta) = \frac{p(\boldsymbol{x}, y;\theta)}{p(\boldsymbol{x};\theta)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y\|\boldsymbol{x};\theta)$ | tiáo jiàn gài lǜ | 给定特征下标签的条件概率 |
| $\boldsymbol{x}$ | tè zhēng xiàng liàng | 特征向量 |
| $y$ | biāo qiān | 类别标签 |
| $\theta$ | cān shù | 模型参数 |

**一句话解释：**
使用贝叶斯公式从联合概率推导条件概率，用于分类任务。

**简单应用场景：**
在朴素贝叶斯分类中计算后验概率以进行类别预测。

---

### 公式 (11.12)：朴素贝叶斯分类2 / Naive Bayes Classification 2

$$
p(y|\boldsymbol{x};\theta) \propto p(\boldsymbol{x}, y;\theta; \theta)p(y; \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\propto$ | chéng zhèng bǐ lì | 成正比例于 |
| $p(\boldsymbol{x}\|y;\theta)$ | lián hé tiáo jiàn gài lǜ | 给定标签的特征条件概率 |
| $p(y;\theta)$ | xiān yàn gài lǜ | 标签的先验概率 |

**一句话解释：**
后验概率与似然度与先验的乘积成正比，省略分母进行计算。

**简单应用场景：**
简化朴素贝叶斯分类的计算，通过相对概率进行类别判断。

---

### 公式 (11.13)：隐马尔可夫模型的联合概率 / HMM Joint Probability

$$
p(\boldsymbol{x}, \boldsymbol{y}; \theta) = \prod_{t=1}^{T} p(y_t|y_{t-1}, \theta_t)p(x_t|y_t, \theta_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | kě jiàn biàn liàng | 可见变量序列 |
| $\boldsymbol{y}$ | yǐn biàn liàng | 隐变量序列 |
| $p(y_t\|y_{t-1}, \theta_t)$ | zhuǎn yí gài lǜ | 转移概率 |
| $p(x_t\|y_t, \theta_t)$ | fā shè gài lǜ | 发射概率 |
| $T$ | shí jiān bù shù | 时间步数 |

**一句话解释：**
隐马尔可夫模型的联合概率分解为转移概率与发射概率的乘积。

**简单应用场景：**
在序列标注、语音识别中建模观察序列与隐状态序列的关系。

---

### 公式 (11.15)：无向图模型的联合概率 / Undirected Graphical Model

$$
p(x_k|x_{\neg k}) = p(x_k|\mathcal{X}_{V(k)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_k$ | jié diǎn | 当前节点变量 |
| $x_{\neg k}$ | qī tā jiē diǎn | 所有其他节点变量 |
| $\mathcal{X}_{V(k)}$ | fù bó jié diǎn | 马尔可夫毯内的节点 |

**一句话解释：**
在无向图中，节点的条件分布仅依赖其邻域节点(马尔可夫毯)。

**简单应用场景：**
确定无向图中的条件独立结构与局部概率计算。

---

## 11.1.4 无向图模型的概率分解

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

### 公式 (11.21)：对数线性模型的势函数 / Log-Linear Model Potential

$$
\phi_c(\boldsymbol{x}_c|\theta_c) = \exp\left(\theta_c f_c(\boldsymbol{x}_c)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\phi_c$ | wèi shì hánshù | 势函数 |
| $\theta_c$ | cān shù | 势函数参数 |
| $f_c(\boldsymbol{x}_c)$ | tè zhēng hánshù | 特征函数 |

**一句话解释：**
对数线性模型中，势函数是参数与特征函数乘积的指数形式。

**简单应用场景：**
在条件随机场和Boltzmann机中定义参数化的势函数。

---

### 公式 (11.22)：对数线性模型的条件概率 / Log-Linear Model Conditional Probability

$$
\log p(x; \theta) = \sum_{c \in C} \theta_c f_c(\boldsymbol{x}_c) - \log Z(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x; \theta)$ | duì shù gài lǜ | 概率的对数 |
| $\theta_c$ | cān shù | 模型参数 |
| $f_c(\boldsymbol{x}_c)$ | tè zhēng | 特征函数值 |
| $Z(\theta)$ | guī huà hánshù | 分割函数 |

**一句话解释：**
对数线性模型的对数概率是特征函数的加权和减去分割函数的对数。

**简单应用场景：**
在参数化势函数模型中计算概率的对数形式。

---

### 公式 (11.23)：最大熵模型1 / Maximum Entropy Model 1

$$
p(y|x; \theta) = \frac{1}{Z(x;\theta)} \exp\left(\theta^T f(x, y)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y\|x; \theta)$ | tiáo jiàn gài lǜ | 条件概率 |
| $Z(x;\theta)$ | guī huà hánshù | 分割函数 |
| $\theta$ | cān shù xiàng liàng | 参数向量 |
| $f(x, y)$ | tè zhēng xiàng liàng | 特征向量 |

**一句话解释：**
最大熵模型通过特征函数的加权和来表示条件概率分布。

**简单应用场景：**
在自然语言处理的标注任务中进行条件概率建模。

---

### 公式 (11.24)：条件随机场 / Conditional Random Field

$$
p(y|\boldsymbol{x}; \theta) = \frac{1}{Z(x;\theta)} \exp\left(\sum_{c \in C} \theta_c f_c(x, y_c)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y\|\boldsymbol{x}; \theta)$ | tiáo jiàn gài lǜ | 给定输入的输出条件概率 |
| $Z(x;\theta)$ | guī huà hánshù | 分割函数 |
| $\theta_c$ | cān shù | 团特征的权重 |
| $f_c(x, y_c)$ | tè zhēng hánshù | 团的特征函数 |

**一句话解释：**
条件随机场对输出序列建立条件概率分布，使用全局归一化的势函数。

**简单应用场景：**
在序列标注、实体识别等任务中建模输入输出的条件关系。

---

### 公式 (11.25)：线性链条件随机场 / Linear-Chain CRF

$$
p(y|\boldsymbol{x}; \theta) = \frac{1}{Z(x;\theta)} \exp\left(\sum_{t=1}^{T} \theta_t f_t(x, y_t) + \sum_{t=1}^{T-1} \theta_t f_t(x, y_t, y_{t+1})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(y\|\boldsymbol{x}; \theta)$ | tiáo jiàn gài lǜ | 条件概率分布 |
| $f_t(x, y_t)$ | jiē diǎn tè zhēng | 节点特征函数 |
| $f_t(x, y_t, y_{t+1})$ | zhuǎn yí tè zhēng | 转移特征函数 |
| $T$ | shí jiān bù shù | 序列长度 |

**一句话解释：**
线性链CRF对序列进行标注，利用节点特征和转移特征建模。

**简单应用场景：**
在词性标注、命名实体识别等序列标注任务中应用。

---

## 11.2 学习

### 公式 (11.26)：有向图与无向图的转换 / Directed-Undirected Transformation

$$
p(\boldsymbol{x}) = p(x_1)p(x_2|x_1)p(x_3|x_1)p(x_4|x_1, x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x})$ | lián hé gài lǜ | 联合概率 |
| $p(x_i\|...)$ | tiáo jiàn gài lǜ | 条件概率 |

**一句话解释：**
有向图的因子分解可以转换为无向图的势函数表示。

**简单应用场景：**
在不同模型表示之间进行转换与对比分析。

---

## 11.2.1 不含隐变量的参数估计

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

### 公式 (11.29)：参数最大似然估计 / Maximum Likelihood Estimation

$$
\hat{\theta}_k = \arg \max_{\theta_k} \sum_{n=1}^{N} \log p(x_k^{(n)}|x_{\pi_k}^{(n)}; \theta_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{\theta}_k$ | zuì yōu cān shù | 参数k的最大似然估计 |
| $\arg \max$ | zuì dà huà | 使函数最大的参数值 |
| $\log p$ | duì shù gài lǜ | 对数概率 |

**一句话解释：**
通过最大化条件对数似然来估计每个节点的参数。

**简单应用场景：**
在贝叶斯网络学习中分别为各节点参数进行MLE。

---

## 11.2.2 含隐变量的参数估计(EM算法)

### 公式 (11.30)：无向图模型的对数似然 / Undirected Model Log-Likelihood

$$
p(x; \theta) = \frac{1}{Z(\theta)} \exp\left(\sum_{c \in C} \theta_c f_c(\boldsymbol{x}_c)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x; \theta)$ | tiáo jiàn gài lǜ | 概率分布 |
| $Z(\theta)$ | guī huà hánshù | 分割函数 |
| $\theta_c$ | cān shù | 团的参数 |
| $f_c(\boldsymbol{x}_c)$ | tè zhēng hánshù | 特征函数 |

**一句话解释：**
无向图使用指数族形式的概率分布，其中分割函数依赖所有参数。

**简单应用场景：**
在条件随机场、Boltzmann机等无向模型中定义参数化分布。

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

### 公式 (11.32)：对数似然的分解 / Log-Likelihood Decomposition

$$
\mathcal{L}(\mathcal{D}; \theta) = \frac{1}{N} \sum_{n=1}^{N} \left(\sum_{c \in C} \theta_c f_c(\boldsymbol{x}_c^{(n)})\right) - \log Z(\theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}(\mathcal{D}; \theta)$ | duì shù sì ránhuàn xìng | 对数似然 |
| $f_c(\boldsymbol{x}_c^{(n)})$ | yàng běn n de tè zhēng | 第n个样本的特征 |
| $Z(\theta)$ | guī huà hánshù | 分割函数 |

**一句话解释：**
对数似然分解为经验特征期望与分割函数的差，便于计算梯度。

**简单应用场景：**
在条件随机场中进行参数优化的梯度计算。

---

### 公式 (11.33)：对数似然的梯度 / Log-Likelihood Gradient

$$
\frac{\partial \mathcal{L}(\mathcal{D}; \theta)}{\partial \theta_c} = \frac{1}{N} \sum_{n=1}^{N} \left(f_c(\boldsymbol{x}^{(n)})\right) - \frac{\partial \log Z(\theta)}{\partial \theta_c}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial \theta_c}$ | piàn dǎo shù | 对参数$\theta_c$的偏导数 |
| $f_c(\boldsymbol{x}^{(n)})$ | yàng běn tè zhēng | 样本特征值 |
| $\log Z(\theta)$ | fēn gē hánshù de duì shù | 分割函数的对数 |

**一句话解释：**
梯度等于经验特征期望减去模型特征期望，使用期望差来优化参数。

**简单应用场景：**
使用梯度下降法优化无向图模型的参数。

---

### 公式 (11.34)：分割函数对参数的偏导 / Partition Function Gradient

$$
\frac{\partial \log Z(\theta)}{\partial \theta_c} = \sum_{z} \frac{1}{Z(\theta)} \exp\left(\sum_c \theta_c f_c(\boldsymbol{x}_c)\right) \cdot f_c(\boldsymbol{x}_c)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \log Z}{\partial \theta_c}$ | guī huà hánshù jù duì | 对分割函数的导数 |
| $\sum_{z}$ | duì suǒ yǒu zhuàng tài qiú hé | 对所有状态求和 |
| $f_c(\boldsymbol{x}_c)$ | tè zhēng hánshù | 特征函数值 |

**一句话解释：**
分割函数的梯度等于模型分布下特征函数的期望值。

**简单应用场景：**
计算条件随机场等模型的梯度时获得模型期望项。

---

### 公式 (11.35)：期望形式的梯度 / Gradient in Expectation Form

$$
\frac{\partial \log Z(\theta)}{\partial \theta_c} = \mathbb{E}_{x \sim p(x;\theta)} \left[f_c(\boldsymbol{x}_c)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{E}$ | qī wàng zhí | 期望算子 |
| $x \sim p(x;\theta)$ | cóng gài lǜ p cái yàng | 从分布p采样 |
| $f_c(\boldsymbol{x}_c)$ | tè zhēng hánshù | 特征函数 |

**一句话解释：**
分割函数的梯度可表示为模型分布下特征函数的期望。

**简单应用场景：**
在推导参数学习算法时使用期望形式的表达。

---

### 公式 (11.36)：参数梯度的简化形式 / Simplified Parameter Gradient

$$
\frac{\partial \mathcal{L}(\mathcal{D}; \theta)}{\partial \theta_c} = \frac{1}{N} \sum_{n=1}^{N} \left[f_c(\boldsymbol{x}^{(n)}) - \mathbb{E}_{x \sim p(x;\theta)} \left[f_c(\boldsymbol{x}_c)\right]\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial \mathcal{L}}{\partial \theta_c}$ | piàn dǎo shù | 梯度 |
| $\mathbb{E}$ | qī wàng zhí | 期望 |
| $f_c$ | tè zhēng hánshù | 特征函数 |

**一句话解释：**
参数梯度等于经验特征期望减去模型特征期望，是特征期望的差异。

**简单应用场景：**
在梯度上升法中使用此形式进行参数优化。

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

### 公式 (11.41)：变分下界(ELBO) / Evidence Lower Bound

$$
\log p(x; \theta) = \log_z \frac{p(x, z; \theta)}{q(z)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x; \theta)$ | duì shù gài lǜ | 对数概率 |
| $q(z)$ | biàn fēn fēn bù | 变分分布 |
| $p(x, z; \theta)$ | lián hé gài lǜ | 联合概率 |

**一句话解释：**
通过引入变分分布$q(z)$来下界边缘对数概率。

**简单应用场景：**
推导EM算法和变分推理的理论基础。

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

### 公式 (11.45)：对数似然的单调增 / Monotonic Log-Likelihood Increase

$$
\log p(x; \theta_{t+1}) \geq \text{ELBO}(q_{t+1}, x; \theta_{t+1}) \geq \text{ELBO}(q_{t+1}, x; \theta_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(x; \theta_{t+1})$ | duì shù gài lǜ | 新参数的对数概率 |
| $\geq$ | dà yú huò děng yú | 大于等于 |

**一句话解释：**
EM算法保证每次迭代对数似然单调增加，收敛到局部最优。

**简单应用场景：**
理解EM算法收敛性的数学保证。

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

### 公式 (11.47)：ELBO的重写2 / ELBO Rewriting 2

$$
= \sum_z q(z) \left[\log p(x, z; \theta) - \log p(x; \theta)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x, z; \theta)$ | lián hé gài lǜ | 联合概率 |
| $\log p(x; \theta)$ | biān jì gài lǜ | 边际概率 |

**一句话解释：**
将对数概率分解为联合概率与边际概率的差。

**简单应用场景：**
ELBO推导的中间步骤。

---

### 公式 (11.48)：ELBO的重写3 / ELBO Rewriting 3

$$
= \sum_z q(z) \log \frac{p(x, z; \theta)}{q(z)} - \sum_z q(z) \log \frac{p(x, z; \theta)}{q(z)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{p(x, z; \theta)}{q(z)}$ | gài lǜ bi | 概率比 |

**一句话解释：**
ELBO包含期望项与KL散度项，衡量模型与变分分布的差异。

**简单应用场景：**
展示ELBO与KL散度的关系。

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
| $p(z\|x; \theta)$ | hòu yàn gài lǜ | 后验概率 |

**一句话解释：**
对数概率等于ELBO加上变分分布与后验的KL散度。

**简单应用场景：**
理解EM算法的理论基础，KL散度衡量近似质量。

---

## 11.2.2.2 高斯混合模型

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

### 公式 (11.51)：给定簇的条件分布 / Conditional Distribution Given Cluster

$$
p(x|z = k; \mu_k, \sigma_k) = \mathcal{N}(x; \mu_k, \sigma_k)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x\|z=k;...)$ | tiáo jiàn gài lǜ | 条件概率 |
| $\mathcal{N}$ | gāo sī fēn bù | 高斯分布 |
| $\mu_k$ | jūn zhí | 第k个簇的均值 |
| $\sigma_k$ | biāo zhǔn chā | 第k个簇的标准差 |

**一句话解释：**
给定簇k，样本$x$服从均值$\mu_k$、方差$\sigma_k$的高斯分布。

**简单应用场景：**
高斯混合模型中定义每个簇的分布。

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

### 公式 (11.56)：E步：后验概率1 / E-Step: Posterior Probability 1

$$
\gamma_{nk} \triangleq p(z^{(n)} = k | x^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\gamma_{nk}$ | hòu yàn gài lǜ | 后验概率 |
| $p(z^{(n)}=k\|x^{(n)})$ | gěi dìng x zhī yǔ z de gài lǜ | 给定$x$知$z$的概率 |

**一句话解释：**
后验概率$\gamma_{nk}$表示样本n属于簇k的概率。

**简单应用场景：**
EM算法的E步，计算隐变量的后验概率。

---

### 公式 (11.57)：E步：后验概率2 / E-Step: Posterior Probability 2

$$
= \frac{p(z^{(n)} = k) p(x^{(n)} | z^{(n)} = k)}{p(x^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(z^{(n)}=k)$ | xiān yàn gài lǜ | 先验概率 |
| $p(x^{(n)}\|z^{(n)}=k)$ | sì ránhuàn xìng | 似然度 |
| $p(x^{(n)})$ | biān jì gài lǜ | 边际概率 |

**一句话解释：**
使用贝叶斯公式计算后验概率：先验乘以似然再除以证据。

**简单应用场景：**
高斯混合模型EM算法的E步后验计算。

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

### 公式 (11.59)：M步：ELBO目标函数 / M-Step: ELBO Objective

$$
\text{ELBO}(\mathcal{D}; \pi, \mu, \sigma) = \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{nk} \log \frac{p(x^{(n)}, z^{(n)} = k)}{\gamma_{nk}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{ELBO}$ | zhèng jù xià jiè | 证据下界 |
| $\gamma_{nk}$ | hòu yàn gài lǜ | 后验概率 |
| $p(x^{(n)}, z^{(n)}=k)$ | lián hé gài lǜ | 联合概率 |

**一句话解释：**
M步的目标是最大化ELBO，关于$\gamma_{nk}$的加权和。

**简单应用场景：**
高斯混合模型EM算法的M步优化目标。

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

### 公式 (11.61)：M步：最终目标函数 / M-Step: Final Objective

$$
= \sum_{n=1}^{K} \gamma_{nk} \left(-\frac{(x - \mu_k)^2}{2\sigma_k^2} - \log \sigma_k + \log \pi_k\right) + C
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $C$ | cháng shù | 不含参数的常数项 |
| $(x - \mu_k)^2$ | píng fāng chā | 平方差 |

**一句话解释：**
ELBO的完整形式，包含高斯项与对数概率项。

**简单应用场景：**
对均值$\mu_k$、方差$\sigma_k$、混合系数$\pi_k$求偏导来更新参数。

---

### 公式 (11.62)：M步：参数优化问题 / M-Step: Parameter Optimization

$$
\max_{\pi, \mu, \sigma} \text{ELBO}(\mathcal{D}; \pi, \mu, \sigma)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà huà | 最大化 |
| $\pi, \mu, \sigma$ | suǒ yǒu cān shù | 所有模型参数 |

**一句话解释：**
M步通过最大化ELBO来更新混合系数、均值和方差。

**简单应用场景：**
高斯混合模型EM算法的M步优化任务描述。

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

## 第11章：推断与学习

### 公式 (11.67)：条件概率的贝叶斯定理 / Bayes' Theorem for Conditional Probability

$$
p(q|e) = \frac{p(e|q)}{p(e)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(q\|e)$ | pài | 给定证据e下查询q的条件概率 |
| $p(e\|q)$ | pài | 给定q下证据e的条件概率 |
| $p(e)$ | pài | 证据e的边缘概率 |
| $e$ | yī | 观测到的部分变量 |
| $q$ | chá xún | 查询变量 |

**一句话解释：**
给定观测证据，通过似然比和先验概率计算后验概率的基本公式。

**简单应用场景：**
医学诊断中，已知症状e，计算患病q的概率。

---

### 公式 (11.68)：边缘概率的归一化 / Normalization of Marginal Probability

$$
p(q|e) = \frac{\sum_z p(q, e, z)}{\sum_{q'z'} p(q', e, z')}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $z$ | zèi | 隐变量 |
| $q$ | chá xún | 查询变量 |
| $e$ | yī | 证据变量 |

**一句话解释：**
通过对隐变量求和，将联合概率转化为条件概率。

**简单应用场景：**
在图模型推理中，通过对所有隐变量求和计算边缘概率。

---

### 公式 (11.69)：条件独立性下的概率分解 / Probability Factorization Under Conditional Independence

$$
p(x_1, x_4) = \sum_{x_2, x_3} p(x_1) p(x_3 | x_1) p(x_3 | x_1) p(x_4 | x_2, x_3)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_1, x_2, x_3, x_4$ | xī | 随机变量 |

**一句话解释：**
利用条件独立性将联合概率分解为条件概率的乘积。

**简单应用场景：**
贝叶斯网络中根据图结构分解概率。

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

### 公式 (11.74)：链式模型的联合概率 / Joint Probability in Chain Model

$$
p(x) = \frac{1}{Z} \prod_{t=1}^{T} \phi(x_t, x_{t+1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Z$ | zī | 配分函数 |
| $\boldsymbol{\phi}(x_t, x_{t+1})$ | fài | 相邻节点间的势函数 |
| $T$ | tài | 序列长度 |
| $t$ | tì | 时间步 |

**一句话解释：**
线性链模型通过相邻节点的势函数乘积表示联合概率。

**简单应用场景：**
序列标注任务中的条件随机场。

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

### 公式 (11.76)：递推形式的边缘概率 / Recursive Form of Marginal Probability

$$
p(x_t) = \frac{1}{Z} \sum_{x_1} \cdots \sum_{x_{t-1}} \sum_{x_{t+1}} \cdots \sum_{x_T} \prod_{i=1}^{T-1} \phi(x_i, x_{i+1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_t$ | xī | 当前时刻变量 |
| $\boldsymbol{\phi}(x_i, x_{i+1})$ | fài | 相邻时刻的势函数 |
| $Z$ | zī | 配分函数 |

**一句话解释：**
将联合概率中的求和改写为递推形式，便于动态规划求解。

**简单应用场景：**
前向后向算法的理论基础。

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

### 公式 (11.81)：观测条件下的概率密度比 / Probability Density Ratio Under Observation

$$
p(z|x) = \frac{p(x, z)}{p(x)} = \frac{p(x, z)}{\int p(x, z) dz}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(z\|x)$ | pài | 观测x下隐变量z的条件概率 |
| $p(x, z)$ | pài | 联合概率 |
| $p(x)$ | pài | 边缘概率 |

**一句话解释：**
通过联合概率与边缘概率的比值得到条件概率。

**简单应用场景：**
变分推理和EM算法中的后验推理。

---

### 公式 (11.82)：变分贝叶斯目标 / Variational Bayes Objective

$$
q^*(z) = \arg\min_{q} KL(q(z) \| p(z|x))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q^*(z)$ | qiú | 最优变分分布 |
| $q(z)$ | qiú | 变分分布 |
| $p(z\|x)$ | pài | 真实后验分布 |
| $KL(\cdot \| \cdot)$ | kèi | KL散度 |

**一句话解释：**
寻找与真实后验最接近的变分分布来近似推理。

**简单应用场景：**
变分自编码器(VAE)和变分推理。

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
| $KL(\cdot \| \cdot)$ | kèi | KL散度 |
| $q(z)$ | qiú | 变分分布 |
| $p(z\|x)$ | pài | 后验分布 |

**一句话解释：**
证据等于ELBO加上变分分布与真实后验的KL散度。

**简单应用场景：**
EM算法和变分推理的理论基础。

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

### 公式 (11.91)：优化后的ELBO形式 / Optimized ELBO Form

$$
ELBO(q, x) = \int q_j \log p(x, z) dz_j - \int q_j \log q_j dz_j + \text{const}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_j$ | qiú | 第j个变量的变分分布 |
| $\log p(x, z)$ | pài | 联合概率对数 |
| const | cháng shù | 常数项 |

**一句话解释：**
消除其他变量的积分后，ELBO简化为目标分布的期望项和熵项。

**简单应用场景：**
坐标上升变分推理。

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

### 公式 (11.93)：最优因子的形式 / Form of Optimal Factor

$$
q_j^*(z_j) = E_{q_{-j}} [\log p(x, z)] + \text{const}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_j^*(z_j)$ | qiú | 第j个变量的最优分布 |
| $E_{q_{-j}}$ | qī wàng | 对其他变量分布的期望 |
| $\log p(x, z)$ | pài | 联合概率对数 |

**一句话解释：**
最优因子正比于联合概率对数在其他变量上的期望。

**简单应用场景：**
均值场变分推理的更新规则。

---

### 公式 (11.94)：最优因子的指数形式 / Exponential Form of Optimal Factor

$$
q_j^*(z_j) \propto p(x, z_j) \exp\left(E_{q_{-j}} [\log p(x, z)] \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q_j^*(z_j)$ | qiú | 第j个变量的最优分布 |
| $p(x, z_j)$ | pài | 观测和第j个变量的联合概率 |
| $E_{q_{-j}}$ | qī wàng | 对其他变量的期望 |

**一句话解释：**
最优变分分布具有指数族的形式。

**简单应用场景：**
实现高效的变分推理。

---

### 公式 (11.95)：期望的定义 / Definition of Expectation

$$
E_p[f(x)] = \int f(x) p(x) dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_p[\cdot]$ | qī wàng | 在分布p下的期望 |
| $f(x)$ | hánshù | 函数 |
| $p(x)$ | pài | 概率分布 |

**一句话解释：**
函数在概率分布下的平均值。

**简单应用场景：**
计算概率统计中的各种期望。

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

### 公式 (11.98)：拒绝采样的接受概率 / Acceptance Probability in Rejection Sampling

$$
\alpha(x) = \frac{p(x)}{kq(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha(x)$ | ā ěr fǎ | 接受概率 |
| $p(x)$ | pài | 目标分布 |
| $q(x)$ | qiú | 提议分布 |
| $k$ | kèi | 缩放系数 |

**一句话解释：**
接受采样样本的概率等于目标分布与缩放提议分布的比值。

**简单应用场景：**
从复杂分布采样。

---

### 公式 (11.99)：重要性采样的期望 / Expectation in Importance Sampling

$$
E_p[f(x)] = \int f(x) p(x) dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_p[\cdot]$ | qī wàng | 在分布p下的期望 |
| $f(x)$ | hánshù | 函数 |
| $p(x)$ | pài | 目标分布 |

**一句话解释：**
原始期望的定义。

**简单应用场景：**
重要性采样的理论基础。

---

### 公式 (11.100)：重要性采样的权重形式 / Weighted Form of Importance Sampling

$$
E_p[f(x)] = \int f(x) \frac{p(x)}{q(x)} q(x) dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_p[\cdot]$ | qī wàng | 期望 |
| $\frac{p(x)}{q(x)}$ | quán | 重要性权重 |
| $q(x)$ | qiú | 提议分布 |
| $p(x)$ | pài | 目标分布 |

**一句话解释：**
通过引入权重比例，将目标分布下的积分转化为提议分布下的积分。

**简单应用场景：**
从易采样分布近似难采样分布的期望。

---

### 公式 (11.101)：重要性加权的展开形式 / Expanded Form of Importance Weighting

$$
E_p[f(x)] = \int f(x) w(x) q(x) dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $w(x)$ | quán | 重要性权重函数 |
| $q(x)$ | qiú | 提议分布 |
| $f(x)$ | hánshù | 目标函数 |

**一句话解释：**
期望可以表示为在提议分布下加权函数的积分。

**简单应用场景：**
重要性采样的加权计算。

---

### 公式 (11.102)：期望的分解形式 / Decomposition Form of Expectation

$$
E_q[f(x) w(x)] = E_q[f(x) u(x)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_q[\cdot]$ | qī wàng | 在分布q下的期望 |
| $w(x)$ | quán | 权重函数 |
| $u(x)$ | yīng yòng | 加权函数 |

**一句话解释：**
期望可表示为加权函数在提议分布下的期望。

**简单应用场景：**
蒙特卡罗估计重要性采样。

---

### 公式 (11.103)：重要性采样的蒙特卡罗估计 / Monte Carlo Estimate of Importance Sampling

$$
\hat{f_N} = \frac{1}{N} \left( f(x^{(1)}) w(x^{(1)}) + \cdots + f(x^{(N)}) w(x^{(N)}) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{f_N}$ | fù | 估计值 |
| $w(x^{(i)})$ | quán | 第i个样本的权重 |
| $x^{(i)}$ | xī | 第i个样本 |
| $N$ | ēn | 样本数量 |

**一句话解释：**
用加权样本均值估计目标期望。

**简单应用场景：**
加权蒙特卡罗采样。

---

### 公式 (11.104)：重要性采样估计的规范化 / Normalized Form of Importance Sampling

$$
E_p[f(x)] = \int f(x) \frac{p(x)}{Z} dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_p[f(x)]$ | qī wàng | 期望 |
| $p(x)$ | pài | 非规范化分布 |
| $Z$ | zī | 配分函数 |

**一句话解释：**
当分布未规范化时，需要通过配分函数调整期望计算。

**简单应用场景：**
非规范化模型的推理。

---

### 公式 (11.105)：加权期望的规范化 / Normalized Weighted Expectation

$$
E_p[f(x)] = \frac{\int_x p(x) f(x) dx}{\int_x p(x) dx}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_p[f(x)]$ | qī wàng | 规范化期望 |
| $p(x)$ | pài | 非规范化分布 |
| $f(x)$ | hánshù | 函数 |

**一句话解释：**
非规范化分布下的期望需要通过分母进行规范化。

**简单应用场景：**
条件随机场和能量模型的推理。

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


## 第12章：深度信念网络


### 公式 12.1：玻尔兹曼机能量定义 / Energy Function of Boltzmann Machine

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

### 公式 12.2：能量函数展开式 / Energy Expansion

$$
E(x) \triangleq E(X = x) = -\left(\sum_{i<j}w_{ij}x_i x_j + \sum_i b_i x_i\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| w_{ij} | quán zhòng | 权重 (Weight) |
| x_i | biàn liàng | 变量 (Variable) |
| b_i | piānyì | 偏置 (Bias) |
| E(x) | néng liàng | 能量 (Energy) |

**一句话解释：**
玻尔兹曼机的能量函数由两项组成：变量之间的相互作用项和偏置项，用负号表示能量与概率的反向关系。

**简单应用场景：**
定义神经网络模型中的全局能量，用于学习和推理过程。

---

### 公式 12.3：玻尔兹曼分布 / Boltzmann Distribution

$$
p_s = \frac{1}{Z}\exp\left(-\frac{E_s}{kT}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| p_s | gài lǜ | 状态s的概率 (Probability of state s) |
| E_s | néng liàng | 状态s的能量 (Energy of state s) |
| k | Bō ěr zī màn cháng shù | 玻尔兹曼常数 (Boltzmann constant) |
| T | wēn dù | 温度 (Temperature) |
| Z | pīn fēn hánshù | 配分函数 (Partition Function) |

**一句话解释：**
热力学中的玻尔兹曼分布，描述粒子在给定温度下处于各个能量状态的概率。

**简单应用场景：**
统计物理学中描述热系统中的粒子分布和相变现象。

---

### 公式 12.4：状态能量比 / Energy Ratio

$$
\frac{p_s}{p_s'} = \exp\left(\frac{E_s - E_s'}{kT}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| p_s | gài lǜ | 状态s的概率 (Probability of state s) |
| p_s' | gài lǜ | 状态s'的概率 (Probability of state s') |
| E_s | néng liàng | 状态s的能量 (Energy of state s) |
| E_s' | néng liàng | 状态s'的能量 (Energy of state s') |
| k | cháng shù | 常数 (Constant) |
| T | wēn dù | 温度 (Temperature) |

**一句话解释：**
两个状态的概率比等于它们能量差的指数，能量低的状态概率更高。

**简单应用场景：**
理解系统中不同能量状态的相对概率关系。

---

### 公式 12.5：条件概率 - Logistic形式1 / Conditional Probability - Logistic Form 1

$$
p(x_i = 1|\boldsymbol{x}_{\setminus i}) = \sigma\left(\sum_j w_{ij}x_j + b_i\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| x_i | biàn liàng | 第i个变量 (Variable i) |
| \boldsymbol{x}_{\setminus i} | biàn liàng | 除了第i个外的所有变量 (All variables except i) |
| \sigma | xī gé mǎ | Sigmoid函数 (Sigmoid function) |
| w_{ij} | quán zhòng | 权重 (Weight) |
| b_i | piānyì | 偏置 (Bias) |

**一句话解释：**
在玻尔兹曼机中，给定其他所有变量的值，某个变量取值为1的条件概率由Sigmoid函数给出。

**简单应用场景：**
MCMC采样和推理过程中计算条件概率。

---

### 公式 12.6：条件概率 - 反向形式 / Conditional Probability - Reverse Form

$$
p(x_i = 0|\boldsymbol{x}_{\setminus i}) = 1 - p(x_i = 1|\boldsymbol{x}_{\setminus i})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| x_i | biàn liàng | 第i个变量 (Variable i) |
| p(x_i = 0\|\boldsymbol{x}_{\setminus i}) | gài lǜ | 条件概率 (Conditional probability) |
| p(x_i = 1\|\boldsymbol{x}_{\setminus i}) | gài lǜ | 条件概率 (Conditional probability) |

**一句话解释：**
变量取值为0的条件概率是1减去取值为1的条件概率，反映二元变量的互斥性。

**简单应用场景：**
二元随机变量的概率计算。

---

### 公式 12.7：能量间隙 / Energy Gap

$$
\Delta E_i(\boldsymbol{x}_i) = E(x_i = 0, \boldsymbol{x}_{\setminus i}) - E(x_i = 1, \boldsymbol{x}_{\setminus i})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \Delta E_i | néng liàng chā | 能量差 (Energy difference) |
| x_i | biàn liàng | 第i个变量 (Variable i) |
| \boldsymbol{x}_{\setminus i} | biàn liàng | 除了第i个外的所有变量 (All variables except i) |

**一句话解释：**
能量间隙表示第i个变量取两个不同值时的能量差，是影响条件概率的关键。

**简单应用场景：**
在推导条件概率时计算能量的变化。

---

### 公式 12.8：能量差求和形式 / Energy Difference Sum

$$
\Delta E_i = \sum_j w_{ij}x_j + b_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \Delta E_i | néng liàng chā | 能量差 (Energy difference) |
| w_{ij} | quán zhòng | 权重 (Weight) |
| x_j | biàn liàng | 变量 (Variable) |
| b_i | piānyì | 偏置 (Bias) |

**一句话解释：**
能量差可以表示为权重与变量乘积的和加上偏置，形式简洁高效。

**简单应用场景：**
在神经网络计算中快速求得能量差。

---

### 公式 12.9：对数似然 / Log Likelihood

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
| \log | duì shù | 对数 (Logarithm) |

**一句话解释：**
能量与概率的对数存在线性关系，通过温度参数T连接。

**简单应用场景：**
概率模型与能量函数的相互转换。

---

### 公式 12.10：能量差与条件概率的关系1 / Energy Difference - Conditional Probability 1

$$
\Delta E_i(\boldsymbol{x}_i) = -T\log p(x_i = 0, \boldsymbol{x}_{\setminus i}) - (-T\log p(x_i = 1, \boldsymbol{x}_{\setminus i}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \Delta E_i | néng liàng chā | 能量差 (Energy difference) |
| T | wēn dù | 温度 (Temperature) |
| p(x_i) | gài lǜ | 条件概率 (Conditional probability) |
| \log | duì shù | 对数 (Logarithm) |

**一句话解释：**
通过能量与概率的关系，可以将能量差表示为概率比的对数形式。

**简单应用场景：**
在条件概率推导中使用能量函数表示。

---

### 公式 12.11：概率比（对数形式）/ Probability Ratio - Log Form

$$
T\log\frac{p(x_i = 1, \boldsymbol{x}_{\setminus i})}{p(x_i = 0, \boldsymbol{x}_{\setminus i})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| T | wēn dù | 温度 (Temperature) |
| p(x_i = 1) | gài lǜ | 条件概率 (Conditional probability) |
| p(x_i = 0) | gài lǜ | 条件概率 (Conditional probability) |
| \log | duì shù | 对数 (Logarithm) |

**一句话解释：**
两个条件概率比的对数乘以温度T等于能量差。

**简单应用场景：**
从能量函数推导条件概率时的中间步骤。

---

### 公式 12.12：条件概率推导形式 / Conditional Probability Derivation Form

$$
T\log\frac{p(x_i = 1, \boldsymbol{x}_{\setminus i})}{1-p(x_i = 1, \boldsymbol{x}_{\setminus i})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| T | wēn dù | 温度 (Temperature) |
| p(x_i = 1) | gài lǜ | 条件概率 (Conditional probability) |
| \log | duì shù | 对数 (Logarithm) |

**一句话解释：**
利用概率的互补性，将两个条件概率的比值表示为单个概率与其补数的比值。

**简单应用场景：**
推导Sigmoid形式的条件概率。

---

### 公式 12.13：Sigmoid形式条件概率 / Conditional Probability - Sigmoid Form

$$
p(x_i = 1|\boldsymbol{x}_{\setminus i}) = \frac{1}{1+\exp\left(-\frac{\Delta E_i(\boldsymbol{x}_i)}{T}\right)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| p(x_i = 1\|\boldsymbol{x}_{\setminus i}) | gài lǜ | 条件概率 (Conditional probability) |
| \Delta E_i | néng liàng chā | 能量差 (Energy difference) |
| T | wēn dù | 温度 (Temperature) |
| \exp | zhǐ shù | 指数函数 (Exponential) |

**一句话解释：**
条件概率的标准形式为Sigmoid函数，由能量差和温度参数决定。

**简单应用场景：**
MCMC采样中计算接受概率，神经网络激活函数。

---

### 公式 12.14：Sigmoid与加权和的关系 / Sigmoid - Weighted Sum Relationship

$$
p(x_i = 1|\boldsymbol{x}_{\setminus i}) = \sigma\left(\frac{\sum_j w_{ij}x_j + b_i}{T}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| p(x_i = 1\|\boldsymbol{x}_{\setminus i}) | gài lǜ | 条件概率 (Conditional probability) |
| \sigma | xī gé mǎ | Sigmoid函数 (Sigmoid function) |
| w_{ij} | quán zhòng | 权重 (Weight) |
| x_j | biàn liàng | 变量 (Variable) |
| b_i | piānyì | 偏置 (Bias) |
| T | wēn dù | 温度 (Temperature) |

**一句话解释：**
条件概率等于加权和加偏置的Sigmoid函数，其中除以温度参数进行标度。

**简单应用场景：**
在神经网络和深度学习中实现条件概率计算。

---

### 公式 12.15：Hopfield网络激活规则 / Hopfield Network Activation Rule

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
| \sum_j | qiú hé | 求和 (Summation) |

**一句话解释：**
Hopfield网络的激活规则是确定性的：当加权和加偏置非负时输出1，否则输出0。

**简单应用场景：**
Hopfield网络的前向传播计算和状态更新。

---

### 公式 12.16：参数学习目标函数 / Parameter Learning Objective Function

$$
\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b}) = \frac{1}{N}\sum_{n=1}^N \log p(\boldsymbol{v}^{(n)}; \boldsymbol{W}, \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \mathcal{L} | sǔn shī hánshù | 损失函数 (Loss function) |
| \mathcal{D} | shù jù jí | 数据集 (Dataset) |
| \boldsymbol{W} | quán zhòng | 权重矩阵 (Weight matrix) |
| \boldsymbol{b} | piānyì | 偏置向量 (Bias vector) |
| p(\boldsymbol{v}) | gài lǜ | 概率 (Probability) |
| N | yàng běn shù | 样本数 (Number of samples) |

**一句话解释：**
参数学习的目标是最大化训练数据的对数似然，即所有样本的概率之和的对数的平均值。

**简单应用场景：**
深度信念网络的参数学习和模型训练。

---

### 公式 12.17：对数似然展开式1 / Log Likelihood Expansion 1

$$
\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b}) = \frac{1}{N}\sum_{n=1}^N \log\sum_h p(\boldsymbol{v}^{(n)}, \boldsymbol{h}; \boldsymbol{W}, \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \mathcal{L} | sǔn shī hánshù | 损失函数 (Loss function) |
| p(\boldsymbol{v}, \boldsymbol{h}) | gài lǜ | 联合概率 (Joint probability) |
| \boldsymbol{v} | kě jiàn biàn liàng | 可见变量 (Visible variables) |
| \boldsymbol{h} | yǐncáng biàn liàng | 隐藏变量 (Hidden variables) |
| \sum_h | qiú hé | 对隐藏变量求和 (Sum over hidden variables) |

**一句话解释：**
通过对隐藏变量求和，将联合概率转换为边际概率，形成完整的似然函数。

**简单应用场景：**
在有隐藏变量的图模型中计算观察数据的似然。

---

### 公式 12.18：对数似然展开式2 / Log Likelihood Expansion 2

$$
\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b}) = \frac{1}{N}\sum_{n=1}^N \log\frac{\sum_h\exp\left(-E(\boldsymbol{v}^{(n)}, \boldsymbol{h})\right)}{\sum_{u,h}\exp\left(-E(\boldsymbol{u}, \boldsymbol{h})\right)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \mathcal{L} | sǔn shī hánshù | 损失函数 (Loss function) |
| E(\boldsymbol{v}, \boldsymbol{h}) | néng liàng | 能量 (Energy) |
| \sum_h | qiú hé | 对隐藏变量求和 (Sum over hidden) |
| \sum_{u,h} | qiú hé | 对所有变量求和 (Sum over all) |
| \exp | zhǐ shù | 指数函数 (Exponential) |

**一句话解释：**
似然函数的另一种形式，分子是数据对应隐藏变量的能量之和的指数，分母是所有可能状态的能量之和的指数。

**简单应用场景：**
分子是能量的"正相位"贡献，分母是配分函数Z的"负相位"贡献。

---

### 公式 12.19：对数似然梯度1 / Log Likelihood Gradient 1

$$
\frac{\partial\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b})}{\partial\theta} = \frac{1}{N}\sum_{n=1}^N\frac{\partial}{\partial\theta}\log\sum_h p(\boldsymbol{v}^{(n)}, \boldsymbol{h}; \boldsymbol{W}, \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \frac{\partial\mathcal{L}}{\partial\theta} | piān dǎo shù | 偏导数 (Partial derivative) |
| \theta | xī tǎ | 参数 (Parameter) |
| N | yàng běn shù | 样本数 (Number of samples) |
| \sum_h | qiú hé | 对隐藏变量求和 (Sum over hidden) |

**一句话解释：**
通过对参数求偏导，得到对数似然的梯度，用于参数优化。

**简单应用场景：**
随机梯度下降法中的梯度计算。

---

### 公式 12.20：对数似然梯度2 / Log Likelihood Gradient 2

$$
\frac{\partial\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b})}{\partial\theta} = \frac{1}{N}\sum_{n=1}^N\frac{\partial}{\partial\theta}\left[\log\sum_h\exp\left(-E(\boldsymbol{v}^{(n)}, \boldsymbol{h})\right) - \log\sum_{u,h}\exp\left(-E(\boldsymbol{u}, \boldsymbol{h})\right)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \frac{\partial\mathcal{L}}{\partial\theta} | piān dǎo shù | 偏导数 (Partial derivative) |
| E(\boldsymbol{v}, \boldsymbol{h}) | néng liàng | 能量 (Energy) |
| \exp | zhǐ shù | 指数函数 (Exponential) |
| \log | duì shù | 对数 (Logarithm) |

**一句话解释：**
梯度分为两项：正相位（数据项）和负相位（配分函数项），分别对能量求导。

**简单应用场景：**
对比散度算法（Contrastive Divergence）中的核心计算。

---

### 公式 12.21：梯度的正负相位分离 / Positive and Negative Phases of Gradient

$$
\frac{\partial\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b})}{\partial\theta} = \frac{1}{N}\sum_{n=1}^N\sum_h\frac{\exp\left(-E(\boldsymbol{v}^{(n)}, \boldsymbol{h})\right)}{\sum_{u,h}\exp\left(-E(\boldsymbol{v}^{(n)}, \boldsymbol{h})\right)}\left[-\frac{\partial E(\boldsymbol{v}^{(n)}, \boldsymbol{h})}{\partial\theta}\right] - \sum_{u,h}\frac{\exp\left(-E(\boldsymbol{u}, \boldsymbol{h})\right)}{\sum_{u,h}\exp\left(-E(\boldsymbol{u}, \boldsymbol{h})\right)}\left[-\frac{\partial E(\boldsymbol{u}, \boldsymbol{h})}{\partial\theta}\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \frac{\partial\mathcal{L}}{\partial\theta} | piān dǎo shù | 偏导数 (Partial derivative) |
| E(\boldsymbol{v}, \boldsymbol{h}) | néng liàng | 能量 (Energy) |
| \frac{\partial E}{\partial\theta} | néng liàng de piān dǎo | 能量的偏导数 (Partial derivative of energy) |
| \sum_h | qiú hé | 对隐藏变量求和 (Sum over hidden) |

**一句话解释：**
梯度分为正相位（数据驱动）和负相位（模型驱动），两者通过能量函数的偏导相连。

**简单应用场景：**
深度学习中分离数据拟合和模型生成的两个阶段。

---

### 公式 12.22：梯度的期望表示 / Gradient - Expectation Representation

$$
\frac{\partial\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b})}{\partial\theta} = \frac{1}{N}\sum_{n=1}^N p(\boldsymbol{h}|\boldsymbol{v}^{(n)})\left[-\frac{\partial E(\boldsymbol{v}^{(n)}, \boldsymbol{h})}{\partial\theta}\right] - \sum_{u,h}p(\boldsymbol{u}, \boldsymbol{h})\left[-\frac{\partial E(\boldsymbol{u}, \boldsymbol{h})}{\partial\theta}\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \frac{\partial\mathcal{L}}{\partial\theta} | piān dǎo shù | 偏导数 (Partial derivative) |
| p(\boldsymbol{h}\|\boldsymbol{v}) | tiáo jiàn gài lǜ | 条件概率 (Conditional probability) |
| p(\boldsymbol{u}, \boldsymbol{h}) | lián hé gài lǜ | 联合概率 (Joint probability) |
| \frac{\partial E}{\partial\theta} | néng liàng de piān dǎo | 能量的偏导数 (Partial derivative of energy) |

**一句话解释：**
通过引入条件和联合概率，梯度可以表示为能量偏导数的期望值的差。

**简单应用场景：**
理论分析和算法推导中的概率解释。

---

### 公式 12.23：梯度的紧凑形式 / Gradient - Compact Form

$$
\frac{\partial\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b})}{\partial\theta} = E_{p(h|v)}[\operatorname{missing}] - E_{p(v, h)}[\operatorname{missing}]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| E_{p(\boldsymbol{h}\|\boldsymbol{v})} | tiáo jiàn qī wàng | 条件期望 (Conditional expectation) |
| E_{p(\boldsymbol{v}, \boldsymbol{h})} | lián hé qī wàng | 联合期望 (Joint expectation) |
| p(h\|v) | tiáo jiàn gài lǜ | 条件概率 (Conditional probability) |
| p(v, h) | lián hé gài lǜ | 联合概率 (Joint probability) |

**一句话解释：**
梯度可以简洁地表示为两个期望的差：条件期望（数据）减去联合期望（模型）。

**简单应用场景：**
简化梯度表达式，便于理论分析和算法设计。

---

### 公式 12.24：权重梯度 / Weight Gradient

$$
\frac{\partial\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b})}{\partial w_{ij}} = E_{p(h|v)}[x_i x_j] - E_{p(h,u)}[x_i x_j]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \frac{\partial\mathcal{L}}{\partial w_{ij}} | quán zhòng piān dǎo shù | 权重偏导数 (Weight gradient) |
| E_{p(h\|v)} | tiáo jiàn qī wàng | 条件期望 (Conditional expectation) |
| E_{p(h,u)} | lián hé qī wàng | 联合期望 (Joint expectation) |
| x_i, x_j | biàn liàng | 变量 (Variables) |

**一句话解释：**
权重梯度等于变量乘积的条件期望减去联合期望，反映数据驱动与模型生成的差异。

**简单应用场景：**
Hebbian学习规则在概率模型中的推广。

---

### 公式 12.25：偏置梯度 / Bias Gradient

$$
\frac{\partial\mathcal{L}(\mathcal{D}; \boldsymbol{W}, \boldsymbol{b})}{\partial b_i} = E_{p(h|v)}[x_i] - E_{p(h,u)}[x_i]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| \frac{\partial\mathcal{L}}{\partial b_i} | piānyì piān dǎo shù | 偏置偏导数 (Bias gradient) |
| E_{p(h\|v)} | tiáo jiàn qī wàng | 条件期望 (Conditional expectation) |
| E_{p(h,u)} | lián hé qī wàng | 联合期望 (Joint expectation) |
| x_i | biàn liàng | 变量 (Variable) |

**一句话解释：**
偏置梯度为单个变量期望值的条件期望减去联合期望，形式与权重梯度类似但不含乘积。

**简单应用场景：**
在梯度下降法中更新偏置参数。

---

### 公式 12.26：权重更新规则 / Weight Update Rule

$$
w_{ij} \leftarrow w_{ij} + \alpha\left(\langle x_i x_j \rangle_{\text{data}} - \langle x_i x_j \rangle_{\text{model}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| w_{ij} | quán zhòng | 权重 (Weight) |
| \alpha | xué xí lǜ | 学习率 (Learning rate) |
| \langle x_i x_j \rangle_{\text{data}} | shù jù qī wàng | 数据期望 (Data expectation) |
| \langle x_i x_j \rangle_{\text{model}} | mó xíng qī wàng | 模型期望 (Model expectation) |

**一句话解释：**
权重按照数据期望与模型期望的差异进行更新，学习率α控制更新的步长。

**简单应用场景：**
Contrastive Divergence算法和RBM训练中的核心权重更新。

---

### 公式 12.27：受限玻尔兹曼机能量函数 / RBM Energy Function

$$
E(\boldsymbol{v}, \boldsymbol{h}) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j}v_i w_{ij} h_j
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| E(\boldsymbol{v}, \boldsymbol{h}) | néng liàng | 能量 (Energy) |
| \boldsymbol{v} | kě jiàn biàn liàng | 可见层 (Visible variables) |
| \boldsymbol{h} | yǐncáng biàn liàng | 隐层 (Hidden variables) |
| a_i | piānyì | 可见层偏置 (Visible bias) |
| b_j | piānyì | 隐层偏置 (Hidden bias) |
| w_{ij} | quán zhòng | 权重 (Weight) |

**一句话解释：**
受限玻尔兹曼机的能量函数由三部分组成：可见层偏置、隐层偏置和可见-隐层之间的相互作用。

**简单应用场景：**
深度信念网络的基本构件单元。

---

### 公式 12.28：RBM能量函数向量形式 / RBM Energy Function - Vector Form

$$
E(\boldsymbol{v}, \boldsymbol{h}) = -\boldsymbol{a}^{\rm T}\boldsymbol{v} - \boldsymbol{b}^{\rm T}\boldsymbol{h} - \boldsymbol{v}^{\rm T}\boldsymbol{W}\boldsymbol{h}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| E(\boldsymbol{v}, \boldsymbol{h}) | néng liàng | 能量 (Energy) |
| \boldsymbol{a} | piānyì xiàng liàng | 可见层偏置向量 (Visible bias vector) |
| \boldsymbol{b} | piānyì xiàng liàng | 隐层偏置向量 (Hidden bias vector) |
| \boldsymbol{W} | quán zhòng jǔ zhèn | 权重矩阵 (Weight matrix) |
| \boldsymbol{v} | kě jiàn xiàng liàng | 可见变量向量 (Visible vector) |
| \boldsymbol{h} | yǐncáng xiàng liàng | 隐变量向量 (Hidden vector) |
| ^{\rm T} | zhuǎn zhì | 转置 (Transpose) |

**一句话解释：**
使用向量和矩阵记号紧凑地表示RBM能量函数，便于高效的矩阵运算。

**简单应用场景：**
在计算机实现中进行向量化计算和矩阵运算。

---

### 公式 12.29：RBM联合概率 / RBM Joint Probability

$$
p(\boldsymbol{v}, \boldsymbol{h}) = \frac{1}{Z}\exp(-E(\boldsymbol{v}, \boldsymbol{h}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| p(\boldsymbol{v}, \boldsymbol{h}) | lián hé gài lǜ | 联合概率 (Joint probability) |
| Z | pīn fēn hánshù | 配分函数 (Partition function) |
| E(\boldsymbol{v}, \boldsymbol{h}) | néng liàng | 能量 (Energy) |
| \exp | zhǐ shù | 指数函数 (Exponential) |

**一句话解释：**
RBM的联合概率分布由能量函数通过Boltzmann分布定义，配分函数Z确保概率和为1。

**简单应用场景：**
RBM的核心概率模型定义。

---

### 公式 12.30：RBM联合概率因式分解 / RBM Joint Probability Factorization

$$
p(\boldsymbol{v}, \boldsymbol{h}) = \frac{1}{Z}\exp(\boldsymbol{a}^{\rm T}\boldsymbol{v})\exp(\boldsymbol{b}^{\rm T}\boldsymbol{h})\exp(\boldsymbol{v}^{\rm T}\boldsymbol{W}\boldsymbol{h})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| p(\boldsymbol{v}, \boldsymbol{h}) | lián hé gài lǜ | 联合概率 (Joint probability) |
| Z | pīn fēn hánshù | 配分函数 (Partition function) |
| \boldsymbol{a} | piānyì xiàng liàng | 可见层偏置向量 (Visible bias vector) |
| \boldsymbol{b} | piānyì xiàng liàng | 隐层偏置向量 (Hidden bias vector) |
| \boldsymbol{W} | quán zhòng jǔ zhèn | 权重矩阵 (Weight matrix) |
| \boldsymbol{v} | kě jiàn xiàng liàng | 可见变量向量 (Visible vector) |
| \boldsymbol{h} | yǐncáng xiàng liàng | 隐变量向量 (Hidden vector) |
| \exp | zhǐ shù | 指数函数 (Exponential) |
| ^{\rm T} | zhuǎn zhì | 转置 (Transpose) |

**一句话解释：**
联合概率展开为三个指数项的乘积，分别对应可见层、隐层和它们之间的相互作用。

**简单应用场景：**
理解RBM中各部分参数对联合概率分布的贡献。

---



## 12.2 受限玻尔兹曼机 / Restricted Boltzmann Machine

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

### 公式 (12.37)：能量函数展开2 / Energy Function Expansion (2)

$$
= \frac{\exp(a^{\top}v)}{Z} \sum_{\boldsymbol{h}} \exp\left(\sum_j \left(b_j h_j + \sum_i w_{ij}v_i h_j\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp(a^{\top}v)$ | zh指shù hánshù | exponential of bias-visible term |
| $Z$ | fēn huà hánshù | partition function |
| $b_j$ | piàn zhì xiàng | bias of hidden unit $j$ |
| $w_{ij}$ | quán zhòng | weight |

**一句话解释：**
提取可观测层的偏置项，使隐变量相关的项可以按照节点进行分解。

**简单应用场景：**
为了进一步简化隐变量的边际化计算，将与可观测变量无关的项提取出来。

---

### 公式 (12.38)：能量函数展开3 / Energy Function Expansion (3)

$$
= \frac{\exp(a^{\top}v)}{Z} \prod_j \exp\left(b_j\left(b_j + \sum_i w_{ij}v_i\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp(a^{\top}v)$ | zhǐ shù hánshù | exponential of bias-visible term |
| $Z$ | fēn huà hánshù | partition function |
| $\prod_j$ | chéng jī | product over hidden units |
| $b_j$ | piàn zhì xiàng | bias of hidden unit $j$ |
| $w_{ij}$ | quán zhòng | weight |

**一句话解释：**
利用隐变量之间的条件独立性，将关于隐变量的求和转换为乘积形式。

**简单应用场景：**
通过乘积形式，可以对每个隐层节点分别计算其对边际概率的贡献。

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

### 公式 (12.40)：能量函数展开5 / Energy Function Expansion (5)

$$
= \frac{\exp(a^{\top}v)}{Z} \prod_j \exp\left(h_j\left(b_j + \sum_i w_{ij}v_i\right)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\exp(a^{\top}v)$ | zhǐ shù hánshù | exponential of bias-visible term |
| $Z$ | fēn huà hánshù | partition function |
| $\prod_j$ | chéng jī | product over hidden units |

**一句话解释：**
对二值隐变量进行求和，每个隐层节点贡献独立的因子。

**简单应用场景：**
每个隐层节点可以独立地对可观测层的边际概率做出贡献。

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

### 公式 (12.42)：隐层激活概率求和 / Summation of Hidden Unit Activations

$$
p(h_j = 1, v) = \frac{1}{Z} \sum_{k=1}^{K}k=1 \exp(-E(v, \boldsymbol{h}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(h_j = 1, v)$ | gěi dìng $v$ shí $h_j = 1$ de shì jiàn gài lǜ | joint probability of $h_j = 1$ and $v$ |
| $Z$ | fēn huà hánshù | partition function |
| $E(v, \boldsymbol{h})$ | néng liàng | energy |

**一句话解释：**
隐变量和可观测变量的联合概率分布。

**简单应用场景：**
在计算条件概率时，需要用到联合分布作为分子。

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

### 公式 (12.44)：隐变量条件概率2 / Conditional Probability of Hidden Variables (2)

$$
p(h_j = 1|\boldsymbol{v}) = \frac{p(h_j = 1, v)}{p(v)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(h_j = 1\|\boldsymbol{v})$ | gěi dìng kě jiàn biàn liàng shí yǐn cáng dān yuán jī huà de tiáo jiàn gài lǜ | conditional probability of $h_j = 1$ given $v$ |
| $p(h_j = 1, v)$ | lián hé gài lǜ | joint probability |
| $p(v)$ | biān jì gài lǜ | marginal probability |

**一句话解释：**
隐变量的条件概率等于联合概率与边际概率的比值。

**简单应用场景：**
利用贝叶斯定理计算条件概率。

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

### 公式 (12.47)：可观测变量条件概率2 / Conditional Probability of Visible Variables (2)

$$
p(v_i = 1|\boldsymbol{h}) = \sigma\left(a_i + \sum_j w_{ij}h_j\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(v_i = 1\|\boldsymbol{h})$ | gěi dìng yǐn cáng biàn liàng shí kě jiàn dān yuán jī huà de tiáo jiàn gài lǜ | conditional probability of $v_i = 1$ given $h$ |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $a_i$ | piàn zhì xiàng | bias of visible unit $i$ |
| $w_{ij}$ | quán zhòng | weight |

**一句话解释：**
给定隐层状态，可观测变量激活的条件概率。

**简单应用场景：**
在RBM的Gibbs采样中，用这个公式计算可观测层节点的激活概率。

---

### 公式 (12.48)：向量形式的隐层条件概率 / Vector Form of Hidden Layer Conditional Probability

$$
p(\boldsymbol{h} = 1|\boldsymbol{v}) = \sigma(\boldsymbol{W}^{\top}\boldsymbol{v} + \boldsymbol{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h} = 1\|\boldsymbol{v})$ | gěi dìng kě jiàn biàn liàng shí yǐn cáng céng de tiáo jiàn gài lǜ | conditional probability of hidden layer given visible variables |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | weight matrix |
| $\boldsymbol{v}$ | kě jiàn biàn liàng xiàng liàng | visible variable vector |
| $\boldsymbol{b}$ | yǐn cáng céng piàn zhì xiàng liàng | bias vector for hidden layer |

**一句话解释：**
隐层条件概率的向量形式，所有隐层节点可以并行计算。

**简单应用场景：**
在RBM的实现中，使用矩阵乘法并行计算所有隐层节点的激活概率。

---

### 公式 (12.49)：向量形式的可观测层条件概率 / Vector Form of Visible Layer Conditional Probability

$$
p(\boldsymbol{v} = 1|\boldsymbol{h}) = \sigma(\boldsymbol{W}\boldsymbol{h} + \boldsymbol{a})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{v} = 1\|\boldsymbol{h})$ | gěi dìng yǐn cáng biàn liàng shí kě jiàn céng de tiáo jiàn gài lǜ | conditional probability of visible layer given hidden variables |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{W}$ | quán zhòng jǔ zhèn | weight matrix |
| $\boldsymbol{h}$ | yǐn cáng biàn liàng xiàng liàng | hidden variable vector |
| $\boldsymbol{a}$ | kě jiàn céng piàn zhì xiàng liàng | bias vector for visible layer |

**一句话解释：**
可观测层条件概率的向量形式，所有可观测层节点可以并行计算。

**简单应用场景：**
在RBM的实现中，使用矩阵乘法并行计算所有可观测层节点的激活概率。

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
| $p(\boldsymbol{h}\|\boldsymbol{v})$ | gěi dìng kě jiàn biàn liàng de yǐn cáng biàn liàng tiáo jiàn gài lǜ | conditional distribution of hidden given visible |
| $p(\boldsymbol{v}, \boldsymbol{h})$ | mó xíng de lián hé gài lǜ | model's joint distribution |

**一句话解释：**
权重的梯度等于数据条件下的期望值减去模型联合分布下的期望值。

**简单应用场景：**
在受限玻尔兹曼机的参数学习中，用这个梯度进行梯度上升优化。

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

### 公式 (12.56)：隐层偏置更新规则 / Hidden Layer Bias Update Rule

$$
b_j \leftarrow b_j + \alpha\left((h_j)_{\text{data}} - (h_j)_{\text{model}}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b_j$ | yǐn cáng céng piàn zhì | hidden layer bias |
| $\alpha$ | xué xí lǜ | learning rate |
| $(h_j)_{\text{data}}$ | shù jù shang de $h_j$ qī wàng zhí | expected value of $h_j$ under data distribution |
| $(h_j)_{\text{model}}$ | mó xíng shang de $h_j$ qī wàng zhí | expected value of $h_j$ under model distribution |

**一句话解释：**
隐层偏置按照数据分布和模型分布下隐变量期望值之差的方向更新。

**简单应用场景：**
在RBM训练中，用这个规则更新隐层的偏置。

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

### 公式 (12.59)：深度信念网络的条件分布 / Conditional Distribution in DBN

$$
p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l-1)}, \ldots, \boldsymbol{h}^{(L)}) = p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h}^{(l)}\|\boldsymbol{h}^{(l-1)}, \ldots, \boldsymbol{h}^{(L)})$ | gěi dìng shàng céng de tiáo jiàn gài lǜ | conditional probability given lower layers |
| $p(\boldsymbol{h}^{(l)}\|\boldsymbol{h}^{(l+1)})$ | duì shàng yī céng tiáo jiàn gài lǜ | conditional probability given upper layer |
| $l$ | céng suō | layer index |

**一句话解释：**
在深度信念网络中，某一层的条件分布只取决于其上一层，具有Markov性质。

**简单应用场景：**
在DBN的推理过程中，可以利用这个性质简化计算。

---

### 公式 (12.60)：联合分布的因式分解1 / Factorization of Joint Distribution (1)

$$
p(v, \boldsymbol{h}^{(1)}, \ldots, \boldsymbol{h}^{(L)}) = p(v|\boldsymbol{h}^{(1)})\left(\prod_{l=1}^{L-2} p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)})\right) p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(v, \boldsymbol{h}^{(1)}, \ldots, \boldsymbol{h}^{(L)})$ | lián hé gài lǜ | joint distribution |
| $p(v\|\boldsymbol{h}^{(1)})$ | kě jiàn céng de tiáo jiàn gài lǜ | conditional distribution of visible layer |
| $\prod_{l=1}^{L-2} p(\boldsymbol{h}^{(l)}\|\boldsymbol{h}^{(l+1)})$ | duì jiàn tiáo jiàn gài lǜ | directed conditional distributions |
| $p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})$ | zuì hǎo liǎng céng de lián hé gài lǜ | joint distribution of top two layers |

**一句话解释：**
深度信念网络的联合分布可以分解为下向的有向条件概率和上层的无向联合分布。

**简单应用场景：**
在计算DBN的联合分布时，可以按这个方式分解。

---

### 公式 (12.61)：联合分布的因式分解2 / Factorization of Joint Distribution (2)

$$
= \left(\prod_{l=0}^{L-2} p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)})\right) p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\prod_{l=0}^{L-2}$ | duì $l$ cóng 0 dào $L-2$ de chéng jī | product from layer 0 to $L-2$ |
| $p(\boldsymbol{h}^{(l)}\|\boldsymbol{h}^{(l+1)})$ | céng jiān de tiáo jiàn gài lǜ | conditional distribution between layers |
| $p(\boldsymbol{h}^{(L-1)}, \boldsymbol{h}^{(L)})$ | zuì hǎo liǎng céng de lián hé gài lǜ | joint distribution of top two layers |

**一句话解释：**
简化后的联合分布因式分解形式，将可观测层视为第0层。

**简单应用场景：**
统一表示所有层之间的关系。

---

### 公式 (12.62)：隐层条件分布（Sigmoid形式） / Hidden Layer Conditional Distribution (Sigmoid Form)

$$
p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l+1)}) = \sigma\left(\boldsymbol{a}^{(l)} + \boldsymbol{W}^{(l+1)}\boldsymbol{h}^{(l+1)}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h}^{(l)}\|\boldsymbol{h}^{(l+1)})$ | dì $l$ céng de tiáo jiàn gài lǜ | conditional distribution of layer $l$ |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{a}^{(l)}$ | dì $l$ céng de piàn zhì xiàng liàng | bias vector of layer $l$ |
| $\boldsymbol{W}^{(l+1)}$ | dì $l+1$ céng de quán zhòng jǔ zhèn | weight matrix of layer $l+1$ |

**一句话解释：**
深度信念网络中，每一层给定上层时的条件分布可以用sigmoid函数表示。

**简单应用场景：**
在DBN中进行前向推理或生成采样时使用。

---

### 公式 (12.63)：深度信念网络顶层的条件概率 / Conditional Probability of Top Layer in DBN

$$
p(v = 1|\boldsymbol{h}) = \sigma\left(\boldsymbol{b} + \boldsymbol{w}^{\top}\boldsymbol{h}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(v = 1\|\boldsymbol{h})$ | gěi dìng yǐn céng de kě jiàn céng tiáo jiàn gài lǜ | conditional probability of visible layer |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{b}$ | piàn zhì xiàng liàng | bias vector |
| $\boldsymbol{w}$ | quán zhòng xiàng liàng | weight vector |
| $\boldsymbol{h}$ | yǐn cáng biàn liàng xiàng liàng | hidden variable vector |

**一句话解释：**
深度信念网络可见层给定隐层时的条件概率。

**简单应用场景：**
在DBN中重构可观测数据时使用。

---

### 公式 (12.64)：逐层预训练的RBM条件分布 / Layer-wise Pretraining RBM Conditional Distribution

$$
p(\boldsymbol{h}^{(l)}|\boldsymbol{h}^{(l-1)}) = \sigma\left(\boldsymbol{b}^{(l)} + \boldsymbol{W}^{(l)}\boldsymbol{h}^{(l-1)}\right), \quad 1 \leq l \leq (L-1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{h}^{(l)}\|\boldsymbol{h}^{(l-1)})$ | dì $l$ céng gěi dìng dì $l-1$ céng de tiáo jiàn gài lǜ | conditional distribution of layer $l$ given layer $l-1$ |
| $\sigma$ | xī gé mǎ | Logistic sigmoid function |
| $\boldsymbol{b}^{(l)}$ | dì $l$ céng de piàn zhì xiàng liàng | bias of layer $l$ |
| $\boldsymbol{W}^{(l)}$ | dì $l$ céng de quán zhòng jǔ zhèn | weight matrix of layer $l$ |

**一句话解释：**
在逐层预训练DBN时，每一层都可以看作一个RBM，其条件分布采用这种形式。

**简单应用场景：**
在DBN的逐层预训练阶段，将每一层的RBM与下一层的激活独立训练。


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

### 公式 (13.2)：隐变量模型的链式法则

$$
p(\boldsymbol{x}, \boldsymbol{z}; \boldsymbol{\theta}) = p(\boldsymbol{x}|\boldsymbol{z}; \boldsymbol{\theta})p(\boldsymbol{z}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{z}$ | z | 隐变量 |
| $\boldsymbol{\theta}$ | xī tǎ | 模型参数 |
| $p(\boldsymbol{x}\|\boldsymbol{z}; \boldsymbol{\theta})$ | p | 条件概率 |

**一句话解释：**
联合概率分解为条件概率和边际概率的乘积。

**简单应用场景：**
生成模型中表示可见和隐变量的关系。

---

### 公式 (13.3)：ELBO分解形式

$$
\log p(\boldsymbol{x}; \boldsymbol{\theta}) = \text{ELBO}(\boldsymbol{q}, \boldsymbol{x}; \boldsymbol{\theta}) + \text{KL}(\boldsymbol{q}(\boldsymbol{z}; \boldsymbol{\theta}), p(\boldsymbol{x}; \boldsymbol{\theta}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{ELBO}$ | ELBO | 证据下界 |
| $\boldsymbol{q}$ | q | 推断分布 |
| $\text{KL}$ | KL | 库尔贝克-莱布勒散度 |

**一句话解释：**
对数似然分解为证据下界和KL散度的和。

**简单应用场景：**
变分推断中的目标函数设计。

---

### 公式 (13.4)：ELBO的定义

$$
\text{ELBO}(\boldsymbol{q}, \boldsymbol{x}; \boldsymbol{\theta}) = E_{\boldsymbol{z} \sim q(\boldsymbol{z})} \left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z}; \boldsymbol{\theta})}{q(\boldsymbol{z}; \boldsymbol{\theta})}\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E$ | E | 期望 |
| $p(\boldsymbol{x}, \boldsymbol{z}; \boldsymbol{\theta})$ | p | 联合概率 |
| $q(\boldsymbol{z}; \boldsymbol{\theta})$ | q | 推断分布 |

**一句话解释：**
证据下界是联合概率和推断分布比值的期望。

**简单应用场景：**
变分自编码器的优化目标。

---

### 公式 (13.5)：后验概率的计算

$$
p(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\theta}) = \frac{p(\boldsymbol{x}, \boldsymbol{z}; \boldsymbol{\theta})p(\boldsymbol{z}; \boldsymbol{\theta})}{p(\boldsymbol{x}; \boldsymbol{\theta})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{z}\|\boldsymbol{x}; \boldsymbol{\theta})$ | p | 后验概率 |
| $p(\boldsymbol{x}; \boldsymbol{\theta})$ | p | 边际概率 |

**一句话解释：**
后验概率等于联合概率乘以先验，除以证据。

**简单应用场景：**
推断隐变量的分布。

---

### 公式 (13.6)：推断网络定义

$$
q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi}) = \mathcal{N}(\boldsymbol{\mu}_l; \boldsymbol{\sigma}_l^2\boldsymbol{I})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $q(\boldsymbol{z}\|\boldsymbol{x}; \boldsymbol{\phi})$ | q | 推断分布 |
| $\mathcal{N}$ | N | 高斯分布 |
| $\boldsymbol{\mu}_l$ | mǔ | 均值 |
| $\boldsymbol{\sigma}_l$ | xī gé mǎ | 标准差 |
| $\boldsymbol{\phi}$ | fài | 推断网络参数 |

**一句话解释：**
用神经网络参数化的高斯分布近似后验。

**简单应用场景：**
变分自编码器中的编码器网络。

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

### 公式 (13.10)：softplus激活函数应用

$$
\boldsymbol{\sigma}_l = \text{softplus}(\boldsymbol{W}^{(3)}\boldsymbol{h} + \boldsymbol{b}^{(3)}), \quad \text{softplus}(x) = \log(1 + e^x)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\sigma}_l$ | xī gé mǎ | 标准差 |
| $\text{softplus}$ | softplus | softplus函数 |
| $\boldsymbol{W}^{(3)}$ | W | 第三层权重 |

**一句话解释：**
使用softplus函数确保预测的方差为正。

**简单应用场景：**
高斯输出层的方差参数化。

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

### 公式 (13.12)：KL散度和ELBO的关系

$$
\text{KL}(q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi}), p(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\theta})) = \log p(\boldsymbol{x}; \boldsymbol{\theta}) - \text{ELBO}(\boldsymbol{q}, \boldsymbol{x}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{KL}$ | KL | 库尔贝克-莱布勒散度 |
| $\text{ELBO}$ | ELBO | 证据下界 |

**一句话解释：**
最小化KL散度等价于最大化ELBO。

**简单应用场景：**
变分推断的理论基础。

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

### 公式 (13.14)：ELBO形式的优化

$$
= \arg \min_{\boldsymbol{\phi}} \log p(\boldsymbol{x}; \boldsymbol{\theta}) - \text{ELBO}(\boldsymbol{q}, \boldsymbol{x}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\boldsymbol{x}; \boldsymbol{\theta})$ | p | 边际似然 |
| $\text{ELBO}$ | ELBO | 证据下界 |

**一句话解释：**
由于边际似然固定，最小化KL等价于最大化ELBO。

**简单应用场景：**
推断网络的实际优化方法。

---

### 公式 (13.15)：生成网络的优化目标

$$
= \arg \max_{\boldsymbol{\theta}} \text{ELBO}(\boldsymbol{q}, \boldsymbol{x}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | xī tǎ | 生成网络参数 |
| $\text{ELBO}$ | ELBO | 证据下界 |

**一句话解释：**
生成网络通过最大化证据下界来优化。

**简单应用场景：**
变分自编码器的解码器训练。

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

### 公式 (13.19)：生成网络的学习目标

$$
\boldsymbol{\theta}^* = \arg \max_{\boldsymbol{\theta}} \text{ELBO}(\boldsymbol{q}, \boldsymbol{x}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}^*$ | xī tǎ | 最优生成参数 |
| $\text{ELBO}$ | ELBO | 证据下界 |

**一句话解释：**
生成网络参数通过最大化证据下界来优化。

**简单应用场景：**
变分自编码器的学习过程。

---

### 公式 (13.20)：变分自编码器的总目标函数

$$
\max_{\boldsymbol{\theta},\boldsymbol{\phi}} \text{ELBO}(\boldsymbol{q}, \boldsymbol{x}; \boldsymbol{\theta}) = \max_{\boldsymbol{\theta},\boldsymbol{\phi}} E_{\boldsymbol{z} \sim q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi})} \left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z}; \boldsymbol{\theta})}{q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi})}\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\theta}$ | xī tǎ | 生成网络参数 |
| $\boldsymbol{\phi}$ | fài | 推断网络参数 |
| $\text{ELBO}$ | ELBO | 证据下界 |

**一句话解释：**
VAE同时优化生成网络和推断网络的参数。

**简单应用场景：**
变分自编码器的完整训练目标。

---

### 公式 (13.21)：ELBO的分解

$$
= \max E_{\boldsymbol{z} \sim q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi})} [\log p(\boldsymbol{x}|\boldsymbol{z}; \boldsymbol{\theta})] - \text{KL}(q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\theta}), p(\boldsymbol{z}; \boldsymbol{\theta}))
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E$ | E | 期望 |
| $\text{KL}$ | KL | 库尔贝克-莱布勒散度 |

**一句话解释：**
ELBO分解为重建误差和正则项的加权和。

**简单应用场景：**
理解VAE的优化目标组成。

---

### 公式 (13.22)：ELBO估计的期望项

$$
E_{\boldsymbol{z} \sim q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi})} [\log p(\boldsymbol{x}|\boldsymbol{z}; \boldsymbol{\theta})]\approx \frac{1}{M} \sum_{m=1}^{M} \log p(x^{(m)}; \boldsymbol{\theta})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $M$ | M | 蒙特卡洛样本数 |
| $x^{(m)}$ | x | 第m个样本 |

**一句话解释：**
使用蒙特卡洛采样估计期望。

**简单应用场景：**
VAE的重建误差项计算。

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

### 公式 (13.24)：简化的KL散度（标准正态先验）

$$
\text{KL}(q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi}), p(\boldsymbol{z}; \boldsymbol{\theta})) = \frac{1}{2}\left(\text{tr}(\boldsymbol{\sigma}I) + \boldsymbol{\mu}_l^T\boldsymbol{\mu}_l - d - \log(\boldsymbol{\sigma}I)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\mu}_l$ | mǔ | 推断均值 |
| $\boldsymbol{\sigma}$ | xī gé mǎ | 推断方差 |
| $d$ | d | 隐变量维度 |

**一句话解释：**
假设标准正态先验时KL散度的简化形式。

**简单应用场景：**
VAE的标准实现中的正则项。

---

### 公式 (13.25)：重参数化技巧

$$
E_{\boldsymbol{z} \sim q(\boldsymbol{z}|\boldsymbol{x}; \boldsymbol{\phi})} [\log p(\boldsymbol{x}|\boldsymbol{z}; \boldsymbol{\theta})] = E_{\boldsymbol{\varepsilon} \sim p(\boldsymbol{\varepsilon})} [\log p(\boldsymbol{x}|g(\boldsymbol{x}, \boldsymbol{\varepsilon}; \boldsymbol{\phi}); \boldsymbol{\theta})]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{\varepsilon}$ | epsilon | 噪声向量 |
| $g$ | g | 重参数化函数 |

**一句话解释：**
重参数化技巧使梯度能够通过采样操作反向传播。

**简单应用场景：**
VAE的梯度计算和训练。

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

### 公式 (13.27)：变分自编码器的目标函数

$$
J(\boldsymbol{\theta}, \boldsymbol{\phi}|\mathcal{D}) = \sum_{n=1}^{N} \left[\frac{1}{M} \sum_{m=1}^{M} \log p(x^{(m)}|\boldsymbol{z}^{(n,m)}; \boldsymbol{\theta}) - \text{KL}(q(\boldsymbol{z}|x^{(n)}; \boldsymbol{\phi}), \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}))\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{D}$ | D | 数据集 |
| $N$ | N | 样本数 |
| $M$ | M | 蒙特卡洛样本数 |

**一句话解释：**
VAE的完整目标函数涵盖所有训练样本。

**简单应用场景：**
批量训练变分自编码器。

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

### 公式 (13.30)：判别网络的损失函数

$$
\min - \left(E_x [\log p(\boldsymbol{y} = 1|\boldsymbol{x}) + (1-y)\log p(\boldsymbol{y} = 0|\boldsymbol{x})]\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E$ | E | 期望 |
| $p(\boldsymbol{y} = 1\|\boldsymbol{x})$ | p | 真实概率 |

**一句话解释：**
判别网络最小化交叉熵损失。

**简单应用场景：**
GAN中判别器的训练。

---

### 公式 (13.31)：生成网络与判别网络的博弈

$$
\max E_{x \sim p_d(x)} [\log D(\boldsymbol{x}; \boldsymbol{\phi})] + E_{y \sim p_d(x)} [\log(1 - D(G(\boldsymbol{z}; \boldsymbol{\theta}); \boldsymbol{\phi}))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D$ | D | 判别网络 |
| $G$ | G | 生成网络 |
| $p_d$ | p | 真实数据分布 |

**一句话解释：**
判别网络最大化正确分类的概率。

**简单应用场景：**
GAN的目标函数。

---

### 公式 (13.32)：生成网络的损失

$$
= \max E_{x \sim p_d(x)} [\log D(\boldsymbol{x}; \boldsymbol{\phi})] + E_{\boldsymbol{z} \sim p_z} [\log(1 - D(G(\boldsymbol{z}; \boldsymbol{\theta}); \boldsymbol{\phi}))]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_z$ | p | 隐变量分布 |
| $\boldsymbol{z}$ | z | 隐变量样本 |

**一句话解释：**
生成网络通过最小化判别网络对生成样本的置信度来学习。

**简单应用场景：**
GAN的生成器训练。

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

### 公式 (13.34)：对数概率目标

$$
= \min\left(E_{\boldsymbol{z} \sim p_z} \left[\log(1 - D(G(\boldsymbol{z}; \boldsymbol{\theta}); \boldsymbol{\phi}))\right]\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log(1 - D(\cdot))$ | log | 对数概率 |

**一句话解释：**
等价地，最小化生成样本被判别为假的概率。

**简单应用场景：**
GAN生成器的替代损失。

---

## 第14章：深度强化学习

### 公式 (14.1)：策略定义 / Policy Definition

$$
\pi(a|s) \triangleq p(a|s)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi(a\|s)$ | zhèngcè | 策略：给定状态s下采取行动a的概率 |
| $p(a\|s)$ | gàilǜ | 概率分布 |
| $a$ | xíngdòng | 行动 |
| $s$ | zhuàngtài | 状态 |

**一句话解释：**
策略是从状态映射到行动概率分布的函数。

**简单应用场景：**
在强化学习中定义智能体的决策规则，例如AlphaGo的策略网络。

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

### 公式 (14.4)：状态转移概率 / State Transition Probability

$$
p(s_{t+1}|s_t, \cdots, s_0) = p(s_{t+1}|s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(s_{t+1}\|s_t, \cdots, s_0)$ | zhuàngtài zhuǎnyí gàilǜ | 完整状态转移概率 |
| $p(s_{t+1}\|s_t)$ | zuìjìn zhuàngtài zhuǎnyí gàilǜ | 马尔可夫性质下的转移概率 |
| $s_t$ | xiànzài zhuàngtài | 当前状态 |
| $s_{t+1}$ | xiàyībù zhuàngtài | 下一个状态 |

**一句话解释：**
马尔可夫性质：下一个状态只取决于当前状态，而与历史无关。

**简单应用场景：**
游戏棋盘的下一个棋局仅取决于当前位置。

---

### 公式 (14.5)：联合状态转移概率 / Joint State Transition Probability

$$
p(s_{t+1}|s_t, a_t, \cdots, s_0, a_0) = p(s_{t+1}|s_t, a_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(s_{t+1}\|s_t, a_t, \cdots, s_0, a_0)$ | wánzhěng zhuǎnyí gàilǜ | 完整转移概率 |
| $p(s_{t+1}\|s_t, a_t)$ | mǎěrkē fū xìngzhì | 马尔可夫性质下的转移 |
| $a_t$ | shíkè xíngdòng | 时刻t的行动 |

**一句话解释：**
给定当前状态和行动，下一状态的概率只与当前信息相关。

**简单应用场景：**
机器人在给定指令和当前位置后，计算移动到新位置的概率。

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

### 公式 (14.7)：轨迹概率分解 / Trajectory Probability Decomposition

$$
p(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi(a_t|s_t) p(s_{t+1}|s_t, a_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(\tau)$ | guǐjī gàilǜ | 轨迹概率 |
| $p(s_0)$ | chūshǐ zhuàngtài gàilǜ | 初始状态分布 |
| $\pi(a_t\|s_t)$ | zhèngcè | 策略 |
| $p(s_{t+1}\|s_t, a_t)$ | huánjìng dònghuà | 环境动力学 |
| $T$ | zuìdà shíchāng | 最大时长 |

**一句话解释：**
轨迹概率由初始分布、策略和环境动力学相乘组成。

**简单应用场景：**
计算特定游戏序列的发生概率。

---

### 公式 (14.8)：累积回报 / Cumulative Return

$$
G(t) = \sum_{t'=t}^{T-1} r_{t'+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(t)$ | jīlěi huíbào | 累积回报 |
| $r_{t'+1}$ | jiǎnglì | 奖励 |
| $t$ | dāngqián shíkè | 当前时刻 |
| $T$ | zuìdà shíchāng | 最大时长 |

**一句话解释：**
从当前时刻起，所有未来奖励的总和。

**简单应用场景：**
游戏中从当前回合开始到游戏结束获得的总分。

---

### 公式 (14.9)：累积回报递归形式 / Cumulative Return Recursive Form

$$
G(t) = \sum_{t'=0}^{T-1} r(s_t, a_t, s_{t+1})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(t)$ | jīlěi huíbào | 累积回报 |
| $r(s_t, a_t, s_{t+1})$ | jiǎnglì hánshù | 奖励函数 |
| $s_t$ | zhuàngtài | 状态 |
| $a_t$ | xíngdòng | 行动 |

**一句话解释：**
累积回报为所有转移步奖励的求和。

**简单应用场景：**
计算棋局中每一步的价值总和。

---

### 公式 (14.10)：折扣累积回报 / Discounted Cumulative Return

$$
G(t) = \sum_{t'=0}^{T-1} \gamma^{t'} r_{t'+1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(t)$ | zhéhuà jīlěi huíbào | 折扣累积回报 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子（0到1之间） |
| $r_{t'+1}$ | jiǎnglì | 奖励 |
| $t'$ | shíjiān bǔchōng | 时间补偿 |

**一句话解释：**
未来奖励相对于当前的价值会随时间衰减。

**简单应用场景：**
投资中考虑通货膨胀时，未来收益折算到现在的价值。

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

### 公式 (14.12)：状态值函数分解 / State Value Function Decomposition

$$
E_{\tau \sim p_\theta(\tau)} \left[G(\tau)\right] = E_{\tau \sim p_\theta(\tau)} \left[\sum_{t=0}^{T-1} \gamma^t r_{t+1}|t_{s_0} = s\right] = E_{\tau \sim p_\theta(\tau)} \left[V^\pi(s)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^\pi(s)$ | zhuàngtài jiàzhíhánshù | 状态值函数 |
| $s$ | zhuàngtài | 状态 |
| $\pi$ | zhèngcè | 策略 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |

**一句话解释：**
状态值函数表示从某状态开始的期望累积回报。

**简单应用场景：**
评估棋盘上某个位置对于玩家的好坏程度。

---

### 公式 (14.13)：状态值函数 / State Value Function

$$
E_{\tau \sim p_\theta(\tau)} \left[V^\pi(s)\right] = V^\pi(s)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^\pi(s)$ | zhuàngtài jiàzhíhánshù | 在策略π下的状态值 |
| $s$ | zhuàngtài | 状态 |
| $\pi$ | zhèngcè | 策略 |

**一句话解释：**
状态值函数定义为期望的回报。

**简单应用场景：**
深度强化学习中用神经网络近似的值函数。

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

### 公式 (14.15)：状态值的展开形式 / State Value Expanded Form

$$
V^\pi(s) = E_{a_t \sim \pi(s)} \left[a + \sum_{t=1}^{T-1} \gamma^{t-1} r_{t}|s_0 = s\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^\pi(s)$ | zhuàngtài jiàzhíhánshù | 状态值函数 |
| $a$ | xíngdòng | 行动 |
| $\pi(s)$ | zhèngcè | 从状态s采样行动的概率 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |

**一句话解释：**
状态值可以分解为当前行动和未来期望值的组合。

**简单应用场景：**
动态规划中的价值迭代算法。

---

### 公式 (14.16)：状态值递推展开 / State Value Recursive Expansion

$$
E_{a_t \sim a(s)} E_{\tau' \sim p'(s',a)} \left[r(s, a') + \sum_{t'=1}^{T-1} \gamma^{t'-1} r_{t+1}|s' = s'\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r(s, a')$ | jiǎnglì hánshù | 在状态s执行行动a'的奖励 |
| $s'$ | xiàyībù zhuàngtài | 下一个状态 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |

**一句话解释：**
状态值包含即时奖励和下一状态的期望值。

**简单应用场景：**
贝尔曼方程中的值分解。

---

### 公式 (14.17)：状态值的完整递推 / State Value Complete Recursive Form

$$
E_{a_t \sim a(s)} E_{\tau' \sim p'(s',a)} \left[r(s, a') + \gamma E_{\tau'' \sim p_\theta(\tau)} \left[\sum_{t'=1}^{T-1} \gamma^{t'-1} r_{t+1}|s' = s'\right]\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $V^\pi(s')$ | xià yī zhuàngtài de jiàzhí | 下一状态的值函数 |

**一句话解释：**
状态值等于当前奖励加折扣后的下一状态值。

**简单应用场景：**
强化学习中的时间差分学习。

---

### 公式 (14.18)：贝尔曼方程 / Bellman Equation

$$
E_{a_t \sim a(s)} E_{\tau' \sim p'(s',a)} \left[r(s, a') + \gamma V^\pi(s')\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^\pi(s)$ | zhuàngtài jiàzhíhánshù | 状态值函数 |
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $s'$ | xiàyībù zhuàngtài | 下一状态 |

**一句话解释：**
当前状态的值等于即时奖励加折扣的下一状态值的期望。

**简单应用场景：**
深度强化学习中计算TD目标的基础公式。

---

### 公式 (14.19)：状态-行动值函数 / State-Action Value Function

$$
Q^\pi(s, a) = E_{\tau' \sim p'(s',a)} \left[r(s, a') + \gamma V^\pi(s')\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^\pi(s, a)$ | zhuàngtài-xíngdòng jiàzhíhánshù | 状态-行动值函数 |
| $s$ | zhuàngtài | 状态 |
| $a$ | xíngdòng | 行动 |
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |

**一句话解释：**
Q值表示在给定状态下执行特定行动的期望回报。

**简单应用场景：**
DQN算法中的核心值函数。

---

### 公式 (14.20)：V函数与Q函数关系 / Relationship between V and Q Functions

$$
V^\pi(s) = E_{a \sim \pi(a|s)} \left[Q^\pi(s, a)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^\pi(s)$ | zhuàngtài jiàzhíhánshù | 状态值函数 |
| $Q^\pi(s, a)$ | zhuàngtài-xíngdòng jiàzhíhánshù | Q值函数 |
| $\pi(a\|s)$ | zhèngcè | 策略分布 |
| $a$ | xíngdòng | 行动 |

**一句话解释：**
状态值是所有可能行动的Q值的加权平均（按策略加权）。

**简单应用场景：**
将值函数和Q函数联系起来。

---

### 公式 (14.21)：Q函数的递推关系 / Q Function Recursive Relation

$$
Q^\pi(s, a) = E_{\tau' \sim p'(s',a)} \left[r(s, a') + \gamma E_{\tau'' \sim \pi} \left[Q^\pi(s', a')\right]\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^\pi(s, a)$ | zhuàngtài-xíngdòng jiàzhíhánshù | Q值函数 |
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $Q^\pi(s', a')$ | xiàyībù Q zhíhánshù | 下一状态的Q值 |

**一句话解释：**
Q值满足贝尔曼最优方程。

**简单应用场景：**
强化学习中的Q值迭代。

---

### 公式 (14.22)：策略优化 / Policy Optimization

$$
\pi^* = \arg \max_a V^\pi(s)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi^*$ | zuì yōu zhèngcè | 最优策略 |
| $\arg \max$ | cānshù zuìdà huà | 参数最大化 |
| $V^\pi(s)$ | zhuàngtài jiàzhíhánshù | 状态值函数 |

**一句话解释：**
最优策略是选择使状态值最大的行动。

**简单应用场景：**
贪心策略：总是选择当前最好的行动。

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
| $\pi'(a\|s)$ | zuì yōu zhèngcè | 最优策略（确定性） |
| $Q^\pi(s, a)$ | zhuàngtài-xíngdòng jiàzhíhánshù | Q值函数 |
| $a$ | xíngdòng | 行动 |

**一句话解释：**
最优策略是选择Q值最大的行动的确定性策略。

**简单应用场景：**
贪心算法在特定状态下选择最佳行动。

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

### 公式 (14.25)：最优策略的性质 / Optimal Policy Property

$$
\text{vs.}, \quad V^{\pi^*}(s) \geq V^\pi(s)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^{\pi^*}(s)$ | zuì yōu zhèngcè xià de jiàzhí | 最优策略下的状态值 |
| $V^\pi(s)$ | rèngyì zhèngcè xià de jiàzhí | 任意策略下的状态值 |
| $s$ | zhuàngtài | 状态 |

**一句话解释：**
最优策略的值函数大于等于任何其他策略的值函数。

**简单应用场景：**
证明最优策略的收敛性。

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

### 公式 (14.27)：最优Q值函数 / Optimal Q Function

$$
V^*(s) = \max_a E_{\tau' \sim p'(s',a)} \left[r(s, a') + \gamma V^*(s')\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $V^*(s)$ | zuì yōu zhuàngtài jiàzhí | 最优状态值函数 |
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $V^*(s')$ | xiàyībù zuì yōu zhuàngtài jiàzhí | 下一状态最优值 |

**一句话解释：**
最优状态值满足贝尔曼最优方程。

**简单应用场景：**
值迭代算法的更新规则。

---

### 公式 (14.28)：贝尔曼最优性方程 / Bellman Optimality Equation

$$
Q^*(s, a) = E_{\tau' \sim p'(s',a)} \left[r(s, a') + \gamma \max_{a'} Q^*(s', a')\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^*(s, a)$ | zuì yōu zhuàngtài-xíngdòng jiàzhí | 最优Q值函数 |
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $\max_{a'}$ | xíngdòng zuìdà huà | 行动最大化 |

**一句话解释：**
最优Q值等于即时奖励加上下一状态最大Q值的折扣。

**简单应用场景：**
DQN算法的核心更新方程。

---

### 公式 (14.29)：无模型RL中Q值近似 / Q Value Approximation in Model-Free RL

$$
Q^\pi(s, a) = E_{\tau \sim p_\theta(\tau)} \left[G(s_\tau=s, a_\tau=a)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^\pi(s, a)$ | zhuàngtài-xíngdòng jiàzhíhánshù | Q值函数 |
| $G(s_\tau=s, a_\tau=a)$ | huíbào | 回报 |
| $\tau$ | guǐjī | 轨迹 |

**一句话解释：**
Q值是在给定状态-行动对下的期望回报。

**简单应用场景：**
经验回放中使用真实轨迹估计Q值。

---

### 公式 (14.30)：Q值的经验估计 / Empirical Q Value Estimation

$$
Q^\pi(s, a) \approx \hat{Q}(s, a) = \frac{1}{N} \sum_{n=1}^{N} G(s_\tau^{(n)}=s, a_\tau^{(n)}=a)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q^\pi(s, a)$ | zhēnshí Q zhí | 真实Q值 |
| $\hat{Q}(s, a)$ | Q zhí jìngsì | Q值估计 |
| $N$ | yàngpǐn shùliàng | 样本数量 |
| $G(s_\tau^{(n)})$ | huíbào | 回报 |

**一句话解释：**
用多个采样的回报平均值近似Q值。

**简单应用场景：**
蒙特卡洛方法评估策略性能。

---

### 公式 (14.31)：ε-贪心策略 / Epsilon-Greedy Policy

$$
\pi^*(s) = \begin{cases}
\pi(s), & \text{按概率} 1-\varepsilon \\
\text{随机选择中的行动}, & \text{按概率} \varepsilon
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi^*(s)$ | ε-tānxīn zhèngcè | ε-贪心策略 |
| $\pi(s)$ | zuì yōu xíngdòng | 最优行动 |
| $\varepsilon$ | ēpsilon | 探索概率 |

**一句话解释：**
以概率ε选择随机行动用于探索，否则利用最优策略。

**简单应用场景：**
平衡探索-利用权衡。

---

### 公式 (14.32)：时序差分学习Q值估计 / Temporal Difference Learning Q Value Estimate

$$
\hat{Q}_N^{(n)}(s, a) = \frac{1}{N} \sum_{n=1}^{N} G(\underbrace{s_\tau=s, a_\tau=a}_{})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}_N^{(n)}(s, a)$ | N cì cèshì hòu de Q zhí | N次测试后的Q值 |
| $G$ | huíbào | 回报 |
| $s$ | zhuàngtài | 状态 |
| $a$ | xíngdòng | 行动 |

**一句话解释：**
用样本平均值估计Q值。

**简单应用场景：**
蒙特卡洛Q学习。

---

### 公式 (14.33)：时序差分Q值更新 / Temporal Difference Q Update

$$
\frac{1}{N} \left(G(\underbrace{s_\tau=s, a_\tau=a}_{}) + \sum_{n=1}^{N-1} G(\underbrace{s_\tau=s, a_\tau=a}_{})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G$ | huíbào | 回报 |
| $N$ | yàngpǐn shùliàng | 样本数量 |
| $n$ | yàngpǐn suǒyǐn | 样本索引 |

**一句话解释：**
累积样本回报的平均值。

**简单应用场景：**
样本高效的Q学习。

---

### 公式 (14.34)：增量Q值更新 / Incremental Q Update

$$
\frac{1}{N} \left(G(\underbrace{s_\tau=s, a_\tau=a}_{}) + (N-1)\hat{Q}_{N-1}(\underbrace{s_\tau=s, a_\tau=a}_{})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}_{N-1}$ | qiánjìn Q zhí | 前一个Q值估计 |
| $N$ | yàngpǐn shùliàng | 样本数量 |

**一句话解释：**
增量方式更新Q值，避免重新计算所有历史数据。

**简单应用场景：**
在线Q学习算法。

---

### 公式 (14.35)：Q值增量形式 / Q Value Incremental Form

$$
\hat{Q}_{N-1}(s, a) + \frac{1}{N}\left(G(\underbrace{s_\tau=s, a_\tau=a}_{})-\hat{Q}_{N-1}(\underbrace{s_\tau=s, a_\tau=a}_{})\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}_{N-1}(s, a)$ | qiánjìn Q zhí | 前一个Q值 |
| $G$ | xīn huíbào | 新回报 |
| $\frac{1}{N}$ | xuéxí lǜ | 学习率 |

**一句话解释：**
按学习率调整Q值朝向新样本的方向。

**简单应用场景：**
梯度下降风格的Q学习更新。

---

### 公式 (14.36)：时序差分误差 / Temporal Difference Error

$$
\hat{Q}^*(s, a) - \hat{Q}(s, a) + \alpha\left(G(\tau_{s=a}=a) - \hat{Q}^*(s, a)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}^*(s, a)$ | zuì yōu Q zhí jìngsì | 最优Q值估计 |
| $\hat{Q}(s, a)$ | dāngqián Q zhí jìngsì | 当前Q值估计 |
| $\alpha$ | ālfa | 学习率 |
| $G(\tau_{s=a}=a)$ | huíbào | 回报 |

**一句话解释：**
TD误差用于衡量预测与实际回报的差异。

**简单应用场景：**
Q学习的TD目标。

---

### 公式 (14.37)：状态-行动奖励转移 / State-Action Reward Transition

$$
G(\tau_{s=a}=a, s'=a') = r(s, a') + \gamma G(\tau_{s=a}=a')
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(\tau_{s=a}=a, s'=a')$ | zhuǎnhuàn huíbào | 转换回报 |
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $G(\tau_{s=a}=a')$ | xiàyībù huíbào | 下一个回报 |

**一句话解释：**
当前奖励加折扣的未来回报。

**简单应用场景：**
贝尔曼方程的递归结构。

---

### 公式 (14.38)：状态转移贝尔曼方程 / State Transition Bellman Equation

$$
\approx r(s, a') + \gamma Q^*(s', a')
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $Q^*(s', a')$ | xiàyībù zuì yōu Q zhí | 下一状态最优Q值 |

**一句话解释：**
用现在奖励和下一状态的最优Q值近似。

**简单应用场景：**
DQN算法中的TD目标。

---

### 公式 (14.39)：Q学习更新规则 / Q Learning Update Rule

$$
\hat{Q}^*(s, a) - \hat{Q}(s, a) + \alpha\left(r(s, a') + \gamma Q^*(s', a') - \hat{Q}(s, a)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{Q}^*(s, a)$ | zuì yōu Q zhí jìngsì | 最优Q值估计 |
| $\hat{Q}(s, a)$ | dāngqián Q zhí | 当前Q值 |
| $\alpha$ | ālfa | 学习率 |
| $r(s, a')$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |

**一句话解释：**
按TD误差和学习率更新Q值估计。

**简单应用场景：**
基础Q学习算法。

---

### 公式 (14.40)：Q学习 / Q-Learning

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\left(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q(s, a)$ | zhuàngtài-xíngdòng jiàzhí | Q值表 |
| $\alpha$ | ālfa | 学习率 |
| $r$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $\max_{a'} Q(s', a')$ | xiàyībù zuìdà Q zhí | 下一状态最大Q值 |

**一句话解释：**
离线策略算法，学习最优Q值而不依赖当前策略。

**简单应用场景：**
经典的表格型强化学习算法。

---

### 公式 (14.41)：深度Q网络 / Deep Q Network

$$
Q_\phi(s, a) \approx Q^\pi(s, a)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Q_\phi(s, a)$ | shēndù Q wǎngluò | 深度Q网络 |
| $\phi$ | fài | 网络参数 |
| $Q^\pi(s, a)$ | zhēnshí Q zhí | 真实Q值 |

**一句话解释：**
用神经网络参数化Q函数。

**简单应用场景：**
DQN算法的核心。

---

### 公式 (14.42)：深度Q网络矩阵形式 / Deep Q Network Matrix Form

$$
\mathbf{Q}_\phi(s) = \begin{pmatrix}
Q_\phi(s, a_1) \\
\vdots \\
Q_\phi(s, a_M)
\end{pmatrix} \approx \begin{pmatrix}
Q^\pi(s, a_1) \\
\vdots \\
Q^\pi(s, a_M)
\end{pmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{Q}_\phi(s)$ | shēndù Q wǎngluò de shūchū xiàng liàng | 深度Q网络的输出向量 |
| $\phi$ | fài | 网络参数 |
| $M$ | xíngdòng zǒngshù | 行动总数 |

**一句话解释：**
输出所有可能行动的Q值向量。

**简单应用场景：**
DQN网络输出的结构。

---

### 公式 (14.43)：损失函数 / Loss Function

$$
\mathcal{L}(s, a, s'|\phi) = \left(r + \gamma \max_{a'} Q_\phi(s', a') - Q_\phi(s, a)\right)^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{L}$ | sŭnshī hánshù | 损失函数 |
| $r$ | jiǎnglì | 奖励 |
| $\gamma$ | zhéhuà yīnzǐ | 折扣因子 |
| $\phi$ | fài | 网络参数 |
| $\max_{a'} Q_\phi(s', a')$ | xiàyībù zuìdà Q zhí | 下一状态最大Q值 |

**一句话解释：**
衡量预测Q值与TD目标之间的均方误差。

**简单应用场景：**
DQN的训练目标函数。

---

### 公式 (14.44)：策略梯度目标函数导数 / Policy Gradient Objective Derivative

$$
\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial}{\partial \theta} \int p_\theta(\tau)G(\tau)d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial J(\theta)}{\partial \theta}$ | mùbiāo hánshù duìyú cānshù de偏导 | 目标函数关于参数的偏导 |
| $J(\theta)$ | mùbiāo hánshù | 目标函数 |
| $p_\theta(\tau)$ | guǐjī gàilǜ | 轨迹分布 |
| $G(\tau)$ | huíbào | 回报 |

**一句话解释：**
求目标函数对参数的梯度。

**简单应用场景：**
策略梯度方法的基础。

---

### 公式 (14.45)：交换求导和积分 / Swapping Derivative and Integral

$$
= \int \frac{\partial}{\partial \theta} p_\theta(\tau) G(\tau) d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \theta} p_\theta(\tau)$ | guǐjī gàilǜ duìyú cānshù de偏导 | 轨迹分布关于参数的偏导 |
| $G(\tau)$ | huíbào | 回报 |

**一句话解释：**
应用莱布尼茨积分法则。

**简单应用场景：**
推导策略梯度的技巧。

---

### 公式 (14.46)：似然比技巧 / Likelihood Ratio Trick

$$
= \int p_\theta(\tau) \left(\frac{1}{p_\theta(\tau)} \frac{\partial}{\partial \theta} p_\theta(\tau)\right) G(\tau) d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_\theta(\tau)$ | guǐjī gàilǜ | 轨迹分布 |
| $\frac{\partial}{\partial \theta} p_\theta(\tau)$ | guǐjī gàilǜ de偏导 | 轨迹分布的偏导 |
| $G(\tau)$ | huíbào | 回报 |

**一句话解释：**
插入$p_\theta(\tau)$的倒数使其成为期望形式。

**简单应用场景：**
将求导变换成期望形式。

---

### 公式 (14.47)：对数似然梯度 / Log Likelihood Gradient

$$
= \int p_\theta(\tau) \left(\frac{\partial}{\partial \theta} \log p_\theta(\tau)\right) G(\tau) d\tau
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \theta} \log p_\theta(\tau)$ | duìshù jīnghuà de偏导 | 对数轨迹分布的偏导 |
| $p_\theta(\tau)$ | guǐjī gàilǜ | 轨迹分布 |
| $G(\tau)$ | huíbào | 回报 |

**一句话解释：**
使用对数求导技巧简化表达。

**简单应用场景：**
策略梯度推导的关键步骤。

---

### 公式 (14.48)：策略梯度期望形式 / Policy Gradient Expected Form

$$
= E_{\tau \sim p_\theta(\tau)}\left[\frac{\partial}{\partial \theta} \log p_\theta(\tau) G(\tau)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $E_{\tau \sim p_\theta(\tau)}$ | duìyú guǐjī de期望 | 关于轨迹的期望 |
| $\frac{\partial}{\partial \theta} \log p_\theta(\tau)$ | duìshù guǐjī gàilǜ de偏导 | 对数轨迹分布的偏导 |
| $G(\tau)$ | huíbào | 回报 |

**一句话解释：**
策略梯度为对数似然与回报乘积的期望。

**简单应用场景：**
REINFORCE算法的核心公式。

---

### 公式 (14.49)：对数轨迹分布展开 / Log Trajectory Distribution Expansion

$$
\frac{\partial}{\partial \theta} \log p_\theta(\tau) = \frac{\partial}{\partial \theta} \log \left(p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial}{\partial \theta} \log p_\theta(\tau)$ | duìshù guǐjī gàilǜ de偏导 | 对数轨迹分布的偏导 |
| $p(s_0)$ | chūshǐ zhuàngtài gàilǜ | 初始状态分布 |
| $\pi_\theta(a_t\|s_t)$ | zhèngcè | 参数化策略 |
| $p(s_{t+1}\|s_t, a_t)$ | huánjìng dònghuà | 环境动力学 |

**一句话解释：**
展开轨迹分布为初始分布、策略和环境的乘积。

**简单应用场景：**
推导策略梯度的中间步骤。

---

### 公式 (14.50)：对数乘积求导 / Logarithm Product Differentiation

$$
= \frac{\partial}{\partial \theta} \left(\log p(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log p(s_{t+1}|s_t, a_t)\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p(s_0)$ | duìshù chūshǐ gàilǜ | 对数初始分布 |
| $\log \pi_\theta(a_t\|s_t)$ | duìshù zhèngcè | 对数策略 |
| $\log p(s_{t+1}\|s_t, a_t)$ | duìshù huánjìng dònghuà | 对数环境模型 |

**一句话解释：**
将乘积的对数转换为求和的对数。

**简单应用场景：**
简化对数似然梯度的计算。

---

### 公式 (14.51)：策略梯度项 / Policy Gradient Term

$$
= \sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_\theta(a_t|s_t)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sum_{t=0}^{T-1}$ | shíjiān qiúhé | 时间求和 |
| $\frac{\partial}{\partial \theta} \log \pi_\theta(a_t\|s_t)$ | zhèngcè duìshù de偏导 | 策略对数的偏导 |
| $\pi_\theta(a_t\|s_t)$ | zhèngcè | 参数化策略 |

**一句话解释：**
对数轨迹分布的导数只包含策略项（环境模型和初始分布的导数为0）。

**简单应用场景：**
策略梯度不依赖于模型。

---

### 公式 (14.52)：策略梯度最终形式 / Policy Gradient Final Form

$$
\frac{\partial J(\theta)}{\partial \theta} = E_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_\theta(a_t|s_t) G(\tau)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\frac{\partial J(\theta)}{\partial \theta}$ | mùbiāo hánshù de偏导 | 目标函数的偏导 |
| $E_{\tau \sim p_\theta(\tau)}$ | duìyú guǐjī de期望 | 关于轨迹的期望 |
| $\frac{\partial}{\partial \theta} \log \pi_\theta(a_t\|s_t)$ | zhèngcè duìshù de偏导 | 策略对数的偏导 |
| $G(\tau)$ | huíbào | 回报 |

**一句话解释：**
策略梯度等于对数策略梯度与回报乘积的期望。

**简单应用场景：**
REINFORCE和策略梯度算法的基础。

---

### 公式 (14.53)：贝尔曼回报的计算 / Bellman Return Computation

$$
= E_{\tau \sim p_\theta(\tau)}\left[\left(\sum_{t'=0}^{T-1} \frac{\partial}{\partial \theta} \log \pi_\theta(a_t|s_t)\right) \left(G(\tau_t) + \gamma' G(\tau, T)\right)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $G(\tau_t)$ | shíkè t zhī hòu de huíbào | 时刻t之后的回报 |
| $\gamma'$ | zhéhuà yīnzǐ | 折扣因子 |
| $G(\tau, T)$ | chōngfèn zuìhòu de huíbào | 充分最后的回报 |

**一句话解释：**
分解回报为当前奖励和未来回报。

**简单应用场景：**
优化值函数的基线。

---

### 公式 (14.54)：轨迹回报求和 / Trajectory Return Summation

$$
= E_{\tau \sim p_\theta(\tau)}\left[\sum_{t'=0}^{T-1} \left(\frac{\partial}{\partial \theta} \log \pi_\theta(a_t|s_t) \right) \gamma' G(\tau_t, T)\right]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sum_{t'=0}^{T-1}$ | shíjiān qiúhé | 时间求和 |
| $\frac{\partial}{\partial \theta} \log \pi_\theta(a_t\|s_t)$ | zhèngcè duìshù de偏导 | 策略对数的偏导 |
| $G(\tau_t, T)$ | chōngfèn zuìhòu de huíbào | 充分最后的回报 |

**一句话解释：**
对所有时间步求和策略梯度与后续回报的乘积。

**简单应用场景：**
策略梯度算法的样本估计。

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

### 公式 (15.3)：条件概率分解 / Conditional Probability Chain Rule

$$
p(\boldsymbol{x}_{1:T}) = \prod_{t=1}^{T} p(\boldsymbol{x}_t | \boldsymbol{x}_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\prod$ | chéngji | 乘积 |
| $p(\boldsymbol{x}_t \mid \boldsymbol{x}_{1:(t-1)})$ | tiáojiàn gài lǜ | 给定前面所有变量时当前变量的条件概率 |
| $\boldsymbol{x}_{1:(t-1)}$ | xiàng liàng | 时间1到t-1的所有观测值 |

**一句话解释：**
任何序列的联合概率都可以分解为每一步的条件概率的乘积。

**简单应用场景：**
神经网络生成序列时，依次预测每个位置的输出。

---

### 公式 (15.5)：序列概率最大似然估计 / Maximum Likelihood for Sequences

$$
\max_{\theta} \sum_{n=1}^{N} \log p_{\theta}(\boldsymbol{x}_{1:T_n}) = \max_{\theta} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \log p_{\theta}(\boldsymbol{x}_t^{(n)} | \boldsymbol{x}_{1:(t-1)}^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | zuì dà huà | 最大化 |
| $\theta$ | cānshù | 模型参数 |
| $N$ | yàngběn shù | 训练样本数 |
| $T_n$ | dì n ge yàngběn de zhǎng dù | 第n个样本的序列长度 |
| $\log$ | duìshù | 对数 |

**一句话解释：**
序列模型的训练目标是最大化所有样本中所有时刻的条件概率的对数之和。

**简单应用场景：**
训练语言模型时，最大化模型对训练数据的对数似然。

---

### 公式 (15.6)：序列生成 / Sequence Generation

$$
\boldsymbol{x}_t \sim p_{\theta}(\boldsymbol{x} | \boldsymbol{x}_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sim$ | cóng...zhōng cāiyàng | 从分布中采样 |
| $p_{\theta}$ | gài lǜ fúnshù | 由参数θ定义的概率分布 |

**一句话解释：**
在已知前面所有输出的情况下，从模型预测的条件概率分布中采样生成下一个输出。

**简单应用场景：**
文本生成模型中，根据已生成的词依次采样生成下一个词。

---

### 公式 (15.7)：贪心搜索 / Greedy Search

$$
\boldsymbol{x}_t = \arg \max_{\boldsymbol{x} \in V} p_{\theta}(\boldsymbol{x} | \boldsymbol{x}_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\arg \max$ | zuì dà huà de cānshù | 使函数取得最大值的参数 |
| $V$ | cíhuì | 词表/输出空间 |

**一句话解释：**
不采样而是贪心地选择每一步概率最高的输出。

**简单应用场景：**
机器翻译中，每步选择概率最高的词而不进行采样。

---

### 公式 (15.8)：束搜索约束 / Beam Search Constraint

$$
\prod_{t=1}^{T} \max_{\boldsymbol{x}_t \in V} p_{\theta}(\boldsymbol{x} | \boldsymbol{x}_{1:(t-1)}) \leq \max_{\boldsymbol{x}_{1:T} \in V^T} \prod_{t=1}^{T} p_{\theta}(\boldsymbol{x} | \boldsymbol{x}_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\leq$ | xiǎo yú huò děngyú | 小于等于 |
| $V^T$ | cíhuì de T cì fāng | 词表的T次方 |

**一句话解释：**
贪心搜索的最优解得分小于等于全局最优解。

**简单应用场景：**
证明贪心搜索的次优性，说明束搜索的必要性。

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

### 公式 (15.22)：条件概率参数化 / Conditional Probability Parameterization

$$
p_\theta(x_t | x_{1:(t-1)}) = f_\theta_s(x_{1:(t-1)}, \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p_\theta$ | tiáojiàn gài lǜ | 条件概率 |
| $f_\theta_s$ | softmax hánshù | Softmax函数 |

**一句话解释：**
深度模型的条件概率通过Softmax将网络输出转换为概率分布。

**简单应用场景：**
神经网络语言模型的输出层设计。

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

### 公式 (15.26)：隐藏层计算 / Hidden Layer Computation

$$
\boldsymbol{h}_t = g(W\theta_\phi, \theta)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g$ | jīhuó hánshù | 激活函数 |
| $W$ | quántè jǔzhèn | 权重矩阵 |
| $\theta_\phi$ | yǐnhuàn céng cānshù | 隐藏层参数 |

**一句话解释：**
隐藏层通过线性变换加激活函数对输入进行非线性变换。

**简单应用场景：**
神经网络中的特征变换。

---

### 公式 (15.27)：跳层连接 / Skip Connection

$$
\boldsymbol{h}_t = \boldsymbol{h}' \oplus g(\boldsymbol{h}'\theta_\phi)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\oplus$ | liánjiē | 拼接 |
| $g(\cdot)$ | jīhuó hánshù | 激活函数 |

**一句话解释：**
跳层连接将原始输入与隐藏层输出拼接，增加模型的表达能力。

**简单应用场景：**
改进神经网络模型的性能。

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

### 公式 (15.31)：对数似然训练目标 / Log-Likelihood Training Objective

$$
\log p_\theta(\boldsymbol{x}_{1:T}) = \sum_{t=1}^{T} \log p_\theta(\boldsymbol{x}_t | \boldsymbol{x}_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\log p_\theta$ | duìshù gài lǜ | 对数概率 |
| $T$ | xùliè zhǎng dù | 序列长度 |

**一句话解释：**
序列的对数似然是所有时刻条件概率对数的求和。

**简单应用场景：**
序列模型的训练目标函数。

---

### 公式 (15.32)：梯度更新 / Gradient Update

$$
\theta \leftarrow \theta + \alpha \frac{\partial \log p_\theta(\boldsymbol{x}_{1:T})}{\partial \theta}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\leftarrow$ | gèngxīn | 更新 |
| $\alpha$ | xuéxí lǜ | 学习率 |
| $\frac{\partial}{\partial \theta}$ | duì cānshù qiúbiàodǎo | 对参数求导 |

**一句话解释：**
使用随机梯度上升法沿着对数似然的梯度方向更新参数。

**简单应用场景：**
神经网络的反向传播训练。

---

### 公式 (15.33)：困惑度 / Perplexity

$$
2^{H(p)} = 2^{-\sum_{x \in X} p(x) \log_2 p(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(p)$ | xìnxī ēntǔpiē | 信息熵 |
| $p(x)$ | gài lǜ | 概率 |
| $\log_2$ | èr jìn duìshù | 二进制对数 |

**一句话解释：**
困惑度是2的熵次方，衡量概率分布的不确定性。

**简单应用场景：**
评估语言模型的性能，困惑度越小越好。

---

### 公式 (15.34)：样本困惑度 / Sample Perplexity

$$
2^{H(\hat{p}_r, p_\theta)} = 2^{-\frac{1}{N} \sum_{n=1}^{N} \log_2 p_\theta(\boldsymbol{x}^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{p}_r$ | jīngyàn fēnbù | 经验分布 |
| $N$ | yàngběn shù | 样本数 |
| $p_\theta(\boldsymbol{x}^{(n)})$ | móxíng duì dì n ge yàngběn de gài lǜ | 模型对第n个样本的概率 |

**一句话解释：**
样本困惑度是模型在测试集上平均对数概率的负指数。

**简单应用场景：**
在实际数据上测试模型的困惑度。

---

### 公式 (15.35)：独立性假设 / Independence Assumption

$$
\prod_{n=1}^{N} p_\theta(\boldsymbol{x}_{1:T_n}^{(n)}) = \prod_{n=1}^{T_n} \prod_{t=1}^{T_n} p_\theta(\boldsymbol{x}_t^{(n)} | \boldsymbol{x}_{1:(t-1)}^{(n)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\prod_{n=1}^{N}$ | duì N ge yàngběn qiúji | 对N个样本求积 |
| $T_n$ | dì n ge yàngběn de zhǎng dù | 第n个样本的长度 |

**一句话解释：**
多个独立样本的联合概率是各个样本概率的乘积。

**简单应用场景：**
训练数据包含多个独立序列时的概率计算。

---

### 公式 (15.36)：困惑度评估定义 / Perplexity Evaluation Definition

$$
\text{PPL}(\theta) = 2^{-\frac{1}{T} \sum_{n=1}^{N} \log_2 p_\theta(\boldsymbol{x}_{1:T_n}^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{PPL}$ | kùnhuò dù | 困惑度 |
| $T$ | zǒng císhù | 总词数 |

**一句话解释：**
困惑度定义为平均每个词的负对数概率的指数。

**简单应用场景：**
评估语言模型在完整测试集上的性能。

---

### 公式 (15.37)：总体困惑度分解 / Overall Perplexity Decomposition

$$
= 2^{-\frac{1}{T} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \log_2 p_\theta(\boldsymbol{x}_t^{(n)} | \boldsymbol{x}_{1:(t-1)}^{(n)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\sum_{n, t}$ | duì suǒyǒu yàngběn hé shíkè qiúhé | 对所有样本和时刻求和 |

**一句话解释：**
总困惑度可以分解为所有样本所有时刻的条件概率对数平均。

**简单应用场景：**
具体计算困惑度的中间步骤。

---

### 公式 (15.38)：困惑度指数形式 / Perplexity Exponential Form

$$
= \left( \prod_{n=1}^{N} \prod_{t=1}^{T_n} p_\theta(\boldsymbol{x}_t^{(n)} | \boldsymbol{x}_{1:(t-1)}^{(n)}) \right)^{-1/T}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $^{-1/T}$ | de -1/T cì fāng | 的-1/T次方 |

**一句话解释：**
困惑度是所有条件概率乘积的倒数的T次根。

**简单应用场景：**
困惑度的几何解释。

---

### 公式 (15.39)：BLEU算法 / BLEU Algorithm

$$
P_N(\boldsymbol{x}) = \frac{\sum_{\omega \in W} \min(c_\omega(\boldsymbol{x}), \max_{\omega(k)} c_\omega(\boldsymbol{x}^{(k)}))}{\sum_{\omega \in W} c_\omega(\boldsymbol{x})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P_N$ | N yuán jīngzhǔn | N元精准率 |
| $\omega$ | N yuán zǔ hé | N元组合 |
| $c_\omega(\boldsymbol{x})$ | yuán zǔ c zài jiēpàn xù liè x zhōng chūxiàn cì shù | 元组在判断序列中出现次数 |

**一句话解释：**
BLEU精准率为修剪后的N元匹配数与总N元数的比值。

**简单应用场景：**
机器翻译质量评估中的N元精准率计算。

---

### 公式 (15.40)：长度惩罚 / Length Penalty

$$
b(\boldsymbol{x}) = \begin{cases}
1 & \text{if } l_s > l_r \\
\exp(1 - l_s/l_r) & \text{if } l_s \leq l_r
\end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $l_s$ | duànjù zhǎng dù | 短句长度 |
| $l_r$ | cānzhào jù zhǎng dù | 参照句长度 |
| $\exp$ | zhĭ shù hánshù | 指数函数 |

**一句话解释：**
对长度过短的候选翻译进行惩罚，鼓励更完整的翻译。

**简单应用场景：**
防止序列生成模型生成过短的输出。

---

### 公式 (15.41)：BLEU-N计算 / BLEU-N Calculation

$$
\text{BLEU-N}(\boldsymbol{x}) = b(\boldsymbol{x}) \times \exp\left( \sum_{N'=1}^{N} \alpha_{N'} \log P_{N'} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha_{N'}$ | dì N' yuán de quán zhòng | 第N'元的权重 |
| $\log P_{N'}$ | duìshù jīngzhǔn | 对数精准率 |

**一句话解释：**
BLEU分数为加权几何平均的N元精准率与长度惩罚的乘积。

**简单应用场景：**
综合评估机器翻译的质量。

---

### 公式 (15.42)：ROUGE算法 / ROUGE Algorithm

$$
\text{ROUGE-N}(\boldsymbol{x}) = \frac{\sum_{k=1}^{K} \sum_{\omega \in W} \min(c_\omega(\boldsymbol{x}), c_\omega(\boldsymbol{x}^{(k)}))}{\sum_{k=1}^{K} \sum_{\omega \in W} c_\omega(\boldsymbol{x}^{(k)})}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $K$ | cānzhào jù shù | 参照句数 |
| $c_\omega(\boldsymbol{x}^{(k)})$ | dì k ge cānzhào jù zhōng yuán zǔ chūxiàn cì shù | 第k个参照句中元组出现次数 |

**一句话解释：**
ROUGE分数为候选文本与参照文本之间的N元重叠比例。

**简单应用场景：**
文本摘要和机器翻译评估。

---

### 公式 (15.43)：轨迹定义 / Trajectory Definition

$$
\tau = \{s_1, a_1, s_2, a_2, \ldots, s_T, a_T\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\tau$ | guǐjī | 轨迹 |
| $s_t$ | zhuàngtài | 状态 |
| $a_t$ | dòngzuò | 动作 |

**一句话解释：**
轨迹是一系列交替出现的状态和动作序列。

**简单应用场景：**
强化学习中表示智能体与环境的交互历史。

---

### 公式 (15.44)：轨迹概率 / Trajectory Probability

$$
p_\theta(\tau) = \prod_{t=1}^{T} \pi_\theta(a_t = x_t | s_t = x_{1:(t-1)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\pi_\theta$ | zhèngcè | 策略 |
| $a_t$ | shíkè t de dòngzuò | 时刻t的动作 |
| $s_t$ | shíkè t de zhuàngtài | 时刻t的状态 |

**一句话解释：**
轨迹的概率为所有时刻在给定状态下采取该动作的概率的乘积。

**简单应用场景：**
强化学习中计算策略生成轨迹的概率。

---

### 公式 (15.45)：强化学习目标 / Reinforcement Learning Objective

$$
\mathcal{J}(\theta) = E_{\tau \sim p_\theta(\tau)}[G(\tau)]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathcal{J}(\theta)$ | mùbiāo hánshù | 目标函数 |
| $E[\cdot]$ | qīwàng | 期望 |
| $G(\tau)$ | huíbào | 回报 |

**一句话解释：**
强化学习的目标是最大化策略生成的轨迹的期望回报。

**简单应用场景：**
策略梯度方法的优化目标。

---

### 公式 (15.46)：策略梯度目标展开 / Policy Gradient Objective Expansion

$$
= E_{x_{1:T} \sim p_\theta(\tau)}[G(x_{1:T})]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $x_{1:T}$ | xù liè | 序列 |
| $G(x_{1:T})$ | xù liè huíbào | 序列回报 |

**一句话解释：**
期望回报等于所有可能序列的回报与其概率的加权平均。

**简单应用场景：**
强化学习和序列生成的统一框架。

---

### 公式 (15.47)：逆序列概率模型 / Inverse Sequence Probability Model

$$
p_\theta(\hat{s}_t | \hat{h}_t) = \text{softmax}\left( s(\hat{s}_t, \hat{h}_t; \theta) \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\hat{s}_t$ | yùliàn zhuàngtài | 预测状态 |
| $\hat{h}_t$ | yǐnhuàn xiàng liàng | 隐藏向量 |
| $s(\cdot)$ | fēnshù hánshù | 分数函数 |

**一句话解释：**
计算给定隐藏状态时各状态的条件概率分布。

**简单应用场景：**
序列生成模型的输出层设计。

---

该文档包含了从Chapter 15 (序列生成模型)中提取的所有主要公式，从基础的序列概率模型到高级的强化学习方法。每个公式都包含了清晰的符号表、单句解释和实际应用场景。


## 提取完成

本文档提取了《神经网络与深度学习》第15章(第341-382页)中的全部43条编号公式，按照指定格式进行了整理和说明。每个公式都包括LaTeX表达式、符号表、中文解释和应用场景，便于学习和参考。

**关键符号对照：**
- θ = xī tǎ (西塔)
- ω = ōu mǐ gā (欧米伽)
- σ = xī gé mǎ (西格玛)
- φ = fài (斐)



## 附录A：线性代数

### 公式 (A.1)：向量 / Vector

$$
\boldsymbol{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | xiàng liàng | 向量 |
| $x_i$ | dì $i$ gè zuò biāo | 第 $i$ 个坐标 |
| $n$ | wēi shù | 维数 |

**一句话解释：**
向量是一个有序的数值集合，每个分量按照特定顺序排列。

**简单应用场景：**
在神经网络中，输入数据、隐藏层激活等都可表示为向量。

---

### 公式 (A.2)：向量加法 / Vector Addition

$$
\boldsymbol{x} + \boldsymbol{y} = \begin{pmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_n + y_n \end{pmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}$ | xiàng liàng $x$ | 向量 $x$ |
| $\boldsymbol{y}$ | xiàng liàng $y$ | 向量 $y$ |
| $+$ | jiāfǎ | 加法 |

**一句话解释：**
两个向量相加就是对应分量逐一相加。

**简单应用场景：**
在深度学习中用于计算残差连接（residual connections）。

---

### 公式 (A.3)：向量数乘 / Scalar Multiplication

$$
\alpha \boldsymbol{x} = \begin{pmatrix} \alpha x_1 \\ \alpha x_2 \\ \vdots \\ \alpha x_n \end{pmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\alpha$ | shù liàng | 数量（标量） |
| $\boldsymbol{x}$ | xiàng liàng | 向量 |

**一句话解释：**
用一个标量乘以向量，就是将向量的每个分量都乘以这个标量。

**简单应用场景：**
在学习率调整和特征缩放中使用。

---

### 公式 (A.4)：向量内积 / Inner Product (Dot Product)

$$
\boldsymbol{x}^T \boldsymbol{y} = \sum_{i=1}^{n} x_i y_i
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{x}^T$ | xiàng liàng $x$ zhuǎnzhì | 向量 $x$ 转置 |
| $\boldsymbol{y}$ | xiàng liàng $y$ | 向量 $y$ |
| $\sum$ | qiúhé | 求和 |

**一句话解释：**
内积是两个向量对应分量乘积的和，反映两向量的相似度和夹角关系。

**简单应用场景：**
计算神经元的加权和、余弦相似度、损失函数等。

---

### 公式 (A.5)：向量范数 / Vector Norm

$$
\|\boldsymbol{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\|\boldsymbol{x}\|_p$ | $\boldsymbol{x}$ de $p$ fànm | $\boldsymbol{x}$ 的 $p$ 范数 |
| $\|\cdot\|$ | fànmù | 范数 |
| $p$ | cì shù | 次数 |

**一句话解释：**
范数是衡量向量"大小"或"长度"的一种方式，常见的有 $L^1$、$L^2$ 范数。

**简单应用场景：**
正则化中的范数惩罚（如 $L^2$ 正则化）。

---

### 公式 (A.6)：矩阵 / Matrix

$$
\boldsymbol{A} = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn | 矩阵 |
| $a_{ij}$ | dì $i$ háng dì $j$ liè | 第 $i$ 行第 $j$ 列元素 |
| $m$ | háng shù | 行数 |
| $n$ | liè shù | 列数 |

**一句话解释：**
矩阵是一个二维数值数组，由行和列组成。

**简单应用场景：**
神经网络中的权重矩阵、特征矩阵等。

---

### 公式 (A.7)：矩阵加法 / Matrix Addition

$$
\boldsymbol{A} + \boldsymbol{B} = (a_{ij} + b_{ij})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn $A$ | 矩阵 $A$ |
| $\boldsymbol{B}$ | jǔ zhèn $B$ | 矩阵 $B$ |
| $+$ | jiāfǎ | 加法 |

**一句话解释：**
两个矩阵相加就是对应位置的元素逐一相加。

**简单应用场景：**
在图神经网络和矩阵分解中使用。

---

### 公式 (A.8)：矩阵乘法 / Matrix Multiplication

$$
(\boldsymbol{AB})_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn $A$ | 矩阵 $A$ |
| $\boldsymbol{B}$ | jǔ zhèn $B$ | 矩阵 $B$ |
| $\sum$ | qiúhé | 求和 |

**一句话解释：**
矩阵乘法中，结果矩阵的第 $(i,j)$ 个元素是第一个矩阵的第 $i$ 行与第二个矩阵的第 $j$ 列的内积。

**简单应用场景：**
前向传播中的输入与权重矩阵相乘。

---

### 公式 (A.9)：矩阵转置 / Matrix Transpose

$$
(\boldsymbol{A}^T)_{ij} = a_{ji}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}^T$ | jǔ zhèn $A$ zhuǎnzhì | 矩阵 $A$ 转置 |

**一句话解释：**
矩阵转置就是将矩阵的行和列互换。

**简单应用场景：**
在梯度计算和损失函数推导中频繁使用。

---

### 公式 (A.10)：矩阵行列式 / Matrix Determinant

$$
\det(\boldsymbol{A}) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\det(\boldsymbol{A})$ | xíngliè shì | 行列式 |
| $S_n$ | $n$ jiè duìchèn zǔ | $n$ 阶对称组 |
| $\sigma$ | pái liè | 排列 |

**一句话解释：**
行列式是方阵的一个标量值，反映矩阵的某些性质如可逆性。

**简单应用场景：**
检验矩阵是否可逆，在概率和信息论中的应用。

---

### 公式 (A.11)：矩阵迹 / Matrix Trace

$$
\text{tr}(\boldsymbol{A}) = \sum_{i=1}^{n} a_{ii}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{tr}(\boldsymbol{A})$ | jǔ zhèn de jī | 矩阵的迹 |
| $a_{ii}$ | duìjiǎo xiàn yuánsù | 对角线元素 |

**一句话解释：**
迹是矩阵对角线元素的和。

**简单应用场景：**
在深度学习中用于计算方差、归一化等。

---

### 公式 (A.12)：矩阵秩 / Matrix Rank

$$
\text{rank}(\boldsymbol{A}) = \text{矩阵线性无关行（列）的最大个数}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{rank}(\boldsymbol{A})$ | jǔ zhèn de zhì | 矩阵的秩 |

**一句话解释：**
秩是矩阵中线性无关行（或列）向量的最大个数。

**简单应用场景：**
在降维、特征选择和模型复杂度分析中使用。

---

### 公式 (A.13)：特征值与特征向量 / Eigenvalue and Eigenvector

$$
\boldsymbol{A}\boldsymbol{v} = \lambda \boldsymbol{v}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn | 矩阵 |
| $\boldsymbol{v}$ | tèzhēng xiàng liàng | 特征向量 |
| $\lambda$ | tèzhēng zhí | 特征值 |

**一句话解释：**
矩阵与特征向量相乘等于特征值乘以该特征向量。

**简单应用场景：**
主成分分析（PCA）、矩阵分解等。

---

### 公式 (A.14)：特征分解 / Eigendecomposition

$$
\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{-1}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn | 矩阵 |
| $\boldsymbol{Q}$ | tèzhēng xiàng liàng zuò chéng de jǔ zhèn | 特征向量矩阵 |
| $\boldsymbol{\Lambda}$ | duìjiǎo tèzhēng zhí jǔ zhèn | 对角特征值矩阵 |

**一句话解释：**
将一个方阵分解为特征向量矩阵、特征值对角矩阵和其逆矩阵的乘积。

**简单应用场景：**
矩阵幂运算、矩阵指数等计算。

---

### 公式 (A.15)：奇异值分解 / Singular Value Decomposition (SVD)

$$
\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\boldsymbol{A}$ | jǔ zhèn | 矩阵 |
| $\boldsymbol{U}$ | zuǒ tèzhēng xiàng liàng jǔ zhèn | 左特征向量矩阵 |
| $\boldsymbol{\Sigma}$ | duìjiǎo奇异值 jǔ zhèn | 对角奇异值矩阵 |
| $\boldsymbol{V}^T$ | yòu tèzhēng xiàng liàng jǔ zhèn zhuǎnzhì | 右特征向量矩阵转置 |

**一句话解释：**
将任意矩阵分解为两个正交矩阵和一个奇异值对角矩阵的乘积。

**简单应用场景：**
矩阵压缩、降维、噪声消除等。

---

## 附录B：概率论

### 公式 (B.1)：样本空间与事件 / Sample Space and Event

$$
\Omega = \{(i,j) | i = 1,\ldots,6, j = 1,\ldots,6\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Omega$ | yàng běn kōng jiān | 样本空间 |
| $(i,j)$ | jī běn shìjiàn | 基本事件 |

**一句话解释：**
样本空间是所有可能结果的集合，事件是样本空间的子集。

**简单应用场景：**
概率模型的基础定义和事件概率计算。

---

### 公式 (B.2)：概率的定义 / Definition of Probability

$$
P(A) = \frac{\text{有利结果数}}{\text{总结果数}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(A)$ | shìjiàn $A$ de gài lǜ | 事件 $A$ 的概率 |
| $A$ | shìjiàn | 事件 |

**一句话解释：**
概率是衡量事件发生可能性的度量，范围在 0 到 1 之间。

**简单应用场景：**
计算分类问题中各类别的概率。

---

### 公式 (B.3)：离散概率分布 / Discrete Probability Distribution

$$
P(X = x_n) = p(x_n), \quad \forall n \in \{1,\ldots,N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(X = x_n)$ | $X$ děng yú $x_n$ de gài lǜ | $X$ 等于 $x_n$ 的概率 |
| $x_n$ | gài lǜ zhí | 概率值 |
| $p(x_n)$ | gài lǜ zhì liàng hánshù | 概率质量函数 |

**一句话解释：**
离散概率分布描述了随机变量取各个特定值的概率。

**简单应用场景：**
分类问题的输出概率分布、词汇的概率分布等。

---

### 公式 (B.4)：伯努利分布 / Bernoulli Distribution

$$
p(x) = \mu^x(1-\mu)^{(1-x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ zhì liàng hánshù | 概率质量函数 |
| $\mu$ | chéng gōng gài lǜ | 成功概率 |
| $x$ | qìquán biàozhǔn | 取值标准 |

**一句话解释：**
伯努利分布描述单次独立二元试验的结果。

**简单应用场景：**
逻辑回归、二分类问题的概率模型。

---

### 公式 (B.5)：二项分布 / Binomial Distribution

$$
P(X = k) = \binom{N}{k}\mu^k(1-\mu)^{N-k}, \quad k = 0,\ldots,N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(X = k)$ | $X$ děng yú $k$ de gài lǜ | $X$ 等于 $k$ 的概率 |
| $\binom{N}{k}$ | $N$ yǔ $k$ de zuǐhéshù | $N$ 与 $k$ 的组合数 |
| $N$ | shìyàn cì shù | 试验次数 |
| $k$ | chéng gōng cì shù | 成功次数 |

**一句话解释：**
二项分布描述独立重复进行 $N$ 次伯努利试验的成功次数分布。

**简单应用场景：**
多次独立试验的成功次数统计。

---

### 公式 (B.6)：均匀分布 / Uniform Distribution

$$
p(x) = \begin{cases} \frac{1}{b-a}, & a \leq x \leq b \\ 0, & x < a \text{ 或 } x > b \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ mìdù hánshù | 概率密度函数 |
| $a$ | xià jiè | 下界 |
| $b$ | shàng jiè | 上界 |

**一句话解释：**
均匀分布在给定区间内所有点的概率密度都相等。

**简单应用场景：**
初始化神经网络权重、生成随机样本等。

---

### 公式 (B.7)：正态分布 / Normal Distribution

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ mìdù hánshù | 概率密度函数 |
| $\mu$ | jūnzhí | 均值 |
| $\sigma$ | biāozhǔn chāyù | 标准差 |

**一句话解释：**
正态分布是自然界中最常见的连续分布，呈钟形曲线。

**简单应用场景：**
特征归一化、参数初始化、贝叶斯模型中的先验分布。

---

### 公式 (B.8)：条件概率 / Conditional Probability

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(A\|B)$ | gěi dìng $B$ tiáojiàn xià $A$ de gài lǜ | 给定 $B$ 条件下 $A$ 的概率 |
| $\cap$ | jiāojí | 交集 |

**一句话解释：**
条件概率是在已知另一事件发生的条件下，某事件发生的概率。

**简单应用场景：**
贝叶斯分类器、条件独立性分析。

---

### 公式 (B.9)：全概率公式 / Law of Total Probability

$$
P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(A)$ | shìjiàn $A$ de gài lǜ | 事件 $A$ 的概率 |
| $P(A\|B_i)$ | tiáojiàn gài lǜ | 条件概率 |
| $P(B_i)$ | shìjiàn $B_i$ de gài lǜ | 事件 $B_i$ 的概率 |

**一句话解释：**
通过完全分解的所有情况的条件概率加权求和来计算某事件的总概率。

**简单应用场景：**
隐变量模型、期望最大化（EM）算法。

---

### 公式 (B.10)：贝叶斯公式 / Bayes' Theorem

$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(B\|A)$ | hòuyàn gài lǜ | 后验概率 |
| $P(A\|B)$ | ránrán gài lǜ | 然然概率 |
| $P(B)$ | xiānjiàn gài lǜ | 先验概率 |
| $P(A)$ | zhèngù gài lǜ | 证据概率 |

**一句话解释：**
贝叶斯公式用于根据新证据更新事件的概率。

**简单应用场景：**
贝叶斯分类器、垃圾邮件过滤、医学诊断等。

---

### 公式 (B.11)：随机变量期望 / Expectation of Random Variable

$$
\mathbb{E}[X] = \sum_{i=1}^{n} x_i p(x_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbb{E}[X]$ | $X$ de qīwàng zhí | $X$ 的期望值 |
| $x_i$ | qìquán | 取值 |
| $p(x_i)$ | gài lǜ | 概率 |

**一句话解释：**
期望是随机变量取值与其对应概率乘积的总和，代表随机变量的平均水平。

**简单应用场景：**
损失函数的期望、模型评估中的评分期望。

---

### 公式 (B.12)：随机变量方差 / Variance of Random Variable

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Var}(X)$ | $X$ de fāngchà | $X$ 的方差 |
| $\mathbb{E}[X^2]$ | $X^2$ de qīwàng zhí | $X^2$ 的期望值 |
| $\mathbb{E}[X]$ | $X$ de qīwàng zhí | $X$ 的期望值 |

**一句话解释：**
方差衡量随机变量与其期望的偏离程度，反映数据的离散程度。

**简单应用场景：**
计算特征标准化、评估模型预测的稳定性。

---

### 公式 (B.13)：协方差 / Covariance

$$
\text{Cov}(X,Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\text{Cov}(X,Y)$ | $X$ hé $Y$ de xiéfāngchà | $X$ 和 $Y$ 的协方差 |
| $\mathbb{E}[\cdot]$ | qīwàng zhí | 期望值 |

**一句话解释：**
协方差衡量两个随机变量的线性相关程度。

**简单应用场景：**
特征之间的相关性分析、协方差矩阵的计算。

---

### 公式 (B.14)：高斯分布的参数表示 / Gaussian Distribution Parameters

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $X$ | suíjī biàoliàng | 随机变量 |
| $\sim$ | fúcóng | 服从 |
| $\mathcal{N}$ | zhèng tài fēnbù | 正态分布 |
| $\mu$ | jūnzhí | 均值 |
| $\sigma^2$ | fāngchà | 方差 |

**一句话解释：**
用均值和方差两个参数描述正态分布的性质。

**简单应用场景：**
生成式模型（如变分自编码器）中的分布表示。

---

### 公式 (B.15)：联合分布与独立性 / Joint Distribution and Independence

$$
p(x,y) = p(x)p(y) \quad \text{若 } X \perp Y
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x,y)$ | liánhé gài lǜ | 联合概率 |
| $p(x)$ | $X$ de gài lǜ | $X$ 的概率 |
| $p(y)$ | $Y$ de gài lǜ | $Y$ 的概率 |
| $\perp$ | dúlì | 独立 |

**一句话解释：**
若两个随机变量独立，其联合分布等于各自边际分布的乘积。

**简单应用场景：**
朴素贝叶斯分类器中的条件独立性假设。

---

### 公式 (B.16)：信息熵 / Information Entropy

$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X)$ | $X$ de xìnxī shàngtǐ | $X$ 的信息熵 |
| $p(x_i)$ | $x_i$ de gài lǜ | $x_i$ 的概率 |
| $\log$ | duìshù | 对数 |

**一句话解释：**
熵衡量随机变量的不确定性或信息量，值越大表示不确定性越高。

**简单应用场景：**
决策树的划分标准、信息增益的计算。

---

## 附录E：信息论

### 公式 (E.1)：自信息 / Self-Information

$$
I(x) = -\log p(x) = \log \frac{1}{p(x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(x)$ | $x$ de zìxìnxī | $x$ 的自信息 |
| $p(x)$ | $x$ de gài lǜ | $x$ 的概率 |
| $\log$ | duìshù | 对数 |

**一句话解释：**
自信息衡量特定事件发生所包含的信息量，概率越小信息量越大。

**简单应用场景：**
编码理论、惊奇度的量化。

---

### 公式 (E.2)：信息熵 / Information Entropy (Entropy)

$$
H(X) = \mathbb{E}[I(X)] = -\sum_{i=1}^{n} p(x_i) \log p(x_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X)$ | $X$ de xìnxī shàngtǐ | $X$ 的信息熵 |
| $\mathbb{E}[\cdot]$ | qīwàng zhí | 期望值 |
| $p(x_i)$ | $x_i$ de gài lǜ | $x_i$ 的概率 |

**一句话解释：**
信息熵是所有可能事件自信息的期望值，反映源的平均信息量。

**简单应用场景：**
衡量数据的混乱程度、决策树的信息增益。

---

### 公式 (E.3)：联合熵 / Joint Entropy

$$
H(X,Y) = -\sum_{i=1}^{m} \sum_{j=1}^{n} p(x_i, y_j) \log p(x_i, y_j)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(X,Y)$ | $X$ hé $Y$ de liánhé shàngtǐ | $X$ 和 $Y$ 的联合熵 |
| $p(x_i, y_j)$ | liánhé gài lǜ | 联合概率 |

**一句话解释：**
联合熵衡量两个随机变量组成系统的平均信息量。

**简单应用场景：**
多变量系统的复杂性分析。

---

### 公式 (E.4)：条件熵 / Conditional Entropy

$$
H(Y|X) = -\sum_{i=1}^{m} p(x_i) \sum_{j=1}^{n} p(y_j|x_i) \log p(y_j|x_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(Y\|X)$ | gěi dìng $X$ tiáojiàn xià $Y$ de shàngtǐ | 给定 $X$ 条件下 $Y$ 的熵 |
| $p(x_i)$ | $X$ de biānyuán gài lǜ | $X$ 的边缘概率 |
| $p(y_j\|x_i)$ | tiáojiàn gài lǜ | 条件概率 |

**一句话解释：**
条件熵衡量已知 $X$ 情况下，$Y$ 的剩余不确定性。

**简单应用场景：**
特征选择、信息增益的定义。

---

### 公式 (E.5)：信息增益 / Information Gain

$$
IG(Y|X) = H(Y) - H(Y|X)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $IG(Y\|X)$ | $X$ duì $Y$ de xìnxī zēngyì | $X$ 对 $Y$ 的信息增益 |
| $H(Y)$ | $Y$ de shàngtǐ | $Y$ 的熵 |
| $H(Y\|X)$ | tiáojiàn shàngtǐ | 条件熵 |

**一句话解释：**
信息增益衡量特征对目标变量不确定性的降低程度。

**简单应用场景：**
决策树中的特征选择标准。

---

### 公式 (E.6)：相互信息 / Mutual Information

$$
I(X;Y) = H(X) + H(Y) - H(X,Y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(X;Y)$ | $X$ hé $Y$ de xiānghù xìnxī | $X$ 和 $Y$ 的相互信息 |
| $H(X)$ | $X$ de shàngtǐ | $X$ 的熵 |
| $H(Y)$ | $Y$ de shàngtǐ | $Y$ 的熵 |
| $H(X,Y)$ | liánhé shàngtǐ | 联合熵 |

**一句话解释：**
相互信息衡量两个随机变量之间的相关程度或共享信息量。

**简单应用场景：**
特征与标签的依赖性分析、聚类评估。

---

### 公式 (E.7)：交叉熵 / Cross Entropy

$$
H(p,q) = -\sum_{i=1}^{n} p(x_i) \log q(x_i)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(p,q)$ | jiāocha shàngtǐ | 交叉熵 |
| $p(x_i)$ | zhēnshí gài lǜ | 真实概率 |
| $q(x_i)$ | yùcè gài lǜ | 预测概率 |

**一句话解释：**
交叉熵衡量真实分布与预测分布的差异，常用作损失函数。

**简单应用场景：**
分类问题的损失函数、神经网络训练的优化目标。

---

### 公式 (E.8)：KL散度 / Kullback-Leibler Divergence

$$
D_{KL}(p||q) = \sum_{i=1}^{n} p(x_i) \log \frac{p(x_i)}{q(x_i)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_{KL}(p\|\|q)$ | KL sǎndù | KL散度 |
| $p(x_i)$ | zhēnshí gài lǜ | 真实概率 |
| $q(x_i)$ | yùcè gài lǜ | 预测概率 |

**一句话解释：**
KL散度衡量两个概率分布之间的不对称差异，值越小两分布越接近。

**简单应用场景：**
变分推断、生成模型的优化目标。

---

### 公式 (E.9)：Wasserstein距离 / Wasserstein Distance

$$
W(p,q) = \min_{\pi} \mathbb{E}_{(x,y) \sim \pi} [\|x - y\|]
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $W(p,q)$ | Wasserstein jùlí | Wasserstein距离 |
| $\pi$ | liánhé fēnbù | 联合分布 |
| $\mathbb{E}[\cdot]$ | qīwàng zhí | 期望值 |

**一句话解释：**
Wasserstein距离衡量两个概率分布之间的"运输成本"，对分布形状变化更敏感。

**简单应用场景：**
生成对抗网络（GAN）的训练目标。

---

### 公式 (E.10)：JS散度 / Jensen-Shannon Divergence

$$
D_{JS}(p||q) = \frac{1}{2} D_{KL}(p||m) + \frac{1}{2} D_{KL}(q||m)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D_{JS}(p\|\|q)$ | JS sǎndù | JS散度 |
| $m$ | pínghuá fēnbù | 平滑分布 |
| $m = \frac{1}{2}(p+q)$ | pínghuá ù | 平滑分布 |

**一句话解释：**
JS散度是KL散度的对称化版本，值域在0到1之间，更适合衡量分布差异。

**简单应用场景：**
评估不同分布的相似度。

---

### 公式 (E.11)：编码长度与熵 / Code Length and Entropy

$$
L = \sum_{i=1}^{n} p(x_i) \ell_i \geq H(X)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $L$ | pínghuá biānmǎ chángdù | 平均编码长度 |
| $p(x_i)$ | gài lǜ | 概率 |
| $\ell_i$ | $x_i$ de biānmǎ chángdù | $x_i$ 的编码长度 |
| $H(X)$ | xìnxī shàngtǐ | 信息熵 |

**一句话解释：**
任何编码方案的平均编码长度不能小于信息熵，熵是理论下界。

**简单应用场景：**
无损数据压缩、哈夫曼编码。

---

### 公式 (E.12)：容量 / Channel Capacity

$$
C = \max_{p(x)} I(X;Y)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $C$ | tōngdào rónngliàng | 通道容量 |
| $p(x)$ | shūruò gài lǜ fēnbù | 输入概率分布 |
| $I(X;Y)$ | xiānghù xìnxī | 相互信息 |

**一句话解释：**
信道容量是通过某信道可靠传输信息的最大速率。

**简单应用场景：**
通信系统的性能分析。

---

### 公式 (E.13)：差分熵 / Differential Entropy

$$
h(X) = -\int p(x) \log p(x) dx
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h(X)$ | chāfēn shàngtǐ | 差分熵 |
| $p(x)$ | gài lǜ mìdù hánshù | 概率密度函数 |
| $\int$ | jīfēn | 积分 |

**一句话解释：**
差分熵是离散熵在连续分布上的推广，衡量连续随机变量的不确定性。

**简单应用场景：**
连续变量系统的复杂性分析。

---

### 公式 (E.14)：信息率 / Information Rate

$$
R = \frac{H(X)}{T}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $R$ | xìnxī lǜ | 信息率 |
| $H(X)$ | xìnxī shàngtǐ | 信息熵 |
| $T$ | shíjiān | 时间 |

**一句话解释：**
信息率是单位时间内传输的平均信息量。

**简单应用场景：**
通信系统吞吐量的衡量。

---

### 公式 (E.15)：冗余度 / Redundancy

$$
R_{red} = 1 - \frac{H(X)}{H_{max}}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $R_{red}$ | róngyúdù | 冗余度 |
| $H(X)$ | xìnxī shàngtǐ | 信息熵 |
| $H_{max}$ | zuìdà shàngtǐ | 最大熵 |

**一句话解释：**
冗余度衡量信息中无用信息的比例，值越大表示越多冗余。

**简单应用场景：**
数据压缩率的评估、特征冗余的分析。

---

### 公式 (E.16)：费诺不等式 / Fano's Inequality

$$
H(Y|X) \leq H_b(P_e) + P_e \log(M-1)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H(Y\|X)$ | tiáojiàn shàngtǐ | 条件熵 |
| $H_b(P_e)$ | èrjìn shàngtǐ | 二进熵 |
| $P_e$ | cuòwù gài lǜ | 错误概率 |
| $M$ | lèi biāozhǔn shù | 类标准数 |

**一句话解释：**
费诺不等式给出了信息论中错误概率与熵的关系下界。

**简单应用场景：**
分类问题的理论性能分析。

---

### 公式 (E.17)：RÃ©nyi熵 / Rényi Entropy

$$
H_\alpha(X) = \frac{1}{1-\alpha} \log \sum_{i=1}^{n} p(x_i)^\alpha
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $H_\alpha(X)$ | $\alpha$ jiè Rényi shàngtǐ | $\alpha$ 阶Rényi熵 |
| $\alpha$ | jiè shù | 阶数 |
| $p(x_i)$ | gài lǜ | 概率 |

**一句话解释：**
Rényi熵是Shannon熵的推广，通过参数 $\alpha$ 控制对不同概率的权重。

**简单应用场景：**
广义信息量的度量、强化学习中的探索策略。

---

### 公式 (E.18)：Tsallis熵 / Tsallis Entropy

$$
S_q(X) = \frac{1}{q-1} \left(1 - \sum_{i=1}^{n} p(x_i)^q\right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $S_q(X)$ | Tsallis shàngtǐ | Tsallis熵 |
| $q$ | cānshù | 参数 |
| $p(x_i)$ | gài lǜ | 概率 |

**一句话解释：**
Tsallis熵是非加性熵，适用于非平衡系统的复杂性度量。

**简单应用场景：**
复杂系统的热力学分析。

---

### 公式 (E.19)：信息分歧 / Divergence Measure

$$
D(p||q) \geq 0, \quad D(p||q) = 0 \iff p = q
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $D(p\|\|q)$ | sǎndù liànghuà | 散度量化 |
| $p$ | zhēnshí fēnbù | 真实分布 |
| $q$ | mònǐ fēnbù | 模拟分布 |

**一句话解释：**
散度是衡量两个分布差异的非对称度量，只有相同时才为0。

**简单应用场景：**
模型优化、分布匹配评估。

---

### 公式 (E.20)：极值分布与信息论 / Extreme Value Distribution

$$
P(X \leq x) = \exp(-e^{-\lambda(x-\mu)})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(X \leq x)$ | zīnéng hánshù | 累积分布 |
| $\lambda$ | xíngzhuàng cānshù | 形状参数 |
| $\mu$ | wèizhì cānshù | 位置参数 |

**一句话解释：**
极值分布描述随机变量极值（最大值或最小值）的分布规律。

**简单应用场景：**
极端事件分析、风险评估。

---

### 公式 (E.21)：信息失真权衡 / Information-Distortion Trade-off

$$
I(X;Z) = I(X;Y) - I(Z;Y|X)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $I(X;Z)$ | $X$ hé $Z$ de xiānghù xìnxī | $X$ 和 $Z$ 的相互信息 |
| $I(X;Y)$ | $X$ hé $Y$ de xiānghù xìnxī | $X$ 和 $Y$ 的相互信息 |
| $I(Z;Y\|X)$ | tiáojiàn xiānghù xìnxī | 条件相互信息 |

**一句话解释：**
信息失真权衡描述压缩过程中保留有用信息与去除无用信息的权衡关系。

**简单应用场景：**
信息瓶颈方法（Information Bottleneck）、模型压缩。

---


---

## 附录C：拉格朗日对偶性与KKT条件 / Appendix C: Lagrange Duality and KKT Conditions

### 公式 (C.12)：等式约束优化问题 / Equality Constrained Optimization Problem

$$
\Lambda(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{m=1}^{M} \lambda_m h_m(\mathbf{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Lambda(\mathbf{x}, \boldsymbol{\lambda})$ | Lā gē lǎng rì hánshù | Lagrange function |
| $f(\mathbf{x})$ | mùbiāo hánshù | Objective function |
| $\lambda_m$ | lā mó dá | Lagrange multiplier |
| $h_m(\mathbf{x})$ | yuē shù hánshù | Constraint function |
| $M$ | yuē shù shù liàng | Number of equality constraints |

**一句话解释：**
Lagrange function combines the objective function with equality constraints using Lagrange multipliers to enable constrained optimization.

**简单应用场景：**
Used in constrained optimization where we need to find critical points of functions subject to equality constraints.

---

### 公式 (C.13)：KKT 条件 - 梯度条件 / KKT Conditions - Gradient Condition

$$
\nabla f(\mathbf{x}) + \sum_{m=1}^{M} \lambda_m \nabla h_m(\mathbf{x}) = \mathbf{0}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla f(\mathbf{x})$ | f de tī dù xiàng liàng | Gradient of objective function |
| $\lambda_m$ | lā mó dá | Lagrange multiplier |
| $\nabla h_m(\mathbf{x})$ | h_m de tī dù xiàng liàng | Gradient of constraint |

**一句话解释：**
The gradient of the objective function must be a linear combination of the gradients of the constraint functions at the optimal point.

**简单应用场景：**
Necessary condition for optimality in constrained optimization problems.

---

### 公式 (C.14)：约束条件 / Constraint Condition

$$
h_m(\mathbf{x}) = 0, \quad \forall m = 1, \cdots, M
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_m(\mathbf{x})$ | yuē shù hánshù | Constraint function |
| $M$ | yuē shù shù liàng | Number of constraints |

**一句话解释：**
All equality constraints must be satisfied at the optimal solution.

**简单应用场景：**
Feasibility requirement for any candidate solution in constrained optimization.

---

### 公式 (C.15)：不等式约束优化问题 / Inequality Constrained Optimization Problem

$$
\Lambda(\mathbf{x}, \mathbf{a}, \mathbf{b}) = f(\mathbf{x}) + \sum_{m=1}^{M} a_m h_m(\mathbf{x}) + \sum_{n=1}^{N} b_n g_n(\mathbf{x})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Lambda(\mathbf{x}, \mathbf{a}, \mathbf{b})$ | Lā gē Lǎng Rì hánshù | Lagrange function with dual variables |
| $f(\mathbf{x})$ | mùbiāo hánshù | Objective function |
| $a_m$ | děng shì yuē shù duì yìng de duó Lā mó dá | Multiplier for equality constraint |
| $h_m(\mathbf{x})$ | děng shì yuē shù | Equality constraint |
| $b_n$ | bù děng shì yuē shù duì yìng de duó Lā mó dá | Multiplier for inequality constraint |
| $g_n(\mathbf{x})$ | bù děng shì yuē shù | Inequality constraint |

**一句话解释：**
Lagrange function that incorporates both equality and inequality constraints for constrained optimization.

**简单应用场景：**
Foundation for solving optimization problems with mixed equality and inequality constraints.

---

### 公式 (C.16)：原问题 / Primal Problem

$$
\min_{\mathbf{x}} \max_{\mathbf{a}, \mathbf{b}} \Lambda(\mathbf{x}, \mathbf{a}, \mathbf{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\min_{\mathbf{x}}$ | duì x qiú zuì xiǎo zhí | Minimize over x |
| $\max_{\mathbf{a}, \mathbf{b}}$ | duì a he b qiú zuì dà zhí | Maximize over a and b |
| $\Lambda(\mathbf{x}, \mathbf{a}, \mathbf{b})$ | Lā gē Lǎng Rì hánshù | Lagrange function |

**一句话解释：**
The min-max formulation of the primal problem that finds the optimal point by minimizing over the primary variables while maximizing over the dual variables.

**简单应用场景：**
Formulates constrained optimization as a min-max problem suitable for dual analysis.

---

### 公式 (C.17)：可行性约束 / Feasibility Constraint

$$
\mathbf{b} \geq \mathbf{0}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{b}$ | duó Lā mó dá xiàng liàng | Dual variable vector |

**一句话解释：**
Dual variables associated with inequality constraints must be non-negative.

**简单应用场景：**
Ensures feasibility condition for the dual problem in constrained optimization.

---

### 公式 (C.18)：对偶函数 / Dual Function

$$
\Gamma(\mathbf{a}, \mathbf{b}) = \inf_{\mathbf{x} \in D} \Lambda(\mathbf{x}, \mathbf{a}, \mathbf{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Gamma(\mathbf{a}, \mathbf{b})$ | duó Lā gē Lǎng Rì hánshù | Dual function |
| $\inf$ | xià quèjiè | Infimum |
| $D$ | kě xíng yù | Feasible region |

**一句话解释：**
The dual function is the infimum of the Lagrangian over the feasible region, providing a lower bound on the optimal value.

**简单应用场景：**
Used to establish lower bounds and analyze duality gaps in optimization problems.

---

### 公式 (C.19)：对偶问题下界 / Dual Problem Lower Bound

$$
\Gamma(\mathbf{a}, \mathbf{b}) = \inf_{\mathbf{x} \in D} \Lambda(\mathbf{x}, \mathbf{a}, \mathbf{b}) \leq \Lambda(\mathbf{x}^*, \mathbf{a}, \mathbf{b}) \leq f(\mathbf{x}^*)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Gamma(\mathbf{a}, \mathbf{b})$ | duó hánshù | Dual function |
| $\mathbf{x}^*$ | zuì yōu jiěduì yìng de x | Optimal point |
| $f(\mathbf{x}^*)$ | zuì yōu zhí | Optimal value |

**一句话解释：**
The dual function provides a lower bound on the primal optimal value, establishing the duality relationship.

**简单应用场景：**
Fundamental relationship in duality theory for proving optimality and analyzing performance.

---

### 公式 (C.20)：弱对偶性 / Weak Duality

$$
\Gamma(\mathbf{a}, \mathbf{b}) \leq p^*
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Gamma(\mathbf{a}, \mathbf{b})$ | duó hánshù | Dual function |
| $p^*$ | yuán wèn zuì yōu zhí | Primal optimal value |

**一句话解释：**
The dual function value is always less than or equal to the primal optimal value, regardless of dual variable values.

**简单应用场景：**
Establishes that any feasible dual solution provides a lower bound on the primal problem.

---

### 公式 (C.21)：对偶问题 / Dual Problem

$$
\max_{\mathbf{a}, \mathbf{b}} \Gamma(\mathbf{a}, \mathbf{b})
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\max$ | qiú zuì dà zhí | Maximize |
| $\Gamma(\mathbf{a}, \mathbf{b})$ | duó hánshù | Dual function |

**一句话解释：**
The dual problem maximizes the dual function over all feasible dual variables.

**简单应用场景：**
Formulation of the dual optimization problem used in duality analysis.

---

### 公式 (C.22)：对偶可行性 / Dual Feasibility

$$
\mathbf{b} \geq \mathbf{0}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\mathbf{b}$ | duó biàoliàng | Dual variable |

**一句话解释：**
Dual variables must be non-negative for feasibility in the dual problem.

**简单应用场景：**
Constraint ensuring the dual solution is feasible.

---

### 公式 (C.23)：强对偶性 - KKT 条件梯度 / Strong Duality - KKT Gradient Condition

$$
\nabla f(\mathbf{x}^*) + \sum_{m=1}^{M} a_m \nabla h_m(\mathbf{x}^*) + \sum_{n=1}^{N} b_n^* \nabla g_n(\mathbf{x}^*) = \mathbf{0}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\nabla f(\mathbf{x}^*)$ | f zài zuì yōu diǎn de tī dù | Gradient of objective at optimum |
| $a_m$ | děng shì yuē shù duó biàoliàng | Equality constraint multiplier |
| $\nabla h_m(\mathbf{x}^*)$ | h_m zài zuì yōu diǎn de tī dù | Gradient of equality constraint |
| $b_n^*$ | bù děng shì yuē shù duó biàoliàng | Inequality constraint multiplier |
| $\nabla g_n(\mathbf{x}^*)$ | g_n zài zuì yōu diǎn de tī dù | Gradient of inequality constraint |

**一句话解释：**
At the optimum, the gradient of the objective function is a non-negative combination of constraint gradients.

**简单应用场景：**
Critical KKT condition for checking optimality in constrained optimization.

---

### 公式 (C.24)：等式约束条件 / Equality Constraint Condition

$$
h_m(\mathbf{x}^*) = 0, \quad m = 1, \cdots, M
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $h_m(\mathbf{x}^*)$ | m ge děng shì yuē shù | m-th equality constraint |
| $M$ | yuē shù zǒng shù | Total number of equality constraints |

**一句话解释：**
All equality constraints must be satisfied at the optimal solution.

**简单应用场景：**
Feasibility requirement for the primal problem.

---

### 公式 (C.25)：不等式约束条件 / Inequality Constraint Condition

$$
g_n(\mathbf{x}^*) \leq 0, \quad n = 1, \cdots, N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $g_n(\mathbf{x}^*)$ | n ge bù děng shì yuē shù | n-th inequality constraint |
| $N$ | bù děng shì yuē shù zǒng shù | Total number of inequality constraints |

**一句话解释：**
All inequality constraints must be satisfied at the optimal solution.

**简单应用场景：**
Feasibility requirement ensuring inequality constraints are not violated.

---

### 公式 (C.26)：互补松弛条件 / Complementary Slackness Condition

$$
b_n^* g_n(\mathbf{x}^*) = 0, \quad n = 1, \cdots, N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b_n^*$ | n ge duó biàoliàng | n-th dual variable at optimum |
| $g_n(\mathbf{x}^*)$ | n ge bù děng shì yuē shù | n-th inequality constraint at optimum |

**一句话解释：**
Either the dual variable is zero or the constraint is active (tight) at optimality.

**简单应用场景：**
Determines which constraints are binding at the optimal solution.

---

### 公式 (C.27)：对偶变量非负性 / Dual Variable Non-negativity

$$
b_n^* \geq 0, \quad n = 1, \cdots, N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $b_n^*$ | n ge duó biàoliàng | n-th dual variable |

**一句话解释：**
Dual variables associated with inequality constraints must be non-negative.

**简单应用场景：**
Ensures feasibility of the dual solution in KKT conditions.

---

## 附录D：概率论 / Appendix D: Probability Theory

### 公式 (D.1)：样本空间 / Sample Space

$$
\Omega = \{(i, j) | i = 1, \ldots, 6, \, j = 1, \ldots, 6\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $\Omega$ | ōu mǐ gā | Sample space |
| $i$ | dì yī wèi | First die outcome |
| $j$ | dì èr wèi | Second die outcome |

**一句话解释：**
The sample space for rolling two dice consists of all ordered pairs of outcomes from 1 to 6.

**简单应用场景：**
Fundamental concept in probability theory for defining all possible outcomes of an experiment.

---

### 公式 (D.2)：离散随机变量 - 示例1 / Discrete Random Variable - Example 1

$$
X(i, j) := i + j, \quad x = 2, 3, \ldots, 12
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $X(i, j)$ | suí jī biàoliàng | Random variable |
| $i$ | dì yī wèi jié guǒ | First die result |
| $j$ | dì èr wèi jié guǒ | Second die result |
| $x$ | suí jī biàoliàng de qǔ zhí | Possible value |

**一句话解释：**
A random variable that maps dice outcomes to their sum, with possible values from 2 to 12.

**简单应用场景：**
Example of defining a random variable as a function of sample space outcomes.

---

### 公式 (D.3)：离散随机变量 - 示例2 / Discrete Random Variable - Example 2

$$
Y(i, j) := |i - j|, \quad y = 0, 1, 2, 3, 4, 5
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $Y(i, j)$ | suí jī biàoliàng | Random variable |
| $i, j$ | sè zǐ jié guǒ | Die outcomes |
| $y$ | suí jī biàoliàng de qǔ zhí | Possible value |

**一句话解释：**
A random variable that maps dice outcomes to the absolute difference between them.

**简单应用场景：**
Example of a function-based random variable definition.

---

### 公式 (D.4)：概率分布 / Probability Distribution

$$
P(X = x_n) = p(x_n), \quad \forall n \in \{1, \cdots, N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(X = x_n)$ | X děng yú x_n de gài lǜ | Probability that X equals x_n |
| $p(x_n)$ | gài lǜ zhí | Probability value |
| $N$ | kě néng jié guǒ de shù mù | Number of possible outcomes |

**一句话解释：**
The probability that a random variable takes a specific value.

**简单应用场景：**
Fundamental definition of probability distribution for discrete random variables.

---

### 公式 (D.5)：概率分布和为1 / Probability Distribution Sums to One

$$
\sum_{n=1}^{N} p(x_n) = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_n)$ | dì n ge jié guǒ de gài lǜ | Probability of n-th outcome |
| $N$ | kě néng jié guǒ de zǒng shù | Total number of outcomes |

**一句话解释：**
The sum of all probabilities for all possible outcomes equals one.

**简单应用场景：**
Fundamental axiom ensuring valid probability distributions.

---

### 公式 (D.6)：非负概率 / Non-negative Probability

$$
p(x_n) \geq 0, \quad \forall n \in \{1, \ldots, N\}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x_n)$ | dì n ge jié guǒ de gài lǜ | Probability value |

**一句话解释：**
Probabilities must be non-negative for all possible outcomes.

**简单应用场景：**
Fundamental axiom of probability theory.

---

### 公式 (D.7)：伯努利分布 / Bernoulli Distribution

$$
p(x) = \mu^x (1 - \mu)^{(1-x)}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ zhī liàng hánshù | Probability mass function |
| $\mu$ | chéng gōng gài lǜ | Success probability |
| $x$ | suí jī biàoliàng, qǔ zhí 0 huò 1 | Random variable, takes 0 or 1 |

**一句话解释：**
Probability distribution for a single binary outcome with success probability μ.

**简单应用场景：**
Models coin flips, binary classification outcomes, and binary experimental results.

---

### 公式 (D.8)：二项分布 / Binomial Distribution

$$
P(X = k) = \binom{N}{k} \mu^k (1 - \mu)^{N-k}, \quad k = 0, \ldots, N
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $P(X = k)$ | k ge chéng gōng de gài lǜ | Probability of k successes |
| $N$ | shìyàn cì shù | Number of trials |
| $\binom{N}{k}$ | N zhōng xuǎn k | Binomial coefficient |
| $\mu$ | dān cì chéng gōng gài lǜ | Success probability per trial |

**一句话解释：**
Probability distribution for the number of successes in N independent Bernoulli trials.

**简单应用场景：**
Models repeated binary experiments like multiple coin flips or success counts in repeated trials.

---

### 公式 (D.9)：概率密度函数积分为1 / Probability Density Function Integral

$$
\int_{-\infty}^{+\infty} p(x) dx = 1
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ mì dù hánshù | Probability density function |
| $dx$ | duì x de jī fēn | Integration differential |

**一句话解释：**
The integral of a probability density function over all values equals one.

**简单应用场景：**
Normalization condition for continuous probability distributions.

---

### 公式 (D.10)：概率密度非负性 / Probability Density Non-negativity

$$
p(x) \geq 0
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ mì dù | Probability density |

**一句话解释：**
Probability density must be non-negative for all values.

**简单应用场景：**
Fundamental property of probability density functions.

---

### 公式 (D.11)：均匀分布 / Uniform Distribution

$$
p(x) = \begin{cases} \frac{1}{b-a}, & a \leq x \leq b \\ 0, & x < a \text{ or } x > b \end{cases}
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ mì dù hánshù | Probability density function |
| $a$ | qū jiān xià jiè | Lower bound |
| $b$ | qū jiān shàng jiè | Upper bound |

**一句话解释：**
Continuous probability distribution with constant density over the interval [a, b].

**简单应用场景：**
Models scenarios with equal probability across a continuous range.

---

### 公式 (D.12)：正态分布 / Normal Distribution

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $p(x)$ | gài lǜ mì dù hánshù | Probability density function |
| $\mu$ | jūn zhí | Mean |
| $\sigma$ | xī gé mǎ | Standard deviation |
| $\sigma^2$ | fāng chā | Variance |

**一句话解释：**
Bell-shaped probability distribution characterized by mean μ and standard deviation σ, fundamental in statistics.

**简单应用场景：**
Models many natural phenomena and is central to statistical inference and machine learning.

---

### 公式 (D.13)：正态分布标记 / Normal Distribution Notation

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

**符号表：**

| 符号 | 汉语拼音读法（带声调） | 含义 |
|------|------------------|------|
| $X$ | suí jī biàoliàng | Random variable |
| $\sim$ | fúcóng | Follows distribution |
| $\mathcal{N}$ | zhèng tài fēn bù | Normal distribution |
| $\mu$ | jūn zhí | Mean parameter |
| $\sigma^2$ | fāng chā | Variance parameter |

**一句话解释：**
Notation indicating that random variable X follows a normal distribution with mean μ and variance σ².

**简单应用场景：**
Standard notation for specifying normal distributions in probability and statistics.

---

## 补充说明 / Additional Notes

本文档汇编了《神经网络与深度学习》附录C、D中的全部公式，包括：

- **附录C**：拉格朗日对偶性与KKT条件（共16个公式：C.12-C.27）
- **附录D**：概率论（共13个公式：D.1-D.13）

每个公式均包含：
1. 准确的LaTeX数学表达式
2. 符号含义的双语对照
3. 拼音发音标注（带声调）
4. 一句话概念解释
5. 实际应用场景示例

**注意**：附录A、B、E的内容未在提供的PDF页面截图中充分显示，本文档基于可见的附录C和D内容进行编译。

