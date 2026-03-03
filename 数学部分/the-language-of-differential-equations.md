# The Language of Differential Equations

## 微分方程的语言 (Wēi fēn fāng chéng de yǔ yán)

A trilingual reference guide for reading, writing, and pronouncing the symbols of differential equations.

---

## 1. Core Symbols — 核心符号

| Symbol | English | 中文 (Chinese) | Pinyin (四声) | What It Does |
|--------|---------|----------------|---------------|-------------|
| $d$ | "dee" | 微分 | wēi fēn | Denotes an infinitesimally small change in a variable |
| $\partial$ | "partial" / "del" | 偏微分 | piān wēi fēn | Denotes a partial derivative (change in one variable while others are held constant) |
| $\frac{dy}{dx}$ | "dee-y by dee-x" | y对x的导数 | y duì x de dǎo shù | The derivative of y with respect to x; the rate of change of y as x changes |
| $\frac{\partial y}{\partial x}$ | "partial y partial x" | y对x的偏导数 | y duì x de piān dǎo shù | The partial derivative of y with respect to x |
| $\frac{d^2y}{dx^2}$ | "dee-two-y by dee-x-squared" | y对x的二阶导数 | y duì x de èr jiē dǎo shù | The second derivative; the rate of change of the rate of change |
| $\Delta$ | "delta" | 差 / 增量 | chā / zēng liàng | A finite change or difference in a quantity |
| $\nabla$ | "nabla" / "del" | 梯度算子 | tī dù suàn zǐ | The gradient operator; points in the direction of steepest increase |
| $\nabla^2$ | "del squared" / "Laplacian" | 拉普拉斯算子 | Lā pǔ lā sī suàn zǐ | The Laplacian; sum of all unmixed second partial derivatives |

---

## 2. Derivative Notations — 导数记法

| Symbol | English | 中文 | Pinyin | What It Does |
|--------|---------|------|--------|-------------|
| $y'$ | "y prime" | y撇 | y piě | First derivative of y (Lagrange notation) |
| $y''$ | "y double prime" | y两撇 | y liǎng piě | Second derivative of y |
| $\dot{y}$ | "y dot" | y点 | y diǎn | First derivative with respect to time (Newton notation) |
| $\ddot{y}$ | "y double dot" | y两点 | y liǎng diǎn | Second derivative with respect to time |
| $Df$ | "D of f" | f的导数 | f de dǎo shù | Derivative operator applied to f (Euler notation) |
| $D^n f$ | "D-n of f" | f的n阶导数 | f de n jiē dǎo shù | The nth derivative of f |

---

## 3. Integral Symbols — 积分符号

| Symbol | English | 中文 | Pinyin | What It Does |
|--------|---------|------|--------|-------------|
| $\int$ | "integral" | 积分 | jī fēn | Finds the antiderivative; reverses differentiation |
| $\int_a^b$ | "integral from a to b" | 从a到b的定积分 | cóng a dào b de dìng jī fēn | Definite integral; computes accumulated value from a to b |
| $\iint$ | "double integral" | 二重积分 | èr chóng jī fēn | Integration over a 2D region |
| $\iiint$ | "triple integral" | 三重积分 | sān chóng jī fēn | Integration over a 3D region |
| $\oint$ | "contour integral" | 环路积分 | huán lù jī fēn | Integration along a closed curve |

---

## 4. Common Functions & Operators — 常见函数与算子

| Symbol | English | 中文 | Pinyin | What It Does |
|--------|---------|------|--------|-------------|
| $f(x)$ | "f of x" | f关于x的函数 | f guān yú x de hán shù | A function that maps input x to an output |
| $\lim$ | "limit" | 极限 | jí xiàn | The value a function approaches as the input approaches some value |
| $\sum$ | "sigma" / "sum" | 求和 | qiú hé | Summation of a sequence of terms |
| $\prod$ | "product" | 连乘 / 求积 | lián chéng / qiú jī | Product of a sequence of terms |
| $\infty$ | "infinity" | 无穷大 | wú qióng dà | A quantity without bound |
| $\equiv$ | "is identically equal to" | 恒等于 | héng děng yú | Identity; true for all values of the variable |
| $\approx$ | "approximately equal" | 约等于 | yuē děng yú | Approximately equal to |
| $\propto$ | "proportional to" | 正比于 | zhèng bǐ yú | Directly proportional to |

---

## 5. Equation Types — 方程类型

| Term | English | 中文 | Pinyin | Description |
|------|---------|------|--------|-------------|
| ODE | Ordinary Differential Equation | 常微分方程 | cháng wēi fēn fāng chéng | Contains derivatives with respect to only one independent variable |
| PDE | Partial Differential Equation | 偏微分方程 | piān wēi fēn fāng chéng | Contains partial derivatives with respect to multiple independent variables |
| IVP | Initial Value Problem | 初值问题 | chū zhí wèn tí | A DE with conditions specified at a single point |
| BVP | Boundary Value Problem | 边值问题 | biān zhí wèn tí | A DE with conditions specified at two or more points |

---

## 6. Solution & Behavior Terms — 解与特性术语

| Symbol / Term | English | 中文 | Pinyin | What It Means |
|---------------|---------|------|--------|--------------|
| $C$ | constant of integration | 积分常数 | jī fēn cháng shù | An arbitrary constant added when integrating |
| $\lambda$ | "lambda" / eigenvalue | 特征值 | tè zhēng zhí | Eigenvalue; determines behavior of solutions |
| $e^{x}$ | "e to the x" | e的x次方 | e de x cì fāng | The exponential function; fundamental solution to $y' = y$ |
| $\sin, \cos$ | "sine, cosine" | 正弦, 余弦 | zhèng xián, yú xián | Trigonometric functions; appear in oscillatory solutions |
| $\ln$ | "natural log" | 自然对数 | zì rán duì shù | The inverse of the exponential function |
| homogeneous | — | 齐次的 | qí cì de | All terms involve the unknown function or its derivatives |
| nonhomogeneous | — | 非齐次的 | fēi qí cì de | Contains a forcing function independent of the unknown |
| linear | — | 线性的 | xiàn xìng de | Unknown function and its derivatives appear to the first power only |
| nonlinear | — | 非线性的 | fēi xiàn xìng de | Contains powers, products, or other nonlinear terms of the unknown |
| particular solution | — | 特解 | tè jiě | One specific solution satisfying given conditions |
| general solution | — | 通解 | tōng jiě | Family of all solutions, includes arbitrary constants |

---

## 7. Greek Letters in DEs — 微分方程中的希腊字母

| Letter | Name (English) | 中文 | Pinyin | Common Use |
|--------|---------------|------|--------|-----------|
| $\alpha$ | alpha | 阿尔法 | ā ěr fǎ | Parameter / coefficient |
| $\beta$ | beta | 贝塔 | bèi tǎ | Parameter / coefficient |
| $\gamma$ | gamma | 伽马 | jiā mǎ | Damping coefficient |
| $\delta$ | delta | 德尔塔 | dé ěr tǎ | Small perturbation / Dirac delta function |
| $\epsilon$ | epsilon | 伊普西隆 | yī pǔ xī lóng | Small quantity for perturbation methods |
| $\lambda$ | lambda | 拉姆达 | lā mǔ dá | Eigenvalue |
| $\mu$ | mu | 缪 | miù | Integrating factor / coefficient |
| $\omega$ | omega | 欧米伽 | ōu mǐ jiā | Angular frequency |
| $\phi, \varphi$ | phi | 斐 | fěi | Phase angle / basis function |
| $\psi$ | psi | 普赛 | pǔ sài | Wave function |
| $\rho$ | rho | 柔 | róu | Density |
| $\sigma$ | sigma | 西格玛 | xī gé mǎ | Summation / stress |
| $\tau$ | tau | 陶 | táo | Time constant / delay |
| $\xi$ | xi | 克赛 | kè sài | Transformed variable |

---

## 8. Key Phrases for Reading Equations — 读方程的关键短语

| English Phrase | 中文 | Pinyin |
|---------------|------|--------|
| "The derivative of y with respect to x" | y对x求导 | y duì x qiú dǎo |
| "The second derivative" | 二阶导数 | èr jiē dǎo shù |
| "Integrate both sides" | 两边积分 | liǎng biān jī fēn |
| "Subject to the initial condition" | 满足初始条件 | mǎn zú chū shǐ tiáo jiàn |
| "Let u equal..." | 令u等于... | lìng u děng yú... |
| "By separation of variables" | 用分离变量法 | yòng fēn lí biàn liàng fǎ |
| "Apply the chain rule" | 应用链式法则 | yìng yòng liàn shì fǎ zé |
| "Converges to" | 收敛于 | shōu liǎn yú |
| "Diverges" | 发散 | fā sàn |
| "Tends to infinity" | 趋向无穷 | qū xiàng wú qióng |
| "Is bounded" | 有界 | yǒu jiè |
| "Satisfies the equation" | 满足方程 | mǎn zú fāng chéng |
| "Substitute into" | 代入 | dài rù |

---

## 9. Famous Equations — 著名方程

| Equation | English Name | 中文 | Pinyin |
|----------|-------------|------|--------|
| $\frac{dy}{dx} = ky$ | Exponential growth/decay | 指数增长/衰减方程 | zhǐ shù zēng zhǎng / shuāi jiǎn fāng chéng |
| $\frac{d^2y}{dx^2} + \omega^2 y = 0$ | Simple harmonic oscillator | 简谐振动方程 | jiǎn xié zhèn dòng fāng chéng |
| $\frac{\partial u}{\partial t} = k\frac{\partial^2 u}{\partial x^2}$ | Heat equation | 热传导方程 | rè chuán dǎo fāng chéng |
| $\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$ | Wave equation | 波动方程 | bō dòng fāng chéng |
| $\nabla^2 u = 0$ | Laplace's equation | 拉普拉斯方程 | Lā pǔ lā sī fāng chéng |

---

*Created as a study aid for bilingual learners of mathematics.*
*为双语数学学习者编写的学习参考。*
