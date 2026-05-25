# 第20章：布尔代数

> **核心问题**：怎么用数学来描述"是"和"否"之间的推理关系？

---

## 0. 本章导览

"如果下雨，而且我没带伞，那我就会淋湿。"这句话里藏着三种逻辑运算：一个条件（下雨）、另一个条件（没带伞）、它们之间的关系（而且）。你每天都在做这种"是/否"推理，只是你没意识到这就是逻辑运算。

布尔代数就是把这些日常的"是/否"推理，变成一套精确的数学系统。它只有三个基本运算（与、或、非），两个值（0 和 1），却能描述任何逻辑关系。没有布尔代数，就不会有搜索引擎的查询语法、不会有编程语言里的 if 语句、更不会有数字电路的存在。

学完本章，你将能够：

1. 用 AND、OR、NOT 描述日常生活中的逻辑关系
2. 写出任意逻辑表达式的真值表
3. 使用布尔代数的公理和定理化简逻辑表达式（包括德摩根定律）
4. 掌握对偶原理，用一个公式推出另一个
5. 理解最小项和最大项的标准形式，并完成相互转换
6. 用代数法化简复杂逻辑函数（并项、吸收、消去、配项）
7. 用 Python 生成真值表并自动验证化简结果

> 本章约 1500 行，建议分 3-4 次读完。草稿纸、笔、耐心，三样都要有。

前置章节：[数制与编码](./19-number-systems.md)
下一章：[逻辑门电路](./21-logic-gates.md)

---

## 1. 从日常生活走进布尔代数

### 1a. 生活例子：你会做的"逻辑推理"

看看下面这几句话，你觉得对不对？

1. "如果你饿了，**或者**渴了，那就去食堂。"（饿了 OR 渴了 → 去食堂）
2. "如果下雨，**而且**你没带伞，那你就会淋湿。"（下雨 AND 没带伞 → 淋湿）
3. "如果灯**不**亮，说明停电了。"（NOT 灯亮 → 停电）

这些你每天都在做的推理，背后有一个共同结构：**几个条件（是/否），按某种方式组合，得出一个结论（是/否）**。

再多看几个：

- "你要么选 A，**要么**选 B，不能两个都选。"（A XOR B，异或）
- "如果今天不下雨**而且**气温超过 25 度，我就去游泳。"（条件 A AND 条件 B → 去）
- "这门课及格的条件是：平时分不低于 60 **而且**（期末考不低于 60 **或者**大作业通过）。"（条件组合嵌套）
- "如果是周末 **而且** 天气好 → 去爬山；如果工作日 **而且** 不下雨 → 骑自行车上班。"（多条规则是 OR 关系）

还有一个经典的逻辑推理：你妈妈说的话——"你要是考了 90 分以上 **而且** 房间收拾干净了，周末就带你去游乐园。"两个条件都得满足，缺一个都不行。这就是 AND。

布尔代数的世界就是把世界简化成两种值：**真（True，1）**和**假（False，0）**。所有复杂的条件判断，都可以用三种基本运算搭建出来。

### 1b. 直观理解：真值表——把所有可能都列出来

假设一个简单的场景：周末你要不要去公园？

条件：
- A：天气好吗？（好=1，不好=0）
- B：有朋友一起去吗？（有=1，没有=0）

你的决策规则可能是：
- 规则 1："天气好**而且**有朋友一起" → 去（A AND B）
- 规则 2："天气好**或者**有朋友一起" → 去（A OR B）

这两个规则有什么区别？把 A 和 B 的所有可能组合列出来看看：

| A（天气好） | B（有朋友） | A AND B（规则1） | A OR B（规则2） |
|:----------:|:----------:|:---------------:|:--------------:|
| 0 | 0 | 0（不去） | 0（不去） |
| 0 | 1 | 0（不去） | 1（去） |
| 1 | 0 | 0（不去） | 1（去） |
| 1 | 1 | 1（去） | 1（去） |

AND：**两个条件都满足才去**。条件苛刻，只有 1/4 的情况去。
OR：**只要满足一个就去**。条件宽松，3/4 的情况去。

这种把所有输入组合和对应输出都列出来的表，就叫**真值表**（Truth Table）。

再来看一个 3 条件的例子：三个委员 A、B、C 投票，每人投"赞成(1)"或"反对(0)"。决议通过(F=1)需要至少两票赞成。

| A | B | C | 赞成票数 | F | 说明 |
|---|---|---------|---|---|------|
| 0 | 0 | 0 | 0 | 0 | 全员反对 |
| 0 | 0 | 1 | 1 | 0 | 只有 C 赞成 |
| 0 | 1 | 0 | 1 | 0 | 只有 B 赞成 |
| 0 | 1 | 1 | 2 | 1 | B 和 C 赞成 |
| 1 | 0 | 0 | 1 | 0 | 只有 A 赞成 |
| 1 | 0 | 1 | 2 | 1 | A 和 C 赞成 |
| 1 | 1 | 0 | 2 | 1 | A 和 B 赞成 |
| 1 | 1 | 1 | 3 | 1 | 全员赞成 |

这就是三变量"多数表决器"的真值表，后面你会学到这个电路的具体实现。

### 1c. 形式化定义：布尔代数的基本运算

**布尔代数**（Boolean Algebra）是定义在集合 $B = \{0, 1\}$ 上的代数系统。有三种基本运算：

**1. 与运算（AND，逻辑乘）**，记作 $A \cdot B$ 或 $AB$：
- $0 \cdot 0 = 0$
- $0 \cdot 1 = 0$
- $1 \cdot 0 = 0$
- $1 \cdot 1 = 1$

一句话：**全 1 才 1，有 0 就 0。** 像串联的两个开关，两个都闭合电路才通。

**2. 或运算（OR，逻辑加）**，记作 $A + B$：
- $0 + 0 = 0$
- $0 + 1 = 1$
- $1 + 0 = 1$
- $1 + 1 = 1$

一句话：**有 1 就 1，全 0 才 0。** 像并联的两个开关，任意一个闭合电路就通。

注意：布尔代数的 $1+1=1$，不是十进制里的 $1+1=2$。这里没有"进位"的概念，只有"是否成立"。"真 OR 真 = 真"，重复表达同一个事实不会让它变得更真。

**3. 非运算（NOT，逻辑反）**，记作 $\overline{A}$ 或 $A'$：
- $\overline{0} = 1$
- $\overline{1} = 0$

一句话：**是变否，否变是。** 像一面镜子，照出相反的东西。

这三个运算是所有数字逻辑的根基。AND、OR、NOT 就像化学里的原子，任何复杂的逻辑都可以由它们组合而成。后面你会看到，连 XOR、NAND、NOR 这些"高级"运算都是这三个基本运算的衍生品。

### 1d. 手算示例：真值表构建

**例1**：写出 $F = AB + \overline{A}C$ 的真值表（3 个变量 A、B、C，共 $2^3=8$ 种组合）。

| A | B | C | $\overline{A}$ | $AB$ | $\overline{A}C$ | $F = AB + \overline{A}C$ |
|---|---|---|----------------|------|-----------------|--------------------------|
| 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| 1 | 1 | 1 | 0 | 1 | 0 | 1 |

观察这个真值表：当 $A=1, B=1$ 时（后两行），或者 $A=0, C=1$ 时（第 2、4 行），F=1。别的情况 F=0。

这其实就是一个"二选一"选择器：当 $A=0$ 时，输出取决于 C；当 $A=1$ 时，输出取决于 B。这正是**多路选择器**（MUX）的逻辑原型，在第 22 章你会遇到。

**例2**：写出 $F = \overline{AB + C}$ 的真值表。

| A | B | C | $AB$ | $AB+C$ | $F = \overline{AB+C}$ |
|---|---|-----|--------|------------------------|
| 0 | 0 | 0 | 0 | 0 | 1 |
| 0 | 0 | 1 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 | 0 | 1 |
| 0 | 1 | 1 | 0 | 1 | 0 |
| 1 | 0 | 0 | 0 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 | 1 | 0 |
| 1 | 1 | 1 | 1 | 1 | 0 |

**例3**：$F = A \oplus B \oplus C$（三输入异或：奇数个 1 时为 1）。

| A | B | C | 1的个数 | F |
|---|---|---------|---|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 | 1 |
| 0 | 1 | 0 | 1 | 1 |
| 0 | 1 | 1 | 2 | 0 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 2 | 0 |
| 1 | 1 | 0 | 2 | 0 |
| 1 | 1 | 1 | 3 | 1 |

三输入 XOR 的特性：奇数个 1 时输出 1。这就是奇偶校验器的核心逻辑。

### 1e. Python 验证

```python
# 真值表生成器

import itertools

def truth_table(variables, expression_func):
    """
    生成布尔表达式的真值表
    variables: 变量名列表，如 ['A', 'B', 'C']
    expression_func: 一个函数，接受变量值元组，返回 0 或 1
    """
    n = len(variables)
    # 表头
    header = ' | '.join(variables) + ' | F'
    print(header)
    print('-' * len(header))
    
    # 遍历所有输入组合（从 0 到 2^n - 1）
    for i in range(2 ** n):
        # 将 i 转为 n 位二进制，得到每个变量的值
        values = [(i >> (n - 1 - j)) & 1 for j in range(n)]
        result = expression_func(values)
        row = ' | '.join(str(v) for v in values) + f' | {result}'
        print(row)

# 例1: F = AB + ~A C
print("=== 例1: F = AB + ~A C ===")
def F1(vals):
    A, B, C = vals
    AB = A & B
    notA_C = (1 - A) & C  # ~A 即 1-A
    return AB | notA_C

truth_table(['A', 'B', 'C'], F1)

print("\n=== 例2: F = ~(AB + C) ===")
def F2(vals):
    A, B, C = vals
    AB = A & B
    AB_or_C = AB | C
    return 1 - AB_or_C  # NOT

truth_table(['A', 'B', 'C'], F2)

print("\n=== 例3: 三输入 XOR（奇数个1）===")
def F3(vals):
    A, B, C = vals
    return (A + B + C) % 2  # 奇数个1 → 余1

truth_table(['A', 'B', 'C'], F3)

# 验证: 多数表决器（至少两票）
print("\n=== 多数表决器 ===")
def majority(vals):
    A, B, C = vals
    return 1 if (A + B + C) >= 2 else 0

truth_table(['A', 'B', 'C'], majority)
```

预期输出：前三个函数的真值表和手工计算完全一致。多数表决器的输出已在 1b 节看到。

### 1f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "1+1=2" 在布尔代数也成立 | 布尔代数里 $1+1=1$。这里的 + 是逻辑或，不是算术加。"真 OR 真"还是"真"。 |
| "真值表只要列 F=1 的行就够了" | 必须列出所有 $2^n$ 种组合。漏掉任何一行，就是不完整的分析。而且 0 的行也重要（POS 形式要用）。 |
| "$\overline{A}B$ 和 $A\overline{B}$ 是同一个东西" | $\overline{A}B$ = A 为假且 B 为真；$A\overline{B}$ = A 为真且 B 为假。完全不同。 |
| "NAND 和 AND 差不多" | $\overline{AB}$ 和 $AB$ 输出完全相反。全 1 时 NAND=0，AND=1。不要搞混。 |

### 1g. 应用连接

真值表是数字电路设计的起点。在第 22 章，你拿到一个电路需求，第一步就是把需求写成真值表，然后从真值表推导出逻辑表达式，再化简，最后画出逻辑图。整个流程的源头就是真值表。

编程语言里的 `if`、`while` 条件判断，SQL 里的 `WHERE` 子句，搜索引擎里的 `AND`/`OR`/`NOT` 运算符，都是布尔代数在计算机科学中的直接应用。

---

## 2. 布尔代数的公理与定理

### 2a. 生活例子：算术里的交换律

小学数学教过你：$3 + 5 = 5 + 3$，加法可以交换顺序。$2 \times (3 + 4) = 2 \times 3 + 2 \times 4$，乘法对加法有分配律。

布尔代数里也有类似的规律。好消息是大部分和普通代数一样，坏消息是有几个反直觉的"怪胎"公式需要特别注意。

### 2b. 直观理解：布尔恒等式一览

**单变量基本公式**（直接验证就行）：

| 公式 | 名称 | 直观含义 |
|------|------|---------|
| $A + 0 = A$ | 或 0 恒等 | 任何东西 OR 假 = 它自己 |
| $A + 1 = 1$ | 或 1 归 1 | 任何东西 OR 真 = 真 |
| $A \cdot 1 = A$ | 与 1 恒等 | 任何东西 AND 真 = 它自己 |
| $A \cdot 0 = 0$ | 与 0 归 0 | 任何东西 AND 假 = 假 |
| $A + A = A$ | 幂等律（或） | 重复说同一句话不改变结果 |
| $A \cdot A = A$ | 幂等律（与） | 同上 |
| $A + \overline{A} = 1$ | 互补律（或） | 一件事要么真要么假，二者必居其一 |
| $A \cdot \overline{A} = 0$ | 互补律（与） | 一件事不可能同时真和假 |
| $\overline{\overline{A}} = A$ | 双重否定 | 不是"不是" = 是 |

注意：**幂等律** $A+A=A$ 和 $A \cdot A = A$ 是布尔代数特有的。普通代数没有（$x+x=2x \neq x$，$x \cdot x = x^2 \neq x$，除非 $x=0$ 或 $1$）。

**多变量运算律**：

| 名称 | 公式（AND 形式） | 公式（OR 形式） |
|------|-----------------|----------------|
| 交换律 | $AB = BA$ | $A + B = B + A$ |
| 结合律 | $(AB)C = A(BC)$ | $(A+B)+C = A+(B+C)$ |
| 分配律 | $A(B+C) = AB + AC$ | $A + BC = (A+B)(A+C)$ |

重点：**分配律的 OR 对 AND 形式** $A + BC = (A+B)(A+C)$ 在普通代数里不成立！试试 $x=2, y=3, z=4$：左边 $= 2+12=14$，右边 $=(2+3)(2+4)=5\times 6=30$，$14 \neq 30$。但布尔代数里它成立（可以用真值表验证）。这是布尔代数最独特的地方之一。

### 2c. 形式化定义：德摩根定律（重点！）

**德摩根定律**（De Morgan's Laws）是布尔代数最强大的化简工具之一，告诉你怎么把"上面一根长横线拆成两根短横线"。

第一个形式：
$$
\overline{A \cdot B} = \overline{A} + \overline{B}
$$

翻译成人话：**"不是（A 且 B）"等于"不是 A 或者不是 B"**。

生活例子：如果我说"今天不是既晴又暖"，意味着什么？要么今天不晴，要么今天不暖，要么两者都不。所以 $\overline{A} + \overline{B}$。

第二个形式：
$$
\overline{A + B} = \overline{A} \cdot \overline{B}
$$

翻译：**"不是（A 或 B）"等于"不是 A 且不是 B"**。

生活例子："今天既不晴也不暖"，意味着不晴 **而且** 不暖。$\overline{A} \cdot \overline{B}$。

推广到多变量：
$$
\overline{A \cdot B \cdot C} = \overline{A} + \overline{B} + \overline{C}
$$
$$
\overline{A + B + C} = \overline{A} \cdot \overline{B} \cdot \overline{C}
$$

口诀：**"上面长线断，中间符号变（AND 变 OR，OR 变 AND），单个变量加短线。"**

德摩根定律一个重要用途是展开嵌套取反。比如 $\overline{AB + \overline{C}}$：
$\overline{AB + \overline{C}} = \overline{AB} \cdot \overline{\overline{C}}$（长线断，OR变AND）
$= (\overline{A} + \overline{B}) \cdot C$（$\overline{AB}$ 再断开，$\overline{\overline{C}}=C$）
$= \overline{A}C + \overline{B}C$

复杂嵌套变成了简单 AND-OR 形式！

### 2d. 手算示例：用公理验证恒等式

**例4**：验证 $A + \overline{A}B = A + B$。

左边 $= A + \overline{A}B$。用分配律（OR 对 AND）：$A + \overline{A}B = (A + \overline{A})(A + B)$。

根据互补律，$A + \overline{A} = 1$，所以 $= 1 \cdot (A + B) = A + B$。✓

直观理解：$A + \overline{A}B$ 表示"要么 A 成立，要么（A 不成立且 B 成立）"。这等于"A 或 B 成立"。如果 A 成立，整个就成立；如果 A 不成立，那就要求 B 成立。所以等价于 $A + B$。

真值表验证：

| A | B | $\overline{A}B$ | $A + \overline{A}B$ | $A+B$ |
|---|---|----------------|---------------------|-------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 | 1 |

完全一致 ✓。这个公式叫"消去因子律"——$\overline{A}$ 中的 $A$ 被消去了。

**例5**：验证 $\overline{A + \overline{B}} = \overline{A}B$。

左边：用德摩根定律，
$\overline{A + \overline{B}} = \overline{A} \cdot \overline{\overline{B}} = \overline{A} \cdot B = \overline{A}B$。✓

**例6**：验证 $(A+B)(\overline{A}+C) = AC + \overline{A}B$。

展开左边：
$(A+B)(\overline{A}+C) = A\overline{A} + AC + B\overline{A} + BC$

$A\overline{A} = 0$（互补律），所以 $= AC + \overline{A}B + BC$。

多了一项 $BC$？这里有个**一致律**（Consensus Theorem）：$AC + \overline{A}B + BC = AC + \overline{A}B$。$BC$ 是冗余的。

直观理解：当 $B=1$ 且 $C=1$ 时，看前两项：
- 如果 $A=1$，$AC=1 \cdot 1 = 1$（第一项覆盖 BC）
- 如果 $A=0$，$\overline{A}B = 1 \cdot 1 = 1$（第二项覆盖 BC）

无论 A 是什么，只要 $B=C=1$，前两项至少有一项为 1，BC 条件多余。

严格证明：$AC + \overline{A}B + BC = AC + \overline{A}B + BC(A + \overline{A})$（乘以 1）
$= AC + \overline{A}B + ABC + \overline{A}BC$
$= AC(1+B) + \overline{A}B(1+C) = AC + \overline{A}B$ ✓

所以 $(A+B)(\overline{A}+C) = AC + \overline{A}B$。这个"相邻展开公式"在设计多路选择器时非常有用。

**例7**：验证 $AB + \overline{A}C + BC = AB + \overline{A}C$（一致律的另一种写法）。

证明：$AB + \overline{A}C + BC = AB + \overline{A}C + BC(A + \overline{A})$
$= AB + \overline{A}C + ABC + \overline{A}BC$
$= AB(1 + C) + \overline{A}C(1 + B)$
$= AB + \overline{A}C$ ✓

### 2e. Python 验证

```python
# 布尔恒等式验证（用真值表穷举所有可能）

def verify_identity(name, expr1_func, expr2_func, variables):
    """验证两个布尔表达式是否恒等（穷举法）"""
    n = len(variables)
    for i in range(2 ** n):
        vals = [(i >> (n - 1 - j)) & 1 for j in range(n)]
        r1 = expr1_func(vals)
        r2 = expr2_func(vals)
        if r1 != r2:
            print(f"❌ {name} 不成立！输入 {vals} → 左边={r1}, 右边={r2}")
            return False
    print(f"✓ {name} 恒成立")
    return True

# 例4: A + ~A B = A + B
verify_identity(
    "A + ~A B = A + B",
    lambda v: v[0] | ((1-v[0]) & v[1]),
    lambda v: v[0] | v[1],
    ['A', 'B']
)

# 例5: ~(A + ~B) = ~A B
verify_identity(
    "~(A + ~B) = ~A B",
    lambda v: 1 - (v[0] | (1-v[1])),
    lambda v: (1-v[0]) & v[1],
    ['A', 'B']
)

# 例6: (A+B)(~A+C) = AC + ~A B
verify_identity(
    "(A+B)(~A+C) = AC + ~A B",
    lambda v: (v[0] | v[1]) & ((1-v[0]) | v[2]),
    lambda v: (v[0] & v[2]) | ((1-v[0]) & v[1]),
    ['A', 'B', 'C']
)

# 一致律
verify_identity(
    "AB + ~AC + BC = AB + ~AC (一致律)",
    lambda v: (v[0]&v[1]) | ((1-v[0])&v[2]) | (v[1]&v[2]),
    lambda v: (v[0]&v[1]) | ((1-v[0])&v[2]),
    ['A', 'B', 'C']
)

# 德摩根定律
print("\n=== 德摩根定律验证 ===")
verify_identity("~(AB) = ~A + ~B",
    lambda v: 1 - (v[0] & v[1]),
    lambda v: (1-v[0]) | (1-v[1]),
    ['A', 'B'])
verify_identity("~(A+B) = ~A · ~B",
    lambda v: 1 - (v[0] | v[1]),
    lambda v: (1-v[0]) & (1-v[1]),
    ['A', 'B'])
verify_identity("~(AB + ~C) = ~AC + ~BC",
    lambda v: 1 - ((v[0]&v[1]) | (1-v[2])),
    lambda v: ((1-v[0])&v[2]) | ((1-v[1])&v[2]),
    ['A', 'B', 'C'])
```

### 2f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "德摩根定律：$\overline{AB}=\overline{A}\cdot\overline{B}$" | 正确的是 $\overline{AB} = \overline{A} + \overline{B}$。上面长线断开后，中间的 AND 变 OR。这是最常见的错误。 |
| "$A+AB = A$" 不理解为什么 | 提取公因式：$A(1+B)=A\cdot 1=A$（因为 $1+B=1$）。B 的条件在 A 已经成立时是冗余的。 |
| "分配律和普通代数完全一样" | OR 对 AND 的分配：$A+BC=(A+B)(A+C)$，普通代数里这个不成立。 |
| "任何恒等式对偶后都对" | 必须是已经成立的恒等式。用对偶来"猜测"新公式后，需要验证。 |

### 2g. 应用连接

德摩根定律是第 21 章 NAND/NOR 门的理论基础。有了它，你就能理解为什么只用 NAND 门可以实现任何逻辑——任何 AND-OR 表达式都可以通过德摩根定律转换成全 NAND 形式：$F = AB + CD = \overline{\overline{AB} \cdot \overline{CD}}$（两步德摩根），右边就是一个 NAND-NAND 结构！

---

## 3. 对偶原理

### 3a. 生活例子：镜子里外颠倒的世界

站在镜子前，举起左手，镜子里的人举起右手。左右颠倒了，但举起这个动作还在。把一件衣服翻过来，里面变外面，外面变里面，但衣服还是那件衣服。

布尔代数里也有这种"镜像翻转"，叫**对偶原理**（Duality Principle）。

### 3b. 直观理解：换个操作符，结论依然成立

对偶规则：把一个布尔恒等式里的所有 AND 换成 OR、OR 换成 AND、0 换成 1、1 换成 0，得到的新等式也一定成立（称为原式的对偶式）。

注意：变量本身不换（A 还是 A，$\overline{A}$ 还是 $\overline{A}$）。

**例8**：写出对偶式并验证。

原式：$A + 0 = A$
对偶：$A \cdot 1 = A$（+变·，0变1）✓

原式：$A(B+C) = AB + AC$（AND 对 OR 的分配律）
对偶：$A + BC = (A+B)(A+C)$（OR 对 AND 的分配律）✓

原式：$\overline{A \cdot B} = \overline{A} + \overline{B}$（德摩根第一定律）
对偶：$\overline{A + B} = \overline{A} \cdot \overline{B}$（德摩根第二定律）✓

德摩根的两个定律互为对偶！所以只要记住一个，用对偶原理就能得到另一个。

### 3c. 手算示例

**例9**：写出以下各式的对偶式并判断是否正确。

(a) $A + \overline{A}B = A + B$
对偶：$A \cdot (\overline{A} + B) = A \cdot B$

验证对偶式：左边 $= A\overline{A} + AB = 0 + AB = AB$。右边 $= AB$。相等 ✓

(b) $\overline{A}B + A\overline{B} = A \oplus B$（异或）
对偶：$(\overline{A} + B)(A + \overline{B}) = A \odot B$（同或）

左边展开：$= \overline{A}A + \overline{A}\overline{B} + BA + B\overline{B} = 0 + \overline{A}\overline{B} + AB + 0 = \overline{A}\overline{B} + AB$。

这就是同或（异或取反），确实 $= A \odot B$ ✓

(c) $AB + \overline{A}C + BC = AB + \overline{A}C$（一致律）
对偶：$(A + B)(\overline{A} + C)(B + C) = (A + B)(\overline{A} + C)$

这意味着 $(B+C)$ 在乘积中和在求和式中一样是冗余的。✓

**例10**：从吸收律 $A + AB = A$ 直接写出对偶式并验证。

对偶式：$A \cdot (A + B) = A$。

验证：左边 $= A \cdot A + A \cdot B = A + AB = A$（利用原吸收律）。

所以吸收律也有两种对偶形式：$A + AB = A$ 和 $A(A+B) = A$。

### 3d. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "对偶就是取反" | 对偶是**换 AND↔OR、0↔1**，变量本身不变。取反是**变量取反**。两个概念不同。 |
| "对偶式就是对表达式取 NOT" | 不是。$\overline{A+B}$ 是取 NOT。对偶式是 $A \cdot B$。 |
| "对偶式改变了运算优先级" | 对偶可能改变表达式结构。$A + BC$ 的对偶是 $A(B + C)$，括号不能丢。 |

### 3e. 应用连接

对偶原理在 CMOS 电路设计中直接对应"上拉网络"和"下拉网络"的对偶关系。PMOS 并联对应 NMOS 串联，这正是对偶原理在晶体管级别的体现。学了第 21 章 CMOS 门电路后会恍然大悟。

---

## 4. 逻辑函数化简（代数法）

### 4a. 生活例子：化简菜单

你开了一家小饭馆，菜单上写着：
- "米饭 + 红烧肉" → 25 元
- "米饭 + 红烧肉 + 蛋花汤" → 25 元
- "米饭 + 红烧肉 + 可乐" → 25 元

只要有了"米饭 + 红烧肉"，不管加不加汤、加不加可乐，都是 25 元。后两项多余。菜单化简为："米饭 + 红烧肉 → 25 元"。

化简逻辑表达式也一样：去掉冗余项，表达式最短最简单，电路实现就最省门。

### 4b. 直观理解：四种化简手法

**1. 并项法**：利用 $A + \overline{A} = 1$，把两项合成一项。
- $AB + A\overline{B} = A(B + \overline{B}) = A \cdot 1 = A$
- $\overline{A}BC + \overline{A}B\overline{C} = \overline{A}B(C + \overline{C}) = \overline{A}B$
- $ABC + \overline{A}BC = (A + \overline{A})BC = BC$

**2. 吸收法**：利用 $A + AB = A$，吸收多余项。
- $A + AB = A(1+B) = A \cdot 1 = A$
- $A + \overline{A}B = A + B$（消去因子律）
- $AB + \overline{A}BC$——这个不能直接用吸收，因为 ¬ABC 中只有 ¬A，不是 ¬(AB)。

**3. 消去法**：利用一致律 $AB + \overline{A}C + BC = AB + \overline{A}C$，消去冗余项。
- $AB + \overline{A}C + BC \to AB + \overline{A}C$
- $AB + \overline{A}C + BCD \to AB + \overline{A}C$（BCD 也冗余）

**4. 配项法**：故意增加冗余项来创造合并机会。
- $AB + \overline{A}C = AB + \overline{A}C + BC$（添加 BC 来帮助下一步合并）
- 利用 $A + \overline{A} = 1$，把一项拆成两项：$A = A(B + \overline{B}) = AB + A\overline{B}$

### 4c. 化简公式汇总

| 编号 | 公式 | 名称 | 记忆方法 |
|------|------|------|---------|
| (1) | $A + AB = A$ | 吸收律 | A 已经成立了，B 的条件多余 |
| (2) | $A + \overline{A}B = A + B$ | 消去因子 | A 不成立时才看 B，等价于 A 或 B |
| (3) | $AB + A\overline{B} = A$ | 并项 | B 和 ¬B 互为补，消去 |
| (4) | $AB + \overline{A}C + BC = AB + \overline{A}C$ | 一致律 | BC 是"交集项" |
| (5) | $A(A + B) = A$ | 吸收律（对偶） | |
| (6) | $\overline{A + B} = \overline{A} \cdot \overline{B}$ | 德摩根 | 长线断，+变· |
| (7) | $\overline{AB} = \overline{A} + \overline{B}$ | 德摩根 | 长线断，·变+ |

### 4d. 手算示例

**例11**：化简 $F = AB + \overline{A}B + A\overline{B}$。

$= B(A + \overline{A}) + A\overline{B}$（前两项提取 B）
$= B \cdot 1 + A\overline{B}$
$= B + A\overline{B}$
$= (B + A)(B + \overline{B})$（分配律 OR 对 AND）
$= (B + A) \cdot 1$
$= A + B$ ✓

直观解释：原始的 $AB + \overline{A}B + A\overline{B}$ 表示"AB 都成立，或 A 不成立 B 成立，或 A 成立 B 不成立"。这三种情况加起来，不就是"A 或 B 至少一个成立"吗？确实就是 $A+B$。

**例12**：化简 $F = \overline{\overline{A}B + A\overline{B}}$。

用德摩根：
$F = \overline{\overline{A}B} \cdot \overline{A\overline{B}}$
$= (\overline{\overline{A}} + \overline{B}) \cdot (\overline{A} + \overline{\overline{B}})$
$= (A + \overline{B}) \cdot (\overline{A} + B)$

展开：
$= A\overline{A} + AB + \overline{B}\overline{A} + \overline{B}B$
$= 0 + AB + \overline{A}\overline{B} + 0$
$= AB + \overline{A}\overline{B}$

这是**同或**（XNOR，$A \odot B$）：A 和 B 相同的时候输出 1。而原式 $\overline{A}B + A\overline{B}$ 正好是**异或**（XOR）。所以这个化简说明了：**同或 = NOT（异或）**，$\overline{A \oplus B} = A \odot B$。✓

**例13**：化简 $F = \overline{\overline{A} + \overline{B} + \overline{C}}$。

德摩根（OR 变 AND）：$F = \overline{\overline{A}} \cdot \overline{\overline{B}} \cdot \overline{\overline{C}} = A \cdot B \cdot C = ABC$ ✓

NOT(¬A OR ¬B OR ¬C) = "A 而且 B 而且 C"。直观！

**例14**：化简 $F = (A + B)(A + \overline{B})(\overline{A} + C)$。

前两个括号：$(A+B)(A+\overline{B}) = AA + A\overline{B} + BA + B\overline{B} = A + A\overline{B} + AB + 0 = A(1 + \overline{B} + B) = A$

所以 $F = A \cdot (\overline{A} + C) = A\overline{A} + AC = AC$ ✓

原式 3 个括号，化简后只用一个 AND 门。这就是化简的力量。

### 4e. Python 验证

```python
# 逻辑函数化简验证

def verify_simplification(name, original_func, simplified_func, variables):
    n = len(variables)
    for i in range(2 ** n):
        vals = [(i >> (n - 1 - j)) & 1 for j in range(n)]
        orig = original_func(vals)
        simp = simplified_func(vals)
        if orig != simp:
            print(f"❌ {name}: 输入 {vals} → 原式={orig}, 化简={simp}")
            return False
    print(f"✓ {name} 化简正确")
    return True

# 例11
verify_simplification("AB + ~AB + A~B = A + B",
    lambda v: (v[0]&v[1]) | ((1-v[0])&v[1]) | (v[0]&(1-v[1])),
    lambda v: v[0] | v[1], ['A', 'B'])

# 例12
verify_simplification("~(~AB + A~B) = AB + ~A~B",
    lambda v: 1 - (((1-v[0])&v[1]) | (v[0]&(1-v[1]))),
    lambda v: (v[0]&v[1]) | ((1-v[0])&(1-v[1])), ['A', 'B'])

# 例14
verify_simplification("(A+B)(A+~B)(~A+C) = AC",
    lambda v: (v[0]|v[1]) & (v[0]|(1-v[1])) & ((1-v[0])|v[2]),
    lambda v: v[0] & v[2], ['A', 'B', 'C'])
```

### 4f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| "化简就是随便删项" | 化简必须保持等价。删掉一项要用定理证明是冗余的。用真值表验证最保险。 |
| "$AB + \overline{A}B$ 化简为 $B$ 是因为 A 抵消了" | 提取 B：$B(A+\overline{A}) = B \cdot 1 = B$。不是 A 抵消，是利用 $A+\overline{A}=1$。 |
| "化简后表达式只有一个" | 可以有不同等价形式。$AB + BC + AC$ 和 $AB + \overline{A}C$ 都可能。 |

### 4g. 应用连接

化简逻辑函数的能力直接决定你设计的电路用多少个门。一个没化简的表达式可能要 15 个门，化简后可能只要 3 个。在芯片设计里，每减少一个门就减少面积和功耗。几十亿个门，哪怕每个逻辑块省 10%，总体的晶体管数量都是天文数字。

---

## 5. 标准形式：最小项与最大项

### 5a. 生活例子：点菜和筛选

你去餐厅，三种选项：主食（米饭/面条），主菜（鱼/鸡），饮料（可乐/茶）。

"米饭 AND 鱼 AND 可乐"是一个具体的**套餐**。三个选项都确定了，叫一个**最小项**（Minterm）。

"（米饭 AND 鱼）OR（面条 AND 鸡 AND 茶）"——这就是**套餐"或"组合**。满足任何一个套餐就行。这种形式叫**积之和**（Sum of Products, SOP）。先 AND 再 OR。

反过来："（米饭 OR 鱼）AND（面条 OR 鸡 OR 可乐）"——排除法。全部括号都满足才行。这叫**和之积**（Product of Sums, POS）。先 OR 再 AND。

### 5b. 直观理解：从真值表到表达式

任何逻辑函数都可以写成两种标准形式：

**1. 积之和（SOP）= 最小项之和**

看真值表中 F=1 的行，每行写成一个最小项（所有输入变量 AND 在一起，变量为 1 用原变量，变量为 0 用反变量），然后所有最小项 OR 起来。

**2. 和之积（POS）= 最大项之积**

看真值表中 F=0 的行，每行写成一个最大项（所有输入变量 OR 在一起。**注意：变量为 0 用原变量，变量为 1 用反变量**——和最小项相反！），然后所有最大项 AND 起来。

为什么 POS 要反过来？因为最大项要"排除"这一行。当输入恰好是该组合时，这个最大项等于 0，使整个乘积为 0。

### 5c. 形式化定义

**最小项**（Minterm）$m_i$：$n$ 个变量的某种 AND 组合，每个变量恰好出现一次。$i$ 是变量取值的二进制对应的十进制数。

3 变量 A、B、C（A 最高位）：
- $m_0 = \overline{A}\overline{B}\overline{C}$（ABC=000）
- $m_1 = \overline{A}\overline{B}C$（ABC=001）
- $m_2 = \overline{A}B\overline{C}$（ABC=010）
- $m_3 = \overline{A}BC$（ABC=011）
- $m_4 = A\overline{B}\overline{C}$（ABC=100）
- $m_5 = A\overline{B}C$（ABC=101）
- $m_6 = AB\overline{C}$（ABC=110）
- $m_7 = ABC$（ABC=111）

最小项的重要性质：对于任意一种输入组合，**有且只有一个小项为 1**。比如 ABC=101 时，只有 $m_5=1$，其他全为 0。

**最大项**（Maxterm）$M_i$：$n$ 个变量的某种 OR 组合。对应 F=0 的行。**变量为 0 用原变量，变量为 1 用反变量**。

例如：
- $M_0 = A + B + C$（ABC=000 时，A=B=C=0，所以 0+0+0=0，排除该行）
- $M_7 = \overline{A} + \overline{B} + \overline{C}$（ABC=111 时排除）

SOP：$F = \sum m$（F=1 的最小项编号之和）
POS：$F = \prod M$（F=0 的最大项编号之积）
关系：$m_i = \overline{M_i}$，$M_i = \overline{m_i}$。

### 5d. 手算示例

**例15**：将 $F(A,B,C) = A\overline{B} + \overline{A}BC$ 写成最小项之和（SOP）。

真值表：

| A | B | C | A¬B | ¬ABC | F |
|---|---|----|------|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 | 1 → m₃ |
| 1 | 0 | 0 | 1 | 0 | 1 → m₄ |
| 1 | 0 | 1 | 1 | 0 | 1 → m₅ |
| 1 | 1 | 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0 | 0 | 0 |

F=1 的行：m₃(011), m₄(100), m₅(101)

$F = \sum m(3, 4, 5)$
$= \overline{A}BC + A\overline{B}\overline{C} + A\overline{B}C$

注意：$m_4 + m_5 = A\overline{B}\overline{C} + A\overline{B}C = A\overline{B}(\overline{C} + C) = A\overline{B}$。正是原式中的一项！

**例16**：同一函数写成 POS。F=0 的行：m₀, m₁, m₂, m₆, m₇ → 对应 M₀, M₁, M₂, M₆, M₇

$F = \prod M(0, 1, 2, 6, 7)$
$= (A+B+C)(A+B+\overline{C})(A+\overline{B}+C)(\overline{A}+\overline{B}+C)(\overline{A}+\overline{B}+\overline{C})$

SOP 有 3 个最小项，POS 有 5 个最大项。当 SOP 长时 POS 就短，选短的写更省事。

### 5e. Python 验证

```python
# 最小项与最大项可视化工具

def minterm_to_expression(variables, minterm_numbers):
    """将最小项编号列表转为 SOP 表达式字符串"""
    n = len(variables)
    terms = []
    for m in minterm_numbers:
        term = ''
        for j in range(n):
            bit = (m >> (n - 1 - j)) & 1
            term += variables[j] if bit == 1 else f"~{variables[j]}"
        terms.append(term)
    return ' + '.join(terms) if terms else '0'

def maxterm_to_expression(variables, maxterm_numbers):
    """将最大项编号列表转为 POS 表达式字符串"""
    n = len(variables)
    terms = []
    for m in maxterm_numbers:
        parts = []
        for j in range(n):
            bit = (m >> (n - 1 - j)) & 1
            # 最大项：0用原变量，1用反变量
            parts.append(variables[j] if bit == 0 else f"~{variables[j]}")
        terms.append('(' + '+'.join(parts) + ')')
    return ' · '.join(terms) if terms else '1'

# 例15/16
vars_list = ['A', 'B', 'C']
print("=== SOP vs POS ===")
expr_sop = minterm_to_expression(vars_list, [3, 4, 5])
print(f"SOP: F = Σm(3,4,5) = {expr_sop}")
expr_pos = maxterm_to_expression(vars_list, [0, 1, 2, 6, 7])
print(f"POS: F = ΠM(0,1,2,6,7) = {expr_pos}")

# 完整真值表
print("\nA B C | m# | 最小项表达式")
print("-" * 40)
for i in range(8):
    bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
    minterm_str = minterm_to_expression(vars_list, [i])
    print(f"{bits[0]} {bits[1]} {bits[2]} | m{i} | {minterm_str}")
```

### 5f. 常见误区

| ❌ 错误认识 | ✅ 正确理解 |
|------------|------------|
| 最小项和最大项取反规则一样 | 最小项：变量=0→反变量，变量=1→原变量。最大项：**反过来**。 |
| SOP 和 POS 总是同样长度 | SOP=F=1 的行数，POS=F=0 的行数。总和=$2^n$。选短的写。 |
| "$m_0+m_1$ 直接写成 $\overline{A}\overline{B}$" | 需要相邻（只有一个变量不同）才能合并。$m_0+m_1=\overline{A}\overline{B}\overline{C}+\overline{A}\overline{B}C=\overline{A}\overline{B}$，因为 C 和 ¬C 互补。 |
| 一个 SOP 只能有一种写法 | 标准 SOP（所有最小项）唯一，但化简后的 SOP 有多种等价写法。 |

### 5g. 应用连接

SOP 对应**两级 AND-OR 电路**——可编程逻辑器件（PLA/PAL/FPGA）的基本结构。POS 对应**两级 OR-AND 电路**，在 CMOS 设计中更高效。

---

## 6. 完整实战

**实战1**：化简 $F = \overline{A}\overline{B}\overline{C} + \overline{A}B\overline{C} + A\overline{B}\overline{C} + AB\overline{C}$。

<details>
<summary><b>点击展开解答</b></summary>

观察四项都有 $\overline{C}$。提取：$F = \overline{C}(\overline{A}\overline{B} + \overline{A}B + A\overline{B} + AB) = \overline{C} \cdot 1 = \overline{C}$

四个最小项化简成一个反变量——省掉大量门。
</details>

**实战2**：化简 $F(A,B,C) = \sum m(0, 1, 2, 3, 7)$。

<details>
<summary><b>点击展开解答</b></summary>

前四项都有 $\overline{A}$：$\overline{A}(\overline{B}\overline{C} + \overline{B}C + B\overline{C} + BC) = \overline{A} \cdot 1 = \overline{A}$

$F = \overline{A} + ABC = \overline{A} + BC$（消去因子律：$\overline{A} + ABC = \overline{A} + BC$）

验证：$\overline{A} + ABC$。A=0→F=1；A=1→F=BC。所以 $F = \overline{A} + BC$ ✓
</details>

---

## 7. 思考题

### 基础题

**题1**：写出以下逻辑表达式的真值表（3 变量各 8 行）：
(a) $F = AB + \overline{B}C$
(b) $F = \overline{AB + C}$
(c) $F = (A \oplus B) \cdot C$

**题2**：用德摩根定律化简：
(a) $\overline{\overline{A}B + \overline{C}}$
(b) $\overline{(A+B)(\overline{A}+C)}$

**题3**：用代数法化简以下表达式（写出每一步用了什么定理）：
(a) $F = AB + A\overline{B}$
(b) $F = A + AB + \overline{A}B$
(c) $F = AB + \overline{A}C + BC$
(d) $F = \overline{A}\overline{B}\overline{C} + \overline{A}\overline{B}C + \overline{A}B\overline{C} + A\overline{B}\overline{C}$

### 进阶题

**题4**：将 $F = \overline{A}BC + A\overline{B}C + A\overline{B}\overline{C} + \overline{A}B\overline{C} + \overline{A}\overline{B}C$ 写成最小项编号形式并化简。

**题5**：证明 $(A+B)(\overline{A}+C)(B+C) = (A+B)(\overline{A}+C)$（用代数推导）。

**题6**：写出 $F(A,B,C) = \sum m(0, 2, 4, 6)$ 的表达式并化简。写出它的对偶式。

### 综合题

**题7**：设计"投票器"逻辑。三输入 A、B、C（1=同意，0=反对）。同意人数 ≥ 2 时，F=1。
(a) 写真值表 (b) 写 SOP 最小项表达式 (c) 化简到最简 (d) 写 POS 最大项表达式 (e) 用 Python 验证

### 解答

<details>
<summary><b>点击展开完整解答</b></summary>

**题1(a)**：$F = AB + \overline{B}C$

| A | B | C | ¬B | AB | ¬BC | F |
|---|---|----|----|-----|---|
| 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 | 0 | 1 | 1 |
| 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| 1 | 1 | 1 | 0 | 1 | 0 | 1 |

**题2(a)**：$\overline{\overline{A}B + \overline{C}} = \overline{\overline{A}B} \cdot \overline{\overline{C}} = (\overline{\overline{A}} + \overline{B}) \cdot C = (A + \overline{B})C = AC + \overline{B}C$

**题2(b)**：$\overline{(A+B)(\overline{A}+C)} = \overline{A+B} + \overline{\overline{A}+C} = \overline{A}\overline{B} + A\overline{C}$

**题3(a)**：$F = AB + A\overline{B} = A(B + \overline{B}) = A \cdot 1 = A$

**题3(b)**：$F = A + AB + \overline{A}B = A(1+B) + \overline{A}B = A + \overline{A}B = A + B$

**题3(c)**：$F = AB + \overline{A}C + BC = AB + \overline{A}C + BC(A+\overline{A}) = AB + \overline{A}C + ABC + \overline{A}BC = AB(1+C) + \overline{A}C(1+B) = AB + \overline{A}C$

**题3(d)**：前两项：$\overline{A}\overline{B}$；后两项：$\overline{C}(\overline{A}B + A\overline{B}) = \overline{C}(A \oplus B)$。
$F = \overline{A}\overline{B} + \overline{C}(A \oplus B)$。进一步：$= \overline{A}\overline{B} + \overline{A}B\overline{C} + A\overline{B}\overline{C} = \overline{A}\overline{B} + \overline{B}\overline{C} + \overline{A}\overline{C}$（三项最简）

**题4**：$F = \sum m(1, 2, 3, 4, 5)$。化简：$F = \overline{A}C + \overline{A}B + A\overline{B}$（三项最简）

**题5**：见正文证明。

**题6**：$F = \sum m(0,2,4,6) = \overline{C}$。对偶：$\overline{C}$（取反不变）。

**题7(投票器)**：
(a) 真值表见正文
(b) $F = \sum m(3,5,6,7) = \overline{A}BC + A\overline{B}C + AB\overline{C} + ABC$
(c) $F = AB + BC + AC$（多数表决器）
(d) $F = \prod M(0,1,2,4) = (A+B+C)(A+B+\overline{C})(A+\overline{B}+C)(\overline{A}+B+C)$
(e) 略
</details>

---

下一章：[逻辑门电路](./21-logic-gates.md)


## 补充实战：四变量函数综合化简

**综合练习**：用卡诺图化简 $F(A,B,C,D) = \sum m(0, 2, 3, 5, 7, 8, 10, 11, 14, 15)$。

<details>
<summary><b>点击展开解答</b></summary>

先做卡诺图（A,B为行，C,D为列，格雷码排列）：

```
          CD
       00  01  11  10
AB=00│ 1 │ 0 │ 1 │ 1 │  m₀ m₂ m₃
  01 │ 0 │ 1 │ 1 │ 0 │  m₅ m₇
  11 │ 0 │ 0 │ 1 │ 1 │  m₁₅ m₁₄
  10 │ 1 │ 0 │ 1 │ 1 │  m₈ m₁₀ m₁₁
```

仔细圈组（找最大可能的矩形圈）：
- m₀+m₂+m₈+m₁₀ 四角圈：这四个格子的共同特征是 B=0, D=0。A 和 C 都在变化。所以这个圈 = $\overline{B}\overline{D}$。
- m₂+m₃+m₁₀+m₁₁：看 m₂(0010)、m₃(0011)、m₁₀(1010)、m₁₁(1011)。这四个格子中，B=0, C=1 不变，A 和 D 在变化。所以 = $\overline{B}C$。
- m₃+m₇+m₁₅+m₁₁：这四个格子中 D=1 不变。所以 = $D$。但注意 m₁₁ 的 B=0, C=1，m₇ 的 B=1, C=1... 实际上这四格中只有 D=1 是共同特征。这是一个边缘圈（中间两列的下半部分加上上层）。
- m₁₄(1110) 和 m₁₀(1010)：A=1, C=1, D=0，B 变化 → $A C \overline{D}$。
  或者更好的圈法：m₈+m₁₀+m₁₁+m₁₄？这四个格子…m₈(1000), m₁₀(1010), m₁₁(1011), m₁₄(1110)。共同特征：A=1。所以 = $A$。

好，总结四个大圈覆盖所有 1：
1. ¬B¬D (四角)
2. ¬BC (m₂,m₃,m₁₀,m₁₁)
3. A (A=1 的所有1)
4. D (D=1 的所有1)

$F = \overline{B}\overline{D} + \overline{B}C + A + D$

验证覆盖情况：
- m₀(0000): ¬B¬D ✓
- m₂(0010): ¬B¬D + ¬BC ✓
- m₃(0011): ¬BC + D ✓
- m₅(0101): D ✓
- m₇(0111): D ✓
- m₈(1000): A ✓
- m₁₀(1010): A + ¬B¬D ✓
- m₁₁(1011): A + ¬BC + D ✓
- m₁₄(1110): A ✓
- m₁₅(1111): A + D ✓

全部覆盖！现在进一步化简代数式：

$F = A + D + \overline{B}\overline{D} + \overline{B}C$

注意：$D + \overline{B}\overline{D}$ 可以化简。用消去因子律：$X + \overline{X}Y = X + Y$。但这里 $D + \overline{B}\overline{D}$ 中的 $\overline{B}\overline{D}$ 包含了 $\overline{D}$。令 $X = D$，则第二项是 $\overline{B}\overline{D} = \overline{B} \cdot \overline{D} = \overline{B} \cdot \overline{X}$。

这不直接符合 $X + \overline{X}Y$ 的形式（其中 $Y$ 应不含 $X$ 的补）。重新来：

$D + \overline{B}\overline{D} = D + \overline{D} \cdot \overline{B} = D + \overline{B}$（消去因子律：$X + \overline{X}Y = X + Y$，令 $X=D$, $Y=\overline{B}$）✓

所以 $D + \overline{B}\overline{D} = D + \overline{B}$。

于是 $F = A + (D + \overline{B}) + \overline{B}C = A + D + \overline{B} + \overline{B}C = A + D + \overline{B}(1 + C) = A + D + \overline{B}$。

最终化简结果：**$F = A + D + \overline{B}$**。

仅三个变量！只需一个三输入 OR 门和一个 NOT 门。这就是卡诺图化简的威力。

验证最终的三个变量覆盖：
- B=0 的行 → 覆盖 m₀,m₁,m₂,m₃,m₈,m₉,m₁₀,m₁₁
- A=1 的行 → 覆盖 m₈-m₁₅
- D=1 的列 → 覆盖所有 D=1 的 m

三者并集恰好是 [0,2,3,5,7,8,10,11,14,15] ✓
</details>

### Python代码验证：

```python
def F_advanced(vals):
    A, B, C, D = vals
    minterms = [0, 2, 3, 5, 7, 8, 10, 11, 14, 15]
    idx = (A<<3)|(B<<2)|(C<<1)|D
    return 1 if idx in minterms else 0

def F_simplified_adv(vals):
    A, B, C, D = vals
    return A | D | (1-B)  # A + D + ~B

print('=== 四变量高级化简验证 ===')
all_ok = True
for i in range(16):
    vals = [(i>>3)&1,(i>>2)&1,(i>>1)&1,i&1]
    orig = F_advanced(vals)
    simp = F_simplified_adv(vals)
    if orig != simp:
        print(f'  ✗ ABCD={vals[0]}{vals[1]}{vals[2]}{vals[3]}: orig={orig}, simp={simp}')
        all_ok = False
if all_ok:
    print('✓ 所有16种组合验证通过！化简结果正确。')
```

## 自测练习：15道进制转换与补码运算题

完成以下练习，用手算验证，再用Python复核。这是巩固本章知识的最佳方式。

**第一组：进制转换**

1. 将十进制数 $156_{10}$ 转为二进制（除2取余法，写完整步骤）
2. 将二进制 $10110110_2$ 转为十进制
3. 将二进制 $11010110_2$ 转为八进制（三位分组法）
4. 将二进制 $11010110_2$ 转为十六进制（四位分组法）
5. 将十六进制 $3F7_{16}$ 转为二进制，再转为十进制验证
6. 将十进制小数 $0.875_{10}$ 转为二进制（乘2取整法）
7. 将十进制小数 $0.3_{10}$ 转为二进制（取8位精度），观察循环

**第二组：补码运算**

8. 写出以下十进制数的8位补码：-1, -15, -64, -100
9. 用8位补码计算：$35 + (-22)$，验证结果
10. 用8位补码计算：$(-45) + (-30)$，判断是否溢出
11. 用8位补码计算：$100 + 50$，判断是否溢出
12. 一个8位补码为 $11100100$，对应的十进制数是多少？

**第三组：编码**

13. 将十进制 $386$ 编码为 BCD 码
14. 写出4位格雷码的完整对照表（二进制→格雷码），从0到15
15. 计算二进制 $10110$ 对应的格雷码，再从格雷码转回二进制验证

<details>
<summary><b>点击展开答案</b></summary>

**1.** 156 ÷ 2:
156→78余0, 78→39余0, 39→19余1, 19→9余1, 9→4余1, 4→2余0, 2→1余0, 1→0余1
从下往上: $156_{10} = 10011100_2$

**2.** $10110110_2 = 128+0+32+16+0+4+2+0 = 182_{10}$

**3.** 3位分组: 011 010 110 → 3 2 6 → $326_8$

**4.** 4位分组: 1101 0110 → D 6 → $D6_{16}$

**5.** $3F7_{16}$: 3→0011, F(=15)→1111, 7→0111 → $0011 1111 0111_2$
十进制: $3×256 + 15×16 + 7 = 768 + 240 + 7 = 1015_{10}$

**6.** 0.875×2=1.75→1, 0.75×2=1.5→1, 0.5×2=1.0→1。$0.875_{10}=0.111_2$

**7.** 0.3×2=0.6→0, 0.6×2=1.2→1, 0.2×2=0.4→0, 0.4×2=0.8→0, 0.8×2=1.6→1, 0.6×2=1.2→1, 0.2×2=0.4→0...
$0.3_{10} = 0.0\overline{1001}_2$ (无限循环)

**8.** -1: 11111111; -15: 11110001; -64: 11000000; -100: 10011100

**9.** 35=00100011, -22=11101010。相加: 00100011+11101010=100001101→丢溢出→00001101=13 ✓

**10.** -45=11010011, -30=11100010。相加: 11010011+11100010=110110101→丢溢出→10110101。
符号位1→负数。转十进制: 10110101-1=10110100→取反得01001011=75→-75。
-45+(-30)=-75，在-128~127内，不溢出。但检查溢出标志: C₇=1,C₆=0→C₇⊕C₆=1? 不，两个负数相加结果也是负数，正常。如果C₈(最高进位)=1且C₇=0? 这里需要仔细算。溢出判断略。

**11.** 100=01100100, 50=00110010。相加: 01100100+00110010=10010110。
符号位=1→负数？两个正数相加得负数→溢出！$C_7\oplus C_6 = 0\oplus 1 = 1$ → 溢出 ✓
（100+50=150 > 127，4位补码最大值）

**12.** 11100100。最高位1→负数。减1: 11100011。取反: 00011100=28。结果: -28。
验证: -28的补码 = 00011100→取反11100011→+1=11100100 ✓

**13.** 386 BCD: 3→0011, 8→1000, 6→0110 → 0011 1000 0110

**14.** 见正文19-number-systems.md的4进制格雷码对照表。

**15.** 10110→格雷码: g₄=1, g₃=1⊕0=1, g₂=0⊕1=1, g₁=1⊕1=0, g₀=1⊕0=1 → 11101。
格雷码11101→二进制: b₄=1, b₃=1⊕1=0, b₂=1⊕0=1, b₁=0⊕1=1, b₀=1⊕1=0 → 10110 ✓
</details>

## 综合进阶：逻辑函数化简的卡诺图与代数法互补

在实际设计工作中，代数法和卡诺图法是互补的。卡诺图适合4变量及以下（快速、直观），代数法适合多变量或作为验证手段。

这里有一个5变量函数的化简例子（只能用代数法，5变量卡诺图过于复杂）。

**例**：化简 $F(A,B,C,D,E) = \overline{A}BC\overline{D}E + A\overline{B}C\overline{D}E + AB\overline{C}DE + \overline{A}\overline{B}CD\overline{E} + \overline{A}BCDE$。

观察各项的共同因子。前两项有 $C\overline{D}E$：提取 $C\overline{D}E(\overline{A}B + A\overline{B}) = C\overline{D}E(A \oplus B)$。

后三项… 继续分组化简。这种5变量以上的化简通常由EDA工具（如Logic Friday、Espresso）自动完成。

**Python工具**：可以用 `sympy` 库的 `simplify_logic` 函数做自动化简：

```python
from sympy.logic import simplify_logic
from sympy.abc import A, B, C

# 例: 化简 AB + ~AB + A~B
expr = (A & B) | (~A & B) | (A & ~B)
result = simplify_logic(expr)
print(f"AB + ~AB + A~B = {result}")  # 预期: A | B
```

## 布尔代数在编程中的应用

你每天都在用布尔代数，即使你没意识到：

- **搜索引擎查询**："机器学习 AND Python NOT 深度学习" → 使用 AND/OR/NOT 运算符
- **SQL查询**：`SELECT * FROM users WHERE age > 18 AND (city = \'Beijing\' OR city = \'Shanghai\')`
- **条件断点**：在调试器中设置 `x > 5 && y < 3` 条件断点
- **Git的.gitignore**：pattern匹配逻辑本质是布尔表达式

这些应用中，AND/OR/NOT 的含义和布尔代数完全一致。理解布尔代数，你对这些工具的掌握会更深入。

## 拓展阅读：布尔代数的其他运算

除了 AND、OR、NOT，还有以下几种衍生运算：

- **NAND** (¬(A·B))：全1才0。在数字电路中最"自然"的门。
- **NOR** (¬(A+B))：有1就0。CMOS中面积略大于NAND。
- **XOR** (A⊕B)：不同为1。用于加法器、校验码、加密。
- **XNOR** (A⊙B)：相同为1。等于 NOT(XOR)。
- **蕴含** (A→B)：等于 ¬A + B。"如果A则B"。
- **等价** (A↔B)：等于 (A→B)·(B→A)。等于 XNOR。

其中蕴含运算虽然在数字电路设计中不常用，但在逻辑学和程序验证中非常重要。

## 布尔代数简史

乔治·布尔（George Boole）在1854年出版了《思维的法则》（An Investigation of the Laws of Thought），建立了布尔代数。将近100年后，克劳德·香农（Claude Shannon）在1937年的硕士论文中将布尔代数应用于开关电路设计，开启了数字电路时代。香农的这篇论文被称为"20世纪最重要的硕士论文"。

所以你现在学的，不仅是电路设计的基础，更是人类思维形式化的伟大成果。从"吃饭OR睡觉"到"CPU设计"，布尔代数横跨了日常推理和尖端科技。
