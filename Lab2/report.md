> Computer Architecture Honor Class Lab 2 Report
>
> 郭耸霄 PB20111712

[TOC]

## 本人工作

### 第一阶段

- 实现`NPC_Generator`、`ImmExtend`、`DataExtend`、`ControllerDecoder`及`BranchDecision`模块。
- 将`CSR_EX`、`CSR_Regfile`及`Hazard`闲置处理。

### 第二阶段

- 实现`Hazard`模块。
- 仿真测试`1testAll`、`2testAll`和`3testAll`。

### 第三阶段

- 实现`CSR_EX`及`CSR_Regfile`模块。
- 根据需求，对`ALU`、`Parameters`及`ControllerDecoder`模块进行适应性调整。
- 仿真测试`CSRtest`。

## 问题及解决方案

#### Apr 10 21:10

- 现象 0：`BubbleE`和`FlushE`信号恒为 X。
- 原因 0：`Hazard`模块未进行闲置处理。
- 解决 0：将`BubbleE`和`FlushE`信号置为特定常量值 0。

#### Apr 10 21:28

- 现象 1：`inst_ID`信号恒为 0。
- 原因 1：仿真时间过短，Cache 尚未完成装载。
- 解决 1：继续仿真 10 $\mu$s。

#### Apr 10 21:48

- 现象 2：`ALU`的输入为 X。
- 原因 2：`ALU`的输入由`CSR`输出通过条件选择，未编写`CSR`模块，导致`CSR`输出为 X。
- 解决 2：将`CSR`的全部输出置为特定常量值 0.

#### Apr 10 21:54

- 现象 3：`CSR`相关信号为 Z 或 X。
- 原因 3：`CSR`模块输出未初始化。
- 解决 3：将`CSR`模块初始化。

#### Apr 11 07:50

- 现象 4：在执行`add`指令时`alu_func`变为 0。
- 原因 4：`Decoder`缺少`else`分支。
- 解决 4：将`Decoder`增加`else`分支及对应输出。

#### Apr 11 08:00

- 现象 5：`beq`与`j`相继出现，使得`PC`跳转到错误的地址。
- 原因 5：`NPC_Generator`优先选择了`j`的跳转地址。
- 解决 5：让`NPC_Generator`优先选择`br`指令跳转的地址。

#### Apr 11 08:18

- 现象 6：`beq`指令导致无限跳转。
- 原因 6：跳转后`EX`段的指令仍为跳转指令。
- 解决 6：`br`指令跳转后`flushE`。

#### Apr 11 08:36

- 现象 7：`bne`指令跳转失败。
- 原因 7：`BranchDecision`模块中未编写相应规则。
- 解决 7：在`BranchDecision`模块中编写`bne`跳转规则。

#### Apr 11 08:42

- 现象 8：`blt`指令跳转错误。
- 原因 8：`BranchDecision`模块`bit`比较错误。
- 解决 8：在`BranchDecision`模块中使用有符号数比较得到`blt`的结果。

#### Apr 11 09:14

- 现象 9：数据相关前递失败。
- 原因 9：未进行`Hazard`相应设计。
- 解决 9：在`load`指令后出现使用`load`结果时，将`IF`、`ID`段`bubble`，`EX`段`flush`。

#### Apr 11 11:56

- 现象 10：宏定义使用报错。
- 原因 10：``define`语句后有分号。
- 解决 10：去掉``define`语句后的分号。

#### Apr 11 14:00

- 现象 11：`CSRRC`无法在不改变数据通路的前提下实现。
- 原因 11：`ALU`中没有相应运算。
- 解决 11：在`ALU`中加入先取反再按位与的运算。

#### Apr 11 14:14

- 现象 12：`CSRRS` 读取值错误。
- 原因 12：`CSR`设为了写优先。
- 解决 12：将`CSR`设为读优先。

## 实验收获

- 复习了《计算机组成原理》课上学习过的 RISC-V 5 级流水 CPU 结构、Verilog HDL 语法以及 Xilinx Vivado EDA 工具的使用。
- 第一次编写了`CSR`寄存器相关的数据通路，对其有了明确的认识。

## 改进意见

- 本次实验本人完成除报告部分花费约 7 h，报告花费约 45 min。
- 实验量总体适中，无尖锐问题。
- 对于“现象 1”所说问题，应该予以提示。
- 对于“现象 11”所说问题，不应该在`ALU`中使用名为`NAND`的操作（`NAND`操作是先与再取反，与实际需求不符），可以改为`ANDN`。



