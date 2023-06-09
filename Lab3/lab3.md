# 计算机体系结构Lab3
## 实验目的
1. 权衡cache size增大带来的命中率提升收益和存储资源电路面积的开销
2. 权衡选择合适的组相连度（相连度增大cache size也会增大，但是冲突miss会减低）
3. 体会使用复杂电路实现复杂替换策略带来的收益和简单替换策略的优势（有时候简单策略比复杂策略效果不差很多甚至可能更好）
4. 理解写回法的优劣

## 时间安排
* Lab3 Cache实验分为2周，分两个阶段，但**只需验收一次**，阶段一和阶段二的最后验收的时间都在cache实验的第二周结束时，并写一份实验报告
* 实验结束后的一周内需要提交Lab3的实验报告
* **实验工具**：Vivado
* **实验方式**：Vivado自带的波形仿真

## 实验内容
* 阶段一：理解我们提供的直接映射策略的cache，将它修改为N路组相连的cache，并通过我们提供的cache读写测试。
* 阶段二：使用阶段一编写的N路组相连cache，正确运行我们提供的几个程序。
* 实验报告：对不同cache策略和参数进行性能和资源的测试评估，编写实验报告。

### 阶段一(35%)
**我们提供：**

*	一个简单的直接映射、写回带写分配的cache代码([cache.sv](./CacheSrcCode/cache.sv))
*	一个python脚本([generate_cache_tb.py](./CacheSrcCode/generate_cache_tb.py))，该脚本用于生成不同规模的testbench文件。这些testbench文件对cache进行多次随机读写，最后从cache中读取数据并验证cache的正确性。同时，我们也提供一个已经生成的testbench。
*	一份文档[cache编写指导](./Document/Lab3-王轩-cache实验指导.docx)，解读我们提供的cache的结构和时序，并简单的介绍N路组相连cache的修改建议。

**要求：**
阅读并理解我们提供的简单cache的代码，将它修改为N路组相连的（要求组相连度使用宏定义可调）、写回并带写分配的cache。要求实现FIFO、LRU两种替换策略。

**验收要求：**
和阶段二一起验收

### 阶段二(15%)
**我们提供：**

*	一个CPU的仿真顶层文件
*	快速排序（大概进行1000个数的快速排序）、矩阵乘法（可指定矩阵规模）的汇编代码和二进制代码

**要求：**
将阶段一实现的Cache添加到Lab2的CPU中（替换先前的dataram），并添加额外的数据通路，统计Cache缺失率，在Cache缺失时， bubble当前指令及之后的指令。要求能成功运行这个算法（所谓成功运行，是指运行后的结果符合预期）

**验收要求：**
验收时，当场使用我们提供的python脚本生成一个新的testbench，并对自己的cache进行验证（要求FIFO和LRU策略都要验证，并修改组相连度等参数进行多次验证），验证正确后向助教讲解你所编写的代码。

### 实验报告(50%)
使用我们提供的快速排序和矩阵乘法的benchmark进行实验，鼓励自己编写更多的汇编benchmark进行测试，体会cache size、组相连度、替换策略针对不同程序的优化效果，以及策略改变带来的电路面积的变化。针对不同程序，权衡性能和电路面积给出一个较优的cache参数和策略。其中“性能”参数使用运行仿真时的时钟周期数量进行评估。“资源占用”参数使用vivado或其它综合工具给出的综合报告进行评估。进行这一步时需要用阶段一的结果进行一些实验，不能仅仅进行理论分析，实验报告中需要给出实验结果（例如仿真波形的截图、vivado综合报告等）。
*提示：为了方便进行性能评估，建议用上阶段二的缺失率统计功能*



### 其他问题:
1.	暂时只做data cache，instruction cache默认不缺失，仍然使用原有代码充当instruction  cache
2.	在进行cache实验时，为了方便Verilog编写，一律不需要处理非字对齐读写，只需考虑sw和lw这两种“整字读写“的指令。我们提供的相关benchmark中的所有load、store指令也将只有sw和lw。
3.  我们的代码利用封装的Bram模拟DDR，Cache命中时间为1 cycle，模拟DDR命中时间设置为50 cycle（因为真实情况下 cache命中时间为1ns，DDR为50-100ns）
