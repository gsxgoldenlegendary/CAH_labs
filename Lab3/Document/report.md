>Computer Architecture H Lab 3 Report
>
>郭耸霄 PB20111712

[TOC]

## 实验数据记录

### 快速排序

| `PROG` | `WAY` | `LINE_ADDR` | `SET_ADDR` | `TAG_ADDR` | `HIT_COUNT` | `MISS_COUNT` | `MISS_RATE` |
| :----: | :---: | :---------: | :--------: | :--------: | :---------: | :----------: | :---------: |
|  SORT  |   2   |      2      |     2      |     8      |             |              |             |
| MATRIX |   2   |      2      |     2      |     8      |             |              |             |
|  SORT  |   2   |      2      |     3      |     7      |             |              |             |
| MATRIX |   2   |      2      |     3      |     7      |             |              |             |
|  SORT  |   2   |      2      |     4      |     6      |             |              |             |
| MATRIX |   2   |      2      |     4      |     6      |             |              |             |
|  SORT  |   2   |      3      |     2      |     7      |             |              |             |
| MATRIX |   2   |      3      |     2      |     7      |             |              |             |
|  SORT  |   2   |      3      |     3      |     6      |             |              |             |
| MATRIX |   2   |      3      |     3      |     6      |             |              |             |
|  SORT  |   2   |      3      |     4      |     5      |             |              |             |
| MATRIX |   2   |      3      |     4      |     5      |             |              |             |
|  SORT  |   2   |      4      |     2      |     6      |             |              |             |
| MATRIX |   2   |      4      |     2      |     6      |             |              |             |
|  SORT  |   2   |      4      |     3      |     5      |             |              |             |
| MATRIX |   2   |      4      |     3      |     5      |             |              |             |
|  SORT  |   2   |      4      |     4      |     4      |             |              |             |
| MATRIX |   2   |      4      |     4      |     4      |             |              |             |
|  SORT  |   4   |      2      |     2      |     8      |             |              |             |
| MATRIX |   4   |      2      |     2      |     8      |             |              |             |
|  SORT  |   4   |      2      |     3      |     7      |             |              |             |
| MATRIX |   4   |      2      |     3      |     7      |             |              |             |
|  SORT  |   4   |      2      |     4      |     6      |             |              |             |
| MATRIX |   4   |      2      |     4      |     6      |             |              |             |
|  SORT  |   4   |      3      |     2      |     7      |             |              |             |
| MATRIX |   4   |      3      |     2      |     7      |             |              |             |
|  SORT  |   4   |      3      |     3      |     6      |             |              |             |
| MATRIX |   4   |      3      |     3      |     6      |             |              |             |
|  SORT  |   4   |      3      |     4      |     5      |             |              |             |
| MATRIX |   4   |      3      |     4      |     5      |             |              |             |
|  SORT  |   4   |      4      |     2      |     6      |             |              |             |
| MATRIX |   4   |      4      |     2      |     6      |             |              |             |
|  SORT  |   4   |      4      |     3      |     5      |             |              |             |
| MATRIX |   4   |      4      |     3      |     5      |             |              |             |
|  SORT  |   4   |      4      |     4      |     4      |             |              |             |
| MATRIX |   4   |      4      |     4      |     4      |             |              |             |
|  SORT  |   8   |      2      |     2      |     8      |             |              |             |
| MATRIX |   8   |      2      |     2      |     8      |             |              |             |
|  SORT  |   8   |      2      |     3      |     7      |             |              |             |
| MATRIX |   8   |      2      |     3      |     7      |             |              |             |
|  SORT  |   8   |      2      |     4      |     6      |             |              |             |
| MATRIX |   8   |      2      |     4      |     6      |             |              |             |
|  SORT  |   8   |      3      |     2      |     7      |             |              |             |
| MATRIX |   8   |      3      |     2      |     7      |             |              |             |
|  SORT  |   8   |      3      |     3      |     6      |             |              |             |
| MATRIX |   8   |      3      |     3      |     6      |             |              |             |
|  SORT  |   8   |      3      |     4      |     5      |             |              |             |
| MATRIX |   8   |      3      |     4      |     5      |             |              |             |
|  SORT  |   8   |      4      |     2      |     6      |             |              |             |
| MATRIX |   8   |      4      |     2      |     6      |             |              |             |
|  SORT  |   8   |      4      |     3      |     5      |             |              |             |
| MATRIX |   8   |      4      |     3      |     5      |             |              |             |
|  SORT  |   8   |      4      |     4      |     4      |             |              |             |
| MATRIX |   8   |      4      |     4      |     4      |             |              |             |

## 数据分析

## 实验结论

## 实验过程记录

### May 6 

13:53 

FIFO_swap_way = X

13:56 

cache_miss 反了

14:31 

tag 不足

14:37 

缺失读回写错

14:50 

LRU 命中时队列错误

15:45 

没有FULL_code

19:28 

riscv32工具链只有windows版本 

### May 7

10:26 

bubbleM

11:43 

FIFO/LRU queue 未分组

14:30 

cache miss 时WB段应该bubble 

16:34 

IR模块不适应存在cache miss的情况

