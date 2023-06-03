> Computer Architecture Honor Class Lab 5 Report
>
> 郭耸霄 PB20111712

[TOC]

# 数据级并行实验

## CPU

```
Device name	THINKPAD-T14
Processor	12th Gen Intel(R) Core(TM) i7-1260P   2.10 GHz
Installed RAM	16.0 GB (15.7 GB usable)
Device ID	CA4A3DA5-AF88-4F61-BFB3-CB6F73AECD3F
Product ID	00326-10000-00000-AA871
System type	64-bit operating system, x64-based processor
Pen and touch	No pen or touch input is available for this display
Edition	Windows 11 Home
Version	22H2
Installed on	‎5/‎15/‎2023
OS build	22621.1702
Serial number	PF46FFXL
Experience	Windows Feature Experience Pack 1000.22641.1000.0
```

### 输入规模对性能的影响

| 矩阵规模\运行时间（ms）\实现方式 | 基础矩阵乘法 | AVX 矩阵乘法 | AVX 分块（分4块）矩阵乘法 |
| :------------------------------: | :----------: | :----------: | :-----------------------: |
|              16x16               |    0.0182    |    0.0099    |           0.015           |
|              32x32               |    0.2478    |    0.047     |          0.0443           |
|              64x64               |    0.9154    |    0.3606    |          0.3271           |
|             128x128              |   13.3552    |    2.0105    |          2.1816           |
|             256x256              |   79.0386    |   21.4222    |          14.1532          |
|             512x512              |   530.615    |   131.292    |          121.066          |
|            1024x1024             |   3488.06    |   804.659    |          1123.11          |
|            2048x2048             |   26709.7    |   5266.31    |          6609.57          |

### 分块参数对性能的影响

| 分块数 | 2048x2048 运行时间（ms) |
| :----: | :---------------------: |
|   1    |         7785.8          |
|   4    |         6501.18         |
|   16   |         8253.93         |
|   64   |         8630.83         |
|  256   |         5775.16         |
|  1024  |         6362.19         |
|  4096  |         6688.74         |
| 16384  |         9002.06         |
| 65536  |         12124.3         |

### CPU 平台上矩阵乘法的其他优化手段

## GPU

```
Device Name : NVIDIA GeForce MX550.
totalGlobalMem : 2147221504.
sharedMemPerBlock : 49152.
regsPerBlock : 65536.
warpSize : 32.
memPitch : 2147483647.
maxThreadsPerBlock : 1024.
maxThreadsDim[0 - 2] : 1024 1024 64.
maxGridSize[0 - 2] : 2147483647 65535 65535.
totalConstMem : 65536.
major.minor : 7.5.
clockRate : 1320000.
textureAlignment : 512.
deviceOverlap : 1.
multiProcessorCount : 16.
```

### 输入规模对性能的影响

| 矩阵规模\运行时间（ms）\实现方式 | 基础矩阵乘法（block size = 16x16） | 分块矩阵乘法（block size = 16x16） |
| :------------------------------: | :--------------------------------: | :--------------------------------: |
|              16x16               |              0.025408              |              0.072096              |
|              32x32               |              0.026624              |              0.07952               |
|              64x64               |              0.054464              |              0.13008               |
|             128x128              |              0.260704              |              0.236128              |
|             256x256              |              1.78966               |               1.7409               |
|             512x512              |              14.9337               |              13.5334               |
|            1024x1024             |              121.576               |              107.266               |
|            2048x2048             |               679.72               |              598.299               |
|            4096x4096             |              5132.32               |              4263.26               |
|            8192x8192             |              39834.4               |              33961.1               |

### grid size 和 block size 对基础矩阵乘法性能的影响

| grid size | block size | 2048x2048 运行时间（ms) |
| :-------: | :--------: | :---------------------: |
|     8     |    256     |         679.72          |
|    32     |     64     |         712.192         |
|    128    |     16     |         2379.2          |
|    512    |     4      |         9271.48         |
|   2048    |     1      |          36870          |

### grid size 和 BLOCK 对分块矩阵乘法性能的影响

| grid size | block size | 2048x2048 运行时间（ms) |
| :-------: | :--------: | :---------------------: |
|     8     |    256     |         598.299         |
|    32     |     64     |         647.032         |
|    128    |     16     |         2658.44         |
|    512    |     4      |         13756.7         |
|   2048    |     1      |         81840.          |
