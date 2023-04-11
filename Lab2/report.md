Apr 10 21:10
E0:BubbleE=X, FlushE=X
S0:代码逻辑错误

Apr 10 21:28
E1:inst_ID=0
S0:仿真时间过短

Apr 10 21:48
E2:ALU sr=X
S2:CSR 输出接0

Apr 10 21:54
E3:CSR Z/X
S3: initial

Apr 11 07:50
E4:alu_func=0
S4: decoder缺少else

Apr 11 08:00
E5:beq与j同时跳转错误
S5:优先选择br

Apr 11 08:18
E6:beq 无限跳转
S6:跳转时flushE

Apr 11 08:36
E7:bne 跳转失败
S7:加上bne跳转逻辑

Apr 11 08:42
E8:blt 跳转错误
S8: 改为有符号数比较

Apr 11 09:14
E9:前递失败
S9:前两级bubble，exe flush