`timescale 1ns / 1ps
//  功能说明
    //  根据跳转信号，决定执行的下一条指令地址
    //  debug端口用于simulation时批量写入数据，可以忽略
// 输入
    // PC                指令地址（PC + 4, 而非PC）
    // jal_target        jal跳转地址
    // jalr_target       jalr跳转地址
    // br_target         br跳转地址
    // jal               jal == 1时，有jal跳转
    // jalr              jalr == 1时，有jalr跳转
    // br                br == 1时，有br跳转
// 输出
    // NPC               下一条执行的指令地址
// 实验要求  
    // 实现NPC_Generator
`define BTB
`define JAL 1
`define JALR 2
`define BRANCH 3

`ifdef BTB
    `define BRANCH_PC 64:33
    `define PREDICTED_PC 32:1
    `define STATE_BIT 0:0
`endif 
//`define BHT
module NPC_Generator(
    input clk, reset,is_branch,
    input wire [31:0] PC, jal_target, jalr_target, br_target,PC_EX,
    input wire jal, jalr, br,
    output reg [31:0] NPC
    );

    // TODO: Complete this module



`ifdef BTB
wire[31:0] PC_Predicted;
localparam BP_BITS = 5;
localparam BP_SIZE = (1 << BP_BITS) - 1;
reg[64:0] bank[BP_SIZE];

assign PC_Predicted = (bank[PC[BP_BITS:0]][`STATE_BIT] == 1)?
                            (bank[PC[BP_BITS:0]][`BRANCH_PC] == PC?
                            bank[PC[BP_BITS:0]][`PREDICTED_PC]
                            :PC)
                        :PC;

always @ (*)
begin
    if (br == 1)
    begin 
        NPC = br_target;
    end
    else if (jalr == 1)
    begin
        NPC = jalr_target;
    end
    else if (jal == 1)
    begin
        NPC = jal_target;
    end
    else 
    begin
        NPC = PC_Predicted;
    end
end

always @(posedge clk) begin
    if(reset)begin
        for (integer i=0;i<BP_SIZE;i++)begin
            bank[i]<=0;
        end
    end
    else if(is_branch)begin
        bank[PC_EX[BP_BITS:0]][`STATE_BIT]<=br;
        bank[PC_EX[BP_BITS:0]][`BRANCH_PC]<=PC_EX;
        bank[PC_EX[BP_BITS:0]][`PREDICTED_PC]<=br_target;
    end
end
`endif 
`ifdef BHT


`endif
endmodule