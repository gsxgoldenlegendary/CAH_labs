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
//`define BHT

`ifdef BHT
    `define BTB
`endif

`define TAG_POSITION 31:BTB_BANK_WIDTH
`define INDEX_POSITION BTB_BANK_WIDTH-1:0

module NPC_Generator(
    input clk, 
    input is_br_EX, reset, bubbleE,
    input wire [31:0] PC, jal_target, jalr_target, br_target,
    input wire [31:0] PC_EX, NPC_EX,
    input wire jal, jalr, br,
    output reg [31:0] NPC,
    output reg pre_fail
    );


`ifdef BTB
    wire [31:0]PC_IF;
    assign PC_IF = PC - 4;
    localparam  BTB_BANK = 64;
    localparam  BTB_BANK_WIDTH = $clog2(BTB_BANK);
    localparam  BTB_TAG_WIDTH = 32 - BTB_BANK_WIDTH;

    reg [31:0]              btb_predict_pc [BTB_BANK];
    reg [BTB_TAG_WIDTH-1:0] btb_branch_tag [BTB_BANK];
    reg                     btb_valid [BTB_BANK];
    reg                     btb_history [BTB_BANK];
    reg [BTB_BANK-1:0] total;
    reg [BTB_BANK-1:0] success;
    
    always @(posedge clk) begin
        if(reset) begin
            total <= 0;
            success <= 0;
            NPC <= 0;
            pre_fail <= 0;
        end
        else if(is_br_EX && !bubbleE) begin
            total <= total + 1;
            if(!pre_fail) 
                success <= success + 1;
        end
    end

    wire [`INDEX_POSITION] btb_index_read, btb_index_write;
    wire [BTB_TAG_WIDTH-1:0] btb_tag_read, btb_tag_write;
    wire btb_hit;

    always @(*) begin
        if(is_br_EX) begin
            if(br) begin
                pre_fail = NPC_EX != br_target;
            end
            else begin
                pre_fail = NPC_EX != PC_EX + 4;
            end
        end
        else begin
            pre_fail = 0;
        end
    end

    assign btb_index_read = PC_IF[`INDEX_POSITION];
    assign btb_tag_read = PC_IF[`TAG_POSITION];
    assign btb_hit = btb_valid[btb_index_read] && (btb_branch_tag[btb_index_read] == btb_tag_read);

    assign btb_index_write = PC_EX[`INDEX_POSITION];
    assign btb_tag_write = PC_EX[31:BTB_BANK_WIDTH];

    always @(posedge clk) begin
        if(reset) begin
            for (integer i = 0; i < BTB_BANK; i++) begin
                btb_valid[i]        = 0;
                btb_branch_tag[i]   = 0;
                btb_predict_pc[i]   = 0;
                btb_history[i]      = 0;
            end
        end
        else if(is_br_EX) begin
`ifndef BHT
            btb_branch_tag[btb_index_write]     <= btb_tag_write;
            btb_predict_pc[btb_index_write]     <= br_target;
            btb_valid[btb_index_write]          <= 1'b1;
            btb_history[btb_index_write]        <= br;//0
`endif   
        end
    end
`endif
`ifdef BHT
    localparam  BHT_BANK = 64;//4096
    localparam  BHT_BANK_WIDTH = $clog2(BHT_BANK);
    localparam  BHT_TAG_WIDTH = 32 - BHT_BANK_WIDTH;

    reg [1:0] bht_state [BHT_BANK];
    wire [BHT_BANK_WIDTH-1:0] bht_index_read, bht_index_write;
    wire bht_hit;
    assign bht_index_read = PC_IF[BHT_BANK_WIDTH-1:0];
    assign bht_index_write = PC_EX[BHT_BANK_WIDTH-1:0];
    assign bht_hit = bht_state[bht_index_read][1];
    
    always @(posedge clk) begin
        if(reset) begin
            for (integer i = 0; i < BHT_BANK; i++) begin
                bht_state[i] <= 1;
            end
        end
        else if(is_br_EX) begin
            if(br) begin
                bht_state[bht_index_write] <= bht_state[bht_index_write] == 3 ? 
                    3 : 
                    bht_state[bht_index_write] + 1;
                btb_branch_tag[btb_index_write]     <= btb_tag_write;
                btb_predict_pc[btb_index_write]     <= br_target;
                btb_valid[btb_index_write]          <= 1'b1;
                btb_history[btb_index_write]        <= bht_state!=0;
            end else begin
                bht_state[bht_index_write] <= bht_state[bht_index_write] == 0 ?
                    0 : 
                    bht_state[bht_index_write] - 1; 
                btb_branch_tag[btb_index_write]     <= btb_tag_write;
                btb_predict_pc[btb_index_write]     <= br_target;
                btb_valid[btb_index_write]          <= 1'b1;
                btb_history[btb_index_write]        <= bht_state!=3;
            end
        end
    end
`endif
`ifndef BTB
always @(*) begin
    pre_fail = br;
end
`endif
 always @(*) begin
`ifdef BTB
        if(pre_fail) begin
            if(br) 
                NPC = br_target;
            else 
                NPC = PC_EX + 4;
        end
`else
        if (br)
        begin
            NPC = br_target;
        end
`endif
        else if(jalr) begin
            NPC = jalr_target;
        end
        else if(jal) begin
            NPC = jal_target;
        end
`ifdef BTB
        else if(btb_hit && btb_history[btb_index_read]
    `ifdef BHT
        && bht_hit
    `endif
        ) begin
            NPC = btb_predict_pc[btb_index_read];
        end
`endif
        else begin
            NPC = PC;
        end
    end
endmodule