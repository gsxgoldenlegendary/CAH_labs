`timescale 1ns / 1ps
// 实验要求
// 补全模块（阶段三）

`include "../Parameters.v"

module CSR_Regfile(input wire clk,
                   input wire rst,
                   input wire CSR_write_en,
                   input wire [11:0] CSR_write_addr,
                   input wire [11:0] CSR_read_addr,
                   input wire [31:0] CSR_data_write,
                   output wire [31:0] CSR_data_read);
    
    // TODO: Complete this module
    
    /* FIXME: Write your code here... */
    reg[31:0] csr_ustatus;
    reg[31:0] csr_uie;
    reg[31:0] csr_utvec;
    reg[31:0] csr_uscratch;
    reg[31:0] csr_uepc;
    reg[31:0] csr_ucause;
    reg[31:0] csr_utval;
    reg[31:0] csr_uip;
    reg[31:0] csr_fflags;
    reg[31:0] csr_frm;
    reg[31:0] csr_fcsr;
    
    // init csr
    initial begin
        csr_ustatus  <= 32'b0;
        csr_uie      <= 32'b0;
        csr_utvec    <= 32'b0;
        csr_uscratch <= 32'b0;
        csr_uepc     <= 32'b0;
        csr_ucause   <= 32'b0;
        csr_utval    <= 32'b0;
        csr_uip      <= 32'b0;
        csr_fflags   <= 32'b0;
        csr_frm      <= 32'b0;
        csr_fcsr     <= 32'b0;
    end
    
    // write in clk negedge, reset in rst posedge
    // if write register in clk posedge,
    // new wb data also write in clk posedge,
    // so old wb data will be written to register
    always @(posedge clk) begin
        if (rst)  begin
            csr_ustatus  <= 32'b0;
            csr_uie      <= 32'b0;
            csr_utvec    <= 32'b0;
            csr_uscratch <= 32'b0;
            csr_uepc     <= 32'b0;
            csr_ucause   <= 32'b0;
            csr_utval    <= 32'b0;
            csr_uip      <= 32'b0;
            csr_fflags   <= 32'b0;
            csr_frm      <= 32'b0;
            csr_fcsr     <= 32'b0;
            end else if (CSR_write_en) begin
            case(CSR_write_addr)
                `ustatus: csr_ustatus   <= CSR_data_write;
                `uie: csr_uie           <= CSR_data_write;
                `utvec: csr_utvec       <= CSR_data_write;
                `uscratch: csr_uscratch <= CSR_data_write;
                `uepc: csr_uepc         <= CSR_data_write;
                `ucause: csr_ucause     <= CSR_data_write;
                `utval: csr_utval       <= CSR_data_write;
                `uip: csr_uip           <= CSR_data_write;
                `fflags: csr_fflags     <= CSR_data_write;
                `frm: csr_frm           <= CSR_data_write;
                `fcsr: csr_fcsr         <= CSR_data_write;
                default: ;
            endcase
        end
    end
    // read data changes when address changes
    assign CSR_data_read = 
    (CSR_read_addr == `ustatus) ? csr_ustatus :
    (CSR_read_addr == `uie) ? csr_uie :
    (CSR_read_addr == `utvec) ? csr_utvec :
    (CSR_read_addr == `uscratch) ? csr_uscratch :
    (CSR_read_addr == `uepc) ? csr_uepc :
    (CSR_read_addr == `ucause) ? csr_ucause :
    (CSR_read_addr == `utval) ? csr_utval :
    (CSR_read_addr == `uip) ? csr_uip :
    (CSR_read_addr == `fflags) ? csr_fflags :
    (CSR_read_addr == `frm) ? csr_frm :
    (CSR_read_addr == `fcsr) ? csr_fcsr :
    32'h0;
    
endmodule
