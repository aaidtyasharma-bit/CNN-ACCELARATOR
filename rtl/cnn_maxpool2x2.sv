`timescale 1ns/1ps

module cnn_maxpool2x2 (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic        [7:0]  p00, p01, p10, p11,
    output logic               out_valid,
    output logic        [7:0]  max_out
);

    logic [7:0] max_row0;
    logic [7:0] max_row1;
    logic [7:0] max_val;

    always_comb begin
        max_row0 = (p00 > p01) ? p00 : p01;
        max_row1 = (p10 > p11) ? p10 : p11;
        max_val  = (max_row0 > max_row1) ? max_row0 : max_row1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            max_out   <= 8'd0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                max_out <= max_val;
            end
        end
    end

endmodule
