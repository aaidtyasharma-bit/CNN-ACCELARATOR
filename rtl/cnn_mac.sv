`timescale 1ns/1ps

module cnn_mac (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               in_valid,
    input  logic               last,
    input  logic signed [7:0]  data_in,
    input  logic signed [7:0]  weight_in,
    input  logic signed [31:0] bias_in,
    output logic signed [31:0] accumulator,
    output logic               busy,
    output logic               done
);

    logic signed [15:0] product;

    always_comb begin
        product = data_in * weight_in;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 32'sd0;
            busy        <= 1'b0;
            done        <= 1'b0;
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                accumulator <= bias_in;
                busy        <= 1'b1;
            end else if (busy && in_valid) begin
                accumulator <= accumulator + {{16{product[15]}}, product};

                if (last) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

endmodule
