`timescale 1ns/1ps

module cnn_conv_engine (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic signed [31:0] bias_in,
    input  logic signed [7:0]  pixel_in,
    input  logic signed [7:0]  weight_in,
    input  logic               pixel_valid,
    input  logic               pixel_last,
    output logic signed [31:0] raw_acc,
    output logic        [7:0]  conv_out_relu,
    output logic               busy,
    output logic               done
);

    wire signed [31:0] mac_acc;
    wire               mac_busy;
    wire               mac_done;

    cnn_mac u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_valid(pixel_valid),
        .last(pixel_last),
        .data_in(pixel_in),
        .weight_in(weight_in),
        .bias_in(bias_in),
        .accumulator(mac_acc),
        .busy(mac_busy),
        .done(mac_done)
    );

    wire [7:0] relu_val;
    cnn_relu_sat_u8 u_relu (
        .value_in(mac_acc),
        .value_out(relu_val)
    );

    assign raw_acc = mac_acc;
    assign conv_out_relu = relu_val;
    assign busy = mac_busy;
    assign done = mac_done;

endmodule
