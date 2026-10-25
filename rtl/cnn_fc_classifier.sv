`timescale 1ns/1ps

module cnn_fc_classifier (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic signed [31:0] bias_in,
    input  logic signed [7:0]  feature_in,
    input  logic signed [7:0]  weight_in,
    input  logic               in_valid,
    input  logic               last_feature,
    input  logic               is_output_class,
    output logic signed [31:0] logit_out,
    output logic        [7:0]  feature_out_relu,
    output logic               busy,
    output logic               done
);

    wire signed [31:0] mac_acc;
    wire               mac_busy;
    wire               mac_done;

    cnn_mac u_fc_mac (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_valid(in_valid),
        .last(last_feature),
        .data_in(feature_in),
        .weight_in(weight_in),
        .bias_in(bias_in),
        .accumulator(mac_acc),
        .busy(mac_busy),
        .done(mac_done)
    );

    wire [7:0] relu_val;
    cnn_relu_sat_u8 u_fc_relu (
        .value_in(mac_acc),
        .value_out(relu_val)
    );

    assign logit_out = mac_acc;
    assign feature_out_relu = relu_val;
    assign busy = mac_busy;
    assign done = mac_done;

endmodule
