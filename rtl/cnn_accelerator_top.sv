`timescale 1ns/1ps

module cnn_accelerator_top (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,

    // Layer selection: 0: CONV, 1: FC
    input  logic               mode_fc,

    // Weight and Bias inputs
    input  logic signed [31:0] bias_in,
    input  logic signed [7:0]  weight_in,

    // Activation / Pixel inputs
    input  logic signed [7:0]  data_in,
    input  logic               in_valid,
    input  logic               last_element,

    // Max pooling inputs
    input  logic               pool_enable,
    input  logic        [7:0]  pool_p00, pool_p01, pool_p10, pool_p11,

    // Outputs
    output logic signed [31:0] acc_out,
    output logic        [7:0]  act_out_relu,
    output logic        [7:0]  pool_out,
    output logic               busy,
    output logic               done
);

    wire signed [31:0] conv_raw_acc;
    wire        [7:0]  conv_relu_out;
    wire               conv_busy;
    wire               conv_done;

    cnn_conv_engine u_conv (
        .clk(clk),
        .rst_n(rst_n),
        .start(start && !mode_fc),
        .bias_in(bias_in),
        .pixel_in(data_in),
        .weight_in(weight_in),
        .pixel_valid(in_valid && !mode_fc),
        .pixel_last(last_element && !mode_fc),
        .raw_acc(conv_raw_acc),
        .conv_out_relu(conv_relu_out),
        .busy(conv_busy),
        .done(conv_done)
    );

    wire signed [31:0] fc_logit_out;
    wire        [7:0]  fc_relu_out;
    wire               fc_busy;
    wire               fc_done;

    cnn_fc_classifier u_fc (
        .clk(clk),
        .rst_n(rst_n),
        .start(start && mode_fc),
        .bias_in(bias_in),
        .feature_in(data_in),
        .weight_in(weight_in),
        .in_valid(in_valid && mode_fc),
        .last_feature(last_element && mode_fc),
        .is_output_class(1'b0),
        .logit_out(fc_logit_out),
        .feature_out_relu(fc_relu_out),
        .busy(fc_busy),
        .done(fc_done)
    );

    wire        [7:0]  mp_out;
    wire               mp_valid;

    cnn_maxpool2x2 u_pool (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pool_enable),
        .p00(pool_p00),
        .p01(pool_p01),
        .p10(pool_p10),
        .p11(pool_p11),
        .out_valid(mp_valid),
        .max_out(mp_out)
    );

    assign acc_out      = mode_fc ? fc_logit_out : conv_raw_acc;
    assign act_out_relu = mode_fc ? fc_relu_out  : conv_relu_out;
    assign pool_out     = mp_out;
    assign busy         = mode_fc ? fc_busy      : conv_busy;
    assign done         = mode_fc ? fc_done      : conv_done;

endmodule
