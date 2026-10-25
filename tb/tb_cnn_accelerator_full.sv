`timescale 1ns/1ps

module tb_cnn_accelerator_full;

    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic mode_fc = 0;

    logic signed [31:0] bias_in = 0;
    logic signed [7:0]  weight_in = 0;
    logic signed [7:0]  data_in = 0;
    logic               in_valid = 0;
    logic               last_element = 0;

    logic               pool_enable = 0;
    logic        [7:0]  p00 = 0, p01 = 0, p10 = 0, p11 = 0;

    logic signed [31:0] acc_out;
    logic        [7:0]  act_out_relu;
    logic        [7:0]  pool_out;
    logic               busy, done;

    // Memory arrays for weights & test stimulus ($readmemh)
    reg signed [7:0] conv1_weights [0:149];
    reg signed [7:0] conv1_bias    [0:5];
    reg signed [7:0] test_image    [0:783];

    always #5 clk = ~clk;

    cnn_accelerator_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .mode_fc(mode_fc),
        .bias_in(bias_in),
        .weight_in(weight_in),
        .data_in(data_in),
        .in_valid(in_valid),
        .last_element(last_element),
        .pool_enable(pool_enable),
        .pool_p00(p00),
        .pool_p01(p01),
        .pool_p10(p10),
        .pool_p11(p11),
        .acc_out(acc_out),
        .act_out_relu(act_out_relu),
        .pool_out(pool_out),
        .busy(busy),
        .done(done)
    );

    integer k;

    initial begin
        $dumpfile("outputs/waveforms/accelerator_full.vcd");
        $dumpvars(0, tb_cnn_accelerator_full);

        $display("================================================================");
        $display("   TESTBENCH: CNN Accelerator Top (Conv + Pool + FC Layers)    ");
        $display("================================================================");

        $readmemh("models/hardware/hex/conv1_weight.hex", conv1_weights);
        $readmemh("models/hardware/hex/conv1_bias.hex",   conv1_bias);
        $readmemh("models/hardware/test_stimulus/test_vector_00_label_7.hex", test_image);

        // 1. Reset
        repeat (2) @(negedge clk);
        rst_n = 1;

        // -------------------------------------------------------------
        // STAGE 1: CONVOLUTION ENGINE (Conv1 Filter 0, 5x5 Kernel)
        // -------------------------------------------------------------
        $display("[STAGE 1] Testing Convolution Engine on MNIST test image...");
        @(negedge clk);
        mode_fc = 0;
        bias_in = {{24{conv1_bias[0][7]}}, conv1_bias[0]};
        start   = 1;

        @(negedge clk);
        start = 0;

        for (k = 0; k < 25; k = k + 1) begin
            @(negedge clk);
            data_in      = test_image[k];
            weight_in    = conv1_weights[k];
            in_valid     = 1;
            last_element = (k == 24);
        end

        @(negedge clk);
        in_valid     = 0;
        last_element = 0;

        wait(done);
        #1;
        $display("  -> Conv1 Filter 0 Raw MAC Accumulator = %0d", acc_out);
        $display("  -> Conv1 ReLU Output Activation       = %0d", act_out_relu);

        // -------------------------------------------------------------
        // STAGE 2: 2x2 MAX POOLING ENGINE
        // -------------------------------------------------------------
        $display("[STAGE 2] Testing 2x2 Max Pooling Engine...");
        @(negedge clk);
        p00 = 8'd14;
        p01 = 8'd88;
        p10 = 8'd45;
        p11 = 8'd22;
        pool_enable = 1;

        @(negedge clk);
        pool_enable = 0;
        @(posedge clk);
        #1;
        $display("  -> Inputs: [14, 88, 45, 22] => MaxPool Output = %0d (Expected: 88)", pool_out);
        if (pool_out !== 8'd88)
            $fatal(1, "FAIL: MaxPool error, expected 88, got %0d", pool_out);

        // -------------------------------------------------------------
        // STAGE 3: FULLY CONNECTED CLASSIFIER ENGINE (FC3 Output Neuron)
        // -------------------------------------------------------------
        $display("[STAGE 3] Testing Fully Connected Classifier Engine...");
        @(negedge clk);
        mode_fc = 1;
        bias_in = 32'sd15;
        start   = 1;

        @(negedge clk);
        start = 0;

        // Stream 4 feature inputs
        @(negedge clk); data_in = 8'd10; weight_in = 8'sd2;  in_valid = 1; last_element = 0;
        @(negedge clk); data_in = 8'd20; weight_in = -8'sd1; in_valid = 1; last_element = 0;
        @(negedge clk); data_in = 8'd5;  weight_in = 8'sd4;  in_valid = 1; last_element = 0;
        @(negedge clk); data_in = 8'd8;  weight_in = -8'sd3; in_valid = 1; last_element = 1;

        @(negedge clk);
        in_valid = 0;
        last_element = 0;

        wait(done);
        #1;
        $display("  -> FC Logit Output = %0d (Expected: 15 + 20 - 20 + 20 - 24 = 11)", acc_out);
        if (acc_out !== 32'sd11)
            $fatal(1, "FAIL: FC Classifier error, expected 11, got %0d", acc_out);

        $display("================================================================");
        $display("   ALL ACCELERATOR ENGINES (CONV, POOL, FC) PASSED HARDWARE!    ");
        $display("================================================================");
        $finish;
    end

    initial begin
        #50000;
        $fatal(1, "TIMEOUT in tb_cnn_accelerator_full");
    end

endmodule
