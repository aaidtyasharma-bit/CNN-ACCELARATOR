`timescale 1ns/1ps

module tb_cnn_mac;

    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;
    logic in_valid = 0;
    logic last = 0;

    logic signed [7:0] data_in = 0;
    logic signed [7:0] weight_in = 0;
    logic signed [31:0] bias_in = 0;

    logic signed [31:0] accumulator;
    logic busy, done;

    always #5 clk = ~clk;

    cnn_mac dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_valid(in_valid),
        .last(last),
        .data_in(data_in),
        .weight_in(weight_in),
        .bias_in(bias_in),
        .accumulator(accumulator),
        .busy(busy),
        .done(done)
    );

    task automatic send_pair(
        input logic signed [7:0] x,
        input logic signed [7:0] w,
        input logic is_last
    );
        begin
            @(negedge clk);
            data_in = x;
            weight_in = w;
            in_valid = 1;
            last = is_last;

            @(negedge clk);
            in_valid = 0;
            last = 0;
        end
    endtask

    initial begin
        $dumpfile("outputs/waveforms/verilator_mac.vcd");
        $dumpvars(0, tb_cnn_mac);
        repeat (2) @(negedge clk);
        rst_n = 1;

        @(negedge clk);
        bias_in = 5;
        start = 1;

        @(negedge clk);
        start = 0;

        send_pair(2, 3, 0);
        send_pair(-1, 4, 1);

        wait(done);
        #1;

        if (accumulator !== 32'sd7)
            $fatal(1, "FAIL: expected 7, got %0d", accumulator);

        $display("PASS: MAC result = %0d", accumulator);
        $finish;
    end

    initial begin
        #1000;
        $fatal(1, "TIMEOUT");
    end

endmodule
