`timescale 1ns/1ps

module tb_lenet5_end_to_end;

    logic clk = 0;
    logic rst_n = 0;
    logic start = 0;

    logic signed [7:0] img_pixel_in = 0;
    logic              img_pixel_valid = 0;
    logic [9:0]        img_pixel_addr;

    logic              busy;
    logic              done;
    logic [3:0]        predicted_digit;
    logic signed [31:0] class_logits [0:9];

    // Stimulus memory
    reg signed [7:0] test_image [0:783];

    always #5 clk = ~clk;

    cnn_lenet5_pipeline dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .img_pixel_in(img_pixel_in),
        .img_pixel_valid(img_pixel_valid),
        .img_pixel_addr(img_pixel_addr),
        .busy(busy),
        .done(done),
        .predicted_digit(predicted_digit),
        .class_logits(class_logits)
    );

    integer p;

    initial begin
        $dumpfile("outputs/waveforms/lenet5_end_to_end.vcd");
        $dumpvars(0, tb_lenet5_end_to_end);

        $display("================================================================");
        $display("       TESTBENCH: LeNet-5 End-to-End Hardware Inference        ");
        $display("================================================================");

        // Load MNIST test stimulus digit 7
        $readmemh("models/hardware/test_stimulus/test_vector_00_label_7.hex", test_image);

        // 1. Reset Sequence
        repeat (2) @(negedge clk);
        rst_n = 1;

        // 2. Start accelerator
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        // 3. Stream 784 input pixels into accelerator
        $display("[TIME: %0t ns] Streaming 784 INT8 input pixels to accelerator...", $time);
        for (p = 0; p < 784; p = p + 1) begin
            @(negedge clk);
            img_pixel_in    = test_image[p];
            img_pixel_valid = 1;
        end

        @(negedge clk);
        img_pixel_valid = 0;
        $display("[TIME: %0t ns] Image load complete. Pipeline executing Conv1->Pool1->Conv2->Pool2->FC1->FC2->FC3...", $time);

        // 4. Wait for accelerator to finish all layers
        wait(done);
        #1;

        $display("================================================================");
        $display("[TIME: %0t ns] ACCELERATOR INFERENCE COMPLETE!", $time);
        $display("================================================================");
        $display("Raw Output Class Logits from Hardware FC3:");
        for (p = 0; p < 10; p = p + 1) begin
            $display("  Class [%0d]: %10d", p, class_logits[p]);
        end
        $display("----------------------------------------------------------------");
        $display("PREDICTED DIGIT FROM HARDWARE : %0d", predicted_digit);
        $display("GROUND TRUTH PYTHON REFERENCE : 7");

        if (predicted_digit === 4'd7) begin
            $display(">>> VERIFICATION STATUS: PASS (Hardware matches Python model!) <<<");
        end else begin
            $fatal(1, ">>> VERIFICATION STATUS: FAIL (Expected 7, got %0d) <<<", predicted_digit);
        end
        $display("================================================================");

        // Hold final output logits and predicted digit on the waveform for 100 cycles
        repeat (100) @(posedge clk);
        $finish;
    end

    initial begin
        #50000000;
        $fatal(1, "TIMEOUT: End-to-end simulation exceeded time limit.");
    end

endmodule
