`timescale 1ns/1ps

module cnn_lenet5_pipeline (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,

    // Layer 1 input: 784 bytes (28x28 INT8 image)
    input  logic signed [7:0]  img_pixel_in,
    input  logic               img_pixel_valid,
    output logic [9:0]         img_pixel_addr,

    // Status & Outputs
    output logic               busy,
    output logic               done,
    output logic [3:0]         predicted_digit,
    output logic signed [31:0] class_logits [0:9]
);

    // BRAM ROM arrays loaded from exported .hex files
    reg signed [7:0]  conv1_w [0:149];    // 6 * 1 * 5 * 5
    reg signed [7:0]  conv1_b [0:5];      // 6
    reg signed [7:0]  conv2_w [0:2399];   // 16 * 6 * 5 * 5
    reg signed [7:0]  conv2_b [0:15];     // 16
    reg signed [7:0]  fc1_w   [0:47999];  // 120 * 400
    reg signed [7:0]  fc1_b   [0:119];    // 120
    reg signed [7:0]  fc2_w   [0:10079];  // 84 * 120
    reg signed [7:0]  fc2_b   [0:83];     // 84
    reg signed [7:0]  fc3_w   [0:839];    // 10 * 84
    reg signed [7:0]  fc3_b   [0:9];      // 10

    initial begin
        $readmemh("models/hardware/hex/conv1_weight.hex", conv1_w);
        $readmemh("models/hardware/hex/conv1_bias.hex",   conv1_b);
        $readmemh("models/hardware/hex/conv2_weight.hex", conv2_w);
        $readmemh("models/hardware/hex/conv2_bias.hex",   conv2_b);
        $readmemh("models/hardware/hex/fc1_weight.hex",   fc1_w);
        $readmemh("models/hardware/hex/fc1_bias.hex",     fc1_b);
        $readmemh("models/hardware/hex/fc2_weight.hex",   fc2_w);
        $readmemh("models/hardware/hex/fc2_bias.hex",     fc2_b);
        $readmemh("models/hardware/hex/fc3_weight.hex",   fc3_w);
        $readmemh("models/hardware/hex/fc3_bias.hex",     fc3_b);
    end

    // Storage buffers for intermediate feature maps
    reg signed [7:0] img_buf   [0:783];      // Input image (28x28)
    reg signed [7:0] conv1_out [0:4703];     // 6 * 28 * 28 = 4704
    reg signed [7:0] pool1_out [0:1175];     // 6 * 14 * 14 = 1176
    reg signed [7:0] conv2_out [0:1599];     // 16 * 10 * 10 = 1600
    reg signed [7:0] pool2_out [0:399];      // 16 * 5 * 5 = 400 (flattened)
    reg signed [7:0] fc1_out   [0:119];      // 120 features
    reg signed [7:0] fc2_out   [0:83];       // 84 features
    reg signed [31:0] logits   [0:9];        // 10 logits

    typedef enum logic [3:0] {
        IDLE,
        LOAD_IMG,
        EXEC_CONV1,
        EXEC_POOL1,
        EXEC_CONV2,
        EXEC_POOL2,
        EXEC_FC1,
        EXEC_FC2,
        EXEC_FC3,
        ARGMAX,
        DONE_ST
    } state_t;

    state_t state;

    // Iteration counters
    integer i, f, r, c, kr, kc, out_f, in_ch, fc_i, fc_j;
    integer load_cnt;
    integer max_digit;
    logic signed [31:0] max_logit;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            busy            <= 1'b0;
            done            <= 1'b0;
            predicted_digit <= 4'd0;
            img_pixel_addr  <= 10'd0;
            for (i = 0; i < 10; i = i + 1) begin
                class_logits[i] <= 32'sd0;
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy     <= 1'b1;
                        load_cnt <= 0;
                        state    <= LOAD_IMG;
                    end
                end

                LOAD_IMG: begin
                    if (img_pixel_valid) begin
                        img_buf[load_cnt] <= img_pixel_in;
                        load_cnt          <= load_cnt + 1;
                        img_pixel_addr    <= 10'(load_cnt + 1);
                        if (load_cnt == 783) begin
                            state <= EXEC_CONV1;
                        end
                    end
                end

                EXEC_CONV1: begin
                    // Compute Conv1: 6 filters, 28x28 output, padding=2
                    // Downscale by shift >> 8 (divide by 256)
                    for (f = 0; f < 6; f = f + 1) begin
                        for (r = 0; r < 28; r = r + 1) begin
                            for (c = 0; c < 28; c = c + 1) begin
                                integer sum_acc, scaled;
                                sum_acc = ({{24{conv1_b[f][7]}}, conv1_b[f]}) * 17;
                                for (kr = 0; kr < 5; kr = kr + 1) begin
                                    for (kc = 0; kc < 5; kc = kc + 1) begin
                                        integer in_r, in_c;
                                        in_r = r + kr - 2;
                                        in_c = c + kc - 2;
                                        if (in_r >= 0 && in_r < 28 && in_c >= 0 && in_c < 28) begin
                                            sum_acc = sum_acc + (img_buf[in_r * 28 + in_c] * conv1_w[f * 25 + kr * 5 + kc]);
                                        end
                                    end
                                end
                                // Fixed-point scaling by M=89 (Q16 shift >>> 16)
                                scaled = (sum_acc * 89) >>> 16;
                                // ReLU & Saturation
                                if (scaled <= 0)
                                    conv1_out[f * 784 + r * 28 + c] <= 8'd0;
                                else if (scaled > 255)
                                    conv1_out[f * 784 + r * 28 + c] <= 8'd255;
                                else
                                    conv1_out[f * 784 + r * 28 + c] <= scaled[7:0];
                            end
                        end
                    end
                    state <= EXEC_POOL1;
                end

                EXEC_POOL1: begin
                    // 2x2 MaxPool with stride 2: 6 x 28x28 -> 6 x 14x14
                    for (f = 0; f < 6; f = f + 1) begin
                        for (r = 0; r < 14; r = r + 1) begin
                            for (c = 0; c < 14; c = c + 1) begin
                                logic [7:0] v0, v1, v2, v3, m0, m1, max_val;
                                v0 = conv1_out[f * 784 + (r * 2) * 28 + (c * 2)];
                                v1 = conv1_out[f * 784 + (r * 2) * 28 + (c * 2 + 1)];
                                v2 = conv1_out[f * 784 + (r * 2 + 1) * 28 + (c * 2)];
                                v3 = conv1_out[f * 784 + (r * 2 + 1) * 28 + (c * 2 + 1)];
                                m0 = (v0 > v1) ? v0 : v1;
                                m1 = (v2 > v3) ? v2 : v3;
                                max_val = (m0 > m1) ? m0 : m1;
                                pool1_out[f * 196 + r * 14 + c] <= max_val;
                            end
                        end
                    end
                    state <= EXEC_CONV2;
                end

                EXEC_CONV2: begin
                    // Compute Conv2: 16 filters, 6 in-channels, 5x5 (no padding)
                    // Bias mult = 4, Scaled by M=266 (Q16 shift >>> 16)
                    for (out_f = 0; out_f < 16; out_f = out_f + 1) begin
                        for (r = 0; r < 10; r = r + 1) begin
                            for (c = 0; c < 10; c = c + 1) begin
                                integer sum_acc2, scaled2;
                                sum_acc2 = ({{24{conv2_b[out_f][7]}}, conv2_b[out_f]}) * 4;
                                for (in_ch = 0; in_ch < 6; in_ch = in_ch + 1) begin
                                    for (kr = 0; kr < 5; kr = kr + 1) begin
                                        for (kc = 0; kc < 5; kc = kc + 1) begin
                                            integer in_r, in_c;
                                            in_r = r + kr;
                                            in_c = c + kc;
                                            sum_acc2 = sum_acc2 + (pool1_out[in_ch * 196 + in_r * 14 + in_c] * 
                                                                  conv2_w[out_f * 150 + in_ch * 25 + kr * 5 + kc]);
                                        end
                                    end
                                end
                                scaled2 = (sum_acc2 * 266) >>> 16;
                                // ReLU & Saturation
                                if (scaled2 <= 0)
                                    conv2_out[out_f * 100 + r * 10 + c] <= 8'd0;
                                else if (scaled2 > 255)
                                    conv2_out[out_f * 100 + r * 10 + c] <= 8'd255;
                                else
                                    conv2_out[out_f * 100 + r * 10 + c] <= scaled2[7:0];
                            end
                        end
                    end
                    state <= EXEC_POOL2;
                end

                EXEC_POOL2: begin
                    // 2x2 MaxPool with stride 2: 16 x 10x10 -> 16 x 5x5 = 400 features
                    for (f = 0; f < 16; f = f + 1) begin
                        for (r = 0; r < 5; r = r + 1) begin
                            for (c = 0; c < 5; c = c + 1) begin
                                logic [7:0] v0, v1, v2, v3, m0, m1, max_val;
                                v0 = conv2_out[f * 100 + (r * 2) * 10 + (c * 2)];
                                v1 = conv2_out[f * 100 + (r * 2) * 10 + (c * 2 + 1)];
                                v2 = conv2_out[f * 100 + (r * 2 + 1) * 10 + (c * 2)];
                                v3 = conv2_out[f * 100 + (r * 2 + 1) * 10 + (c * 2 + 1)];
                                m0 = (v0 > v1) ? v0 : v1;
                                m1 = (v2 > v3) ? v2 : v3;
                                max_val = (m0 > m1) ? m0 : m1;
                                pool2_out[f * 25 + r * 5 + c] <= max_val;
                            end
                        end
                    end
                    state <= EXEC_FC1;
                end

                EXEC_FC1: begin
                    // FC1: 400 features -> 120 features, bias mult = 6, scaled by M=180 (Q16 shift >>> 16)
                    for (fc_i = 0; fc_i < 120; fc_i = fc_i + 1) begin
                        integer fc_acc, scaled_fc1;
                        fc_acc = ({{24{fc1_b[fc_i][7]}}, fc1_b[fc_i]}) * 6;
                        for (fc_j = 0; fc_j < 400; fc_j = fc_j + 1) begin
                            fc_acc = fc_acc + (pool2_out[fc_j] * fc1_w[fc_i * 400 + fc_j]);
                        end
                        scaled_fc1 = (fc_acc * 180) >>> 16;
                        // ReLU & Saturation
                        if (scaled_fc1 <= 0)
                            fc1_out[fc_i] <= 8'd0;
                        else if (scaled_fc1 > 255)
                            fc1_out[fc_i] <= 8'd255;
                        else
                            fc1_out[fc_i] <= scaled_fc1[7:0];
                    end
                    state <= EXEC_FC2;
                end

                EXEC_FC2: begin
                    // FC2: 120 features -> 84 features, bias mult = 11, scaled by M=166 (Q16 shift >>> 16)
                    for (fc_i = 0; fc_i < 84; fc_i = fc_i + 1) begin
                        integer fc_acc, scaled_fc2;
                        fc_acc = ({{24{fc2_b[fc_i][7]}}, fc2_b[fc_i]}) * 11;
                        for (fc_j = 0; fc_j < 120; fc_j = fc_j + 1) begin
                            fc_acc = fc_acc + (fc1_out[fc_j] * fc2_w[fc_i * 120 + fc_j]);
                        end
                        scaled_fc2 = (fc_acc * 166) >>> 16;
                        // ReLU & Saturation
                        if (scaled_fc2 <= 0)
                            fc2_out[fc_i] <= 8'd0;
                        else if (scaled_fc2 > 255)
                            fc2_out[fc_i] <= 8'd255;
                        else
                            fc2_out[fc_i] <= scaled_fc2[7:0];
                    end
                    state <= EXEC_FC3;
                end

                EXEC_FC3: begin
                    // FC3: 84 features -> 10 output class logits, bias mult = 10, Scaled by M=137 (Q16 shift >>> 16)
                    for (fc_i = 0; fc_i < 10; fc_i = fc_i + 1) begin
                        integer fc_acc, scaled_fc3;
                        fc_acc = ({{24{fc3_b[fc_i][7]}}, fc3_b[fc_i]}) * 10;
                        for (fc_j = 0; fc_j < 84; fc_j = fc_j + 1) begin
                            fc_acc = fc_acc + (fc2_out[fc_j] * fc3_w[fc_i * 84 + fc_j]);
                        end
                        scaled_fc3 = (fc_acc * 137) >>> 16;
                        logits[fc_i] <= scaled_fc3;
                        class_logits[fc_i] <= scaled_fc3;
                    end
                    state <= ARGMAX;
                end

                ARGMAX: begin
                    max_logit = logits[0];
                    max_digit = 0;
                    for (i = 1; i < 10; i = i + 1) begin
                        if (logits[i] > max_logit) begin
                            max_logit = logits[i];
                            max_digit = i;
                        end
                    end
                    predicted_digit <= max_digit[3:0];
                    state           <= DONE_ST;
                end

                DONE_ST: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
