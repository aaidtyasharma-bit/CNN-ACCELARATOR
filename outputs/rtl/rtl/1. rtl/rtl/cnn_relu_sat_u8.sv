module cnn_relu_sat_u8 (
    input  logic signed [31:0] value_in,
    output logic        [7:0]  value_out
);

    always_comb begin
        if (value_in <= 0)
            value_out = 8'd0;
        else if (value_in > 32'sd255)
            value_out = 8'd255;
        else
            value_out = value_in[7:0];
    end

endmodule
