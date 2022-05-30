module mac_calculator_layer_2	#(parameter number_inputs = 5,      // not the NN output, but the tree output            // SCALABLE DESIGN... whole portion
                                parameter input_size = 59,	              	// weight size
                        
              	    				        	parameter tree_layers = 3				             		// 16 inputs --> 8,4,2,1   ....  (16/(2_power_4)) ... number of layers: 4
		              				         )						                                   		// 48 inputs --> 24,12,6,3,2,1       48/(2_power_6) = 0.75      ceil(0.75) = 1						
			             
			      (
			      input  wire signed[(number_inputs * input_size) - 1 : 0] in, 
					
					input clk,
				  
				  output wire signed[(input_size + tree_layers)-1 :0] out
				  );


wire signed [input_size-1 :0] number_1 = in[(1*input_size)-1 : 0*input_size];
wire signed [input_size-1 :0] number_2 = in[(2*input_size)-1 : 1*input_size];
wire signed [input_size-1 :0] number_3 = in[(3*input_size)-1 : 2*input_size];
wire signed [input_size-1 :0] number_4 = in[(4*input_size)-1 : 3*input_size];
wire signed [input_size-1 :0] number_5 = in[(5*input_size)-1 : 4*input_size];

// ===================================================================================================================== // =============================
						                    // ============================L1 adders // ==========================                             
						 
wire signed [input_size + 1 :0] adder_1_layer_1;							// check whether this is right or not
wire signed [input_size + 1 :0] adder_2_layer_1;					// two layers merged



	(*keep="true"*)
reg signed [input_size + 1 :0] postpipe_1_layer_1;
reg signed [input_size + 1 :0] postpipe_2_layer_1;			// input - 1 ...  to match the size of number 5


// ===================================================================================================================== // =============================
						                    // ============================L3 adders // ==========================                             


assign adder_1_layer_1   = (number_1  + number_2)   + (number_3  + number_4)   ;
assign adder_2_layer_1   =  number_5;



always @ (posedge clk)
begin
		postpipe_1_layer_1 <= adder_1_layer_1;
		postpipe_2_layer_1 <= adder_2_layer_1;
end



assign out = (postpipe_1_layer_1  + postpipe_2_layer_1) ;


endmodule 

