
module activation_function_layer_2  #(
                    						 parameter voltage_size = 63,                        
                    						 parameter activation_size = 131,                        
                    						 parameter bias_size = 6                        

						 						     ) 



					   (
					   input wire signed [(voltage_size -1 -1) : 0] sum_w2_into_a1, 			// -1 -1 to match size	      	
					   input wire signed [bias_size -1 : 0] bias, 				      	

		         output reg signed [activation_size -1: 0] activation_value			                 
		                                                                      
					   );                                                              

  wire signed [activation_size -1 : 0] sum_w2_into_a1_plus_bias; 				      	

  assign sum_w2_into_a1_plus_bias = sum_w2_into_a1 + bias;


  always @ (*)
	begin
		if (-2 <= sum_w2_into_a1_plus_bias < 0)
			begin
				activation_value = sum_w2_into_a1_plus_bias + (sum_w2_into_a1_plus_bias * sum_w2_into_a1_plus_bias) + 1;
			end
			
		else if (0 <= sum_w2_into_a1_plus_bias <= 2)
			begin
				activation_value = sum_w2_into_a1_plus_bias - (sum_w2_into_a1_plus_bias * sum_w2_into_a1_plus_bias) + 1;
			end

		else if (sum_w2_into_a1_plus_bias > 2)
			begin
				activation_value = 256;						// a pseudo number to represent 1 in fixed point rep.
			end
			
		else
			begin
				activation_value = 0;
			end	
	end	



endmodule		


