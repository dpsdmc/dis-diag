module activation_function  #(
                    						 parameter voltage_size = 24,
                    						 parameter activation_size = 54,
                    						 parameter bias_size = 6
						 						     ) 



					   (
					   input wire signed  [voltage_size -1 -1 : 0] sum_weights_into_inputs, 				// -1 -1 to match size      	
						input wire signed [bias_size -1 : 0] bias,			 

						output reg signed [activation_size -1 : 0] activation_value			 
		         
					   );                                                              

  wire signed  [voltage_size : 0] sum_weights_into_inputs_plus_bias ;


  assign sum_weights_into_inputs_plus_bias = sum_weights_into_inputs + bias;		// pseudo bias
  
  
  always @ (*)
	begin
		if (sum_weights_into_inputs_plus_bias > 0)
			begin
				activation_value = sum_weights_into_inputs_plus_bias + (sum_weights_into_inputs_plus_bias * sum_weights_into_inputs_plus_bias);
			end
			
		else if (-2 <= sum_weights_into_inputs_plus_bias <= 0)
			begin
				activation_value = sum_weights_into_inputs_plus_bias + ((sum_weights_into_inputs_plus_bias * sum_weights_into_inputs_plus_bias) << 4 );				
			end
			
		else
			begin
				activation_value = 0;
			end	
	end	


endmodule		

