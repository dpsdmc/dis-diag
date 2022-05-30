module x_w_calculator_layer_1  #(
                    						 parameter weight_size = 7,
                    						 parameter input_size = 11
						 						 ) 

					   (
					   input wire signed [input_size - 1 : 0] x,
						
					   input wire signed   [weight_size -1 : 0] weight, 				      	  
					 
					   output wire signed  [(weight_size + input_size) -1 : 0] input_into_weight 		
					   );                                                       


//  assign input_into_weight = ((x == 0)  ||   (weight == 0)  ) ? 0   : (x * weight);  			 

  assign input_into_weight = (x * weight);  			 // place ICL..something



endmodule		
