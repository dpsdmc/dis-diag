module activate_layer_2  #(
                    						 parameter weight_size = 5,
                    						 parameter bias_size = 6,

                    						 parameter number_of_hidden_neurons = 5,       // number of hidden neurons (layer 1)  
                    						 parameter number_of_outputs = 2,
                    						 parameter number_of_weights = number_of_hidden_neurons * number_of_outputs,       // #weights = #outputs . #hidden_units
                    						 parameter hidden_neuron_size = 54,            // sum(inputs.weights1)                    // recheck
                    						 parameter activation_size = 131               // re-check
						 						 ) 

					   (
						
						input clk,
					   input wire signed  [(number_of_hidden_neurons  *  hidden_neuron_size) -1 : 0] hidden_input_vector,
					   input wire signed [(number_of_weights  *  weight_size)  -1 : 0] weight_vector, 				  // weights for all the hidden neurons   
					   input wire signed [(bias_size  *  number_of_outputs)  -1 : 0] bias_vector, 			

					   output wire signed [(activation_size  *  number_of_outputs)  -1 : 0] activation_vector 			
					   );

  
             //     wire signed [(number_of_weights  *  (weight_size + hidden_neuron_size))  -1 : 0] inputs_into_weights_vector ;        // recheck


  
    genvar j;								
	generate											
		for(j=0; j < number_of_outputs; j = j+1)					                // number of inputs = 8 ... 8 weights per output neuron --> j ... 0 to 7
  			begin: x_w_layer_2	                                        						// one weight --> 8 bits --> j*(wt_size) +: (wt_size)							

weight_vector_separater_layer_2   a1_w2_inst 
                                
                        (
								 .clk(clk), 
                         .hidden_input_vector(hidden_input_vector),
                         .weight_vector(weight_vector[(( (j+1) * number_of_hidden_neurons * weight_size) -1) : (j * number_of_hidden_neurons * weight_size)]),
                         .bias(bias_vector[ ((j+1) * bias_size) -1 : (j * bias_size)]),

                         .activation_value(activation_vector[ ((j+1) * activation_size) -1 : (j * activation_size)])
                         
                        );

			end
	endgenerate




endmodule			

