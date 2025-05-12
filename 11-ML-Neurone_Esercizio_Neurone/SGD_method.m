function Weight = SGD_method(Weight, input, correct_Output)
                                                                                %input 3 neurons
                                                                                
    LR = 0.9;

    N = 4;

    for k=1:N
       transposed_Input = input(k, :)';
       expected = correct_Output(k);
       lin_comb = Weight*transposed_Input;
       predicted = Sigmoid(lin_comb);                                    

       cost_function_der = expected - predicted;                               % cost_function_der is derivate of cost function
       delta = predicted*(1-predicted)*cost_function_der;                         % delta is derivate of sigmoid*cost_function 
                                                                               % derivate of the sigmoid is sigmoid*(1-sigmoid)

       dWeight = LR*delta*transposed_Input;                                    % dWeigth = LR*derivate(cost_function) = LR*(delta*input)
                                                                               % weight correction given by learning rate
                                                                               % multiplied by delta and the transposed input
       Weight(1) = Weight(1) + dWeight(1);
       Weight(2) = Weight(2) + dWeight(2);
       Weight(3) = Weight(3) + dWeight(3);
    end
end