function loss = (Y, T)

    [~, max_idx] = max(Y, [], 3);
    % Create a matrix of zeros with the same size as A
    Y = zeros(size(Y));
    % Set the positions of the maximum values to 1 in the corresponding channel
    for i = 1:size(Y,3)
        % Logical mask where the maximum occurs in the i-th channel
        mask = (max_idx == i);
        % Set those positions to 1 in the corresponding channel of B
        Y(:,:,i) = mask;
    end

    % Compute the Generalized Dice loss (assuming you have a function for it)
    
    dice_loss = zeros(1,size(Y,3));
    for  K = 1:(size(O,3))
         dice_loss(K)= 1 - generalizedDice(single(Y(:,:,K)), single(T(:,:,K)));
    end

    % Compute the final loss as mean of 1 - dice loss
    loss = mean(dice_loss);
end