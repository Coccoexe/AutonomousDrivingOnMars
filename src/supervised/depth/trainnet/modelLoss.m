function loss = modelLoss(Y, T)
    
    losses = zeros(1,size(Y,4));

    for j=1:size(Y,4)
        Y_j = squeeze(Y(:,:,:,j));
        T_j = squeeze(T(:,:,:,j));

        [~, max_idx] = max(Y_j, [], 3);
        % Create a matrix of zeros with the same size as A
        Y_j(:) = 0;
        % Set the positions of the maximum values to 1 in the corresponding channel
        for i = 1:size(Y_j,3)
            % Logical mask where the maximum occurs in the i-th channel
            mask = (max_idx == i);
            % Set those positions to 1 in the corresponding channel of B
            Y_j(:,:,i) = mask;
        end
    
        % Compute the Generalized Dice loss (assuming you have a function for it)
        
        dice_loss = zeros(1,size(Y_j,3));
        for  K = 1:(size(Y_j,3))
             dice_loss(K)= 1 - generalizedDice(Y_j(:,:,K), T_j(:,:,K));
        end
    
        % Compute the final loss as mean of 1 - dice loss
        losses(j) = mean(dice_loss);
    end
    loss = mean(losses);
end