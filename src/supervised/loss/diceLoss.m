function [loss,gradients] = diceLoss(Y,T)

numClasses = size(Y, 3);
    
    % Initialize accumulators for loss and gradient
    diceLossSum = 0;
    gradient = zeros(size(Y), 'like', Y);
    
    % Loop over each class
    for c = 1:numClasses
        % Extract the c-th class predictions and true labels
        Y_c = Y(:,:,c,:);
        T_c = T(:,:,c,:);
        
        % Compute intersection and union
        intersection = sum(Y_c .* T_c, 'all');
        area = sum(Y_c + T_c, 'all');
        
        % Compute Dice coefficient
        diceCoeff = (2 * intersection) / (area + 1e-7);
        
        % Accumulate Dice loss
        diceLossSum = diceLossSum + (1 - diceCoeff);
        
        % Compute the gradient for this class
        grad_c = (2 * (T_c .* area - intersection)) / (area^2 + 1e-7);
        
        % Accumulate the gradient
        gradient(:,:,c,:) = grad_c;
    end
    
    % Average Dice loss across all classes
    loss = diceLossSum / numClasses;
    
    % Average gradient across all classes
    gradients = gradient / numClasses;

end
