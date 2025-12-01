function [loss,grad] = lossFcn(neuralNet,X)
%% compute the loss and its gradient for a given input    
    Y = forward(neuralNet,X); % compute Y(X) from network
    Ux = Y(1); % for readability
    Uy = Y(2);
    P = Y(3);

    % conservation of mass
    % (div(U) = 0)

    % conservation of momentum
        % in x-direction:

        % in y-direction:

    loss = [];
    grad = dlgradient(loss, neuralNet.Learnables);
end