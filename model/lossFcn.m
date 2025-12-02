function [loss,grad] = lossFcn(neuralNet,dlX,dlY,L,D,rho,nu,Nx,Ny)
%% compute the loss and its gradient for a given input

    % forward pass
    Y = forward(neuralNet, dlX); 
    U = Y(:,:,1);
    V = Y(:,:,2);
    P = Y(:,:,3);

    % Grid spacing
    Nx = size(Y,1);
    Ny = size(Y,2);
    dx = L / (Nx - 1);
    dy = D / (Ny - 1);

    % first derivatives (central differences)
    dUdx = (U(3:end,2:end-1) - U(1:end-2,2:end-1)) / (2*dx);
    dUdy = (U(2:end-1,3:end) - U(2:end-1,1:end-2)) / (2*dy);
    
    dVdx = (V(3:end,2:end-1) - V(1:end-2,2:end-1)) / (2*dx);
    dVdy = (V(2:end-1,3:end) - V(2:end-1,1:end-2)) / (2*dy);
    
    dPdx = (P(3:end,2:end-1) - P(1:end-2,2:end-1)) / (2*dx);
    dPdy = (P(2:end-1,3:end) - P(2:end-1,1:end-2)) / (2*dy);

    % second derivatives (Laplacian terms)
    Ui = U(2:end-1, 2:end-1); % interior points for central differences
    Vi = V(2:end-1, 2:end-1);

    dUdxx = (U(3:end,2:end-1) - 2*Ui + U(1:end-2,2:end-1)) / dx^2;
    dUdyy = (U(2:end-1,3:end) - 2*Ui + U(2:end-1,1:end-2)) / dy^2;

    dVdxx = (V(3:end,2:end-1) - 2*Vi + V(1:end-2,2:end-1)) / dx^2;
    dVdyy = (V(2:end-1,3:end) - 2*Vi + V(2:end-1,1:end-2)) / dy^2;

    % governing equation residuals
    continuity = dUdx + dVdy;

    xmom = Ui .* dUdx + Vi .* dUdy + dPdx./rho - nu*(dUdxx + dUdyy);
    ymom = Vi .* dVdy + Ui .* dVdx + dPdy./rho - nu*(dVdxx + dVdyy);

    % gov eq. loss (mean-square)
    physicsLoss = mean(continuity.^2,'all') ...
                + mean(xmom.^2,'all') ...
                + mean(ymom.^2,'all');

    % data loss (vs. training data)
    dataLoss = mean((Y - dlY).^2,'all');

    % total loss (biased towards physics)
    loss = physicsLoss + 0.1*dataLoss;

    grad = dlgradient(loss,neuralNet.Learnables);
end