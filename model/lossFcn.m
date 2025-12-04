function [loss,grad,lossDisp] = lossFcn(neuralNet,dlX,dlY,Nx,Ny,L,D,rho,nu,Uin,w_phys,w_bc,w_UV,w_P)
%% compute the loss and its gradient for a given input

    % forward pass
    Y = forward(neuralNet, dlX); 
    U = Y(:,:,1);
    V = Y(:,:,2);
    P = Y(:,:,3);
    % X = extractdata(dlX);

    % grid spacing
    Nx = size(Y,1);
    Ny = size(Y,2);
    dx = L / (Nx - 1);
    dy = D / (Ny - 1);

    % governing equation residuals
        
        % allocate arrays
        dUdx = zeros(Nx,Ny);
        dVdx = zeros(Nx,Ny);
        dUdy = zeros(Nx,Ny);
        dVdy = zeros(Nx,Ny);
        dPdx = zeros(Nx,Ny);
        dPdy = zeros(Nx,Ny);
        dUdxx = zeros(Nx,Ny);
        dVdxx = zeros(Nx,Ny);
        dUdyy = zeros(Nx,Ny);
        dVdyy = zeros(Nx,Ny);

        % 2nd order 1st derivative FDM for boundary points
        dUdx(1,:) = (-3*U(1,:) + 4*U(2,:) - U(3,:)) / (2*dx);
        dVdx(1,:) = (-3*V(1,:) + 4*V(2,:) - V(3,:)) / (2*dx);
        dPdx(1,:) = (-3*P(1,:) + 4*P(2,:) - P(3,:)) / (2*dx);
        dUdx(end,:) = (3*U(end,:) - 4*U(end-1,:) + U(end-2,:)) / (2*dx);
        dVdx(end,:) = (3*V(end,:) - 4*V(end-1,:) + V(end-2,:)) / (2*dx);
        dPdx(end,:) = (3*P(end,:) - 4*P(end-1,:) + P(end-2,:)) / (2*dx);
        
        dUdy(:,1) = (-3*U(:,1) + 4*U(:,2) - U(:,3)) / (2*dy);
        dVdy(:,1) = (-3*V(:,1) + 4*V(:,2) - V(:,3)) / (2*dy);
        dPdy(:,1) = (-3*P(:,1) + 4*P(:,2) - P(:,3)) / (2*dy);
        dUdy(:,end) = (3*U(:,end) - 4*U(:,end-1) + U(:,end-2)) / (2*dy);
        dVdy(:,end) = (3*V(:,end) - 4*V(:,end-1) + V(:,end-2)) / (2*dy);
        dPdy(:,end) = (3*P(:,end) - 4*P(:,end-1) + P(:,end-2)) / (2*dy);

        % 2nd order 1st derivative central difference scheme for interior points
        dUdx(2:end-1,:) = (U(3:end,:)-U(1:end-2,:)) / (2*dx);
        dVdx(2:end-1,:) = (V(3:end,:)-V(1:end-2,:)) / (2*dx);
        dPdx(2:end-1,:) = (P(3:end,:)-P(1:end-2,:)) / (2*dx);

        dUdy(:,2:end-1) = (U(:,3:end)-U(:,1:end-2)) / (2*dy);
        dVdy(:,2:end-1) = (V(:,3:end)-V(:,1:end-2)) / (2*dy);
        dPdy(:,2:end-1) = (P(:,3:end)-P(:,1:end-2)) / (2*dy);

        % 2nd order 2nd derivative FDM for boundary points
        dUdxx(1,:) = (2*U(1,:) - 5*U(2,:) + 4*U(3,:) - U(4,:)) / (dx^2);
        dVdxx(1,:) = (2*V(1,:) - 5*V(2,:) + 4*V(3,:) - V(4,:)) / (dx^2);
        dUdxx(end,:) = (2*U(end,:) - 5*U(end-1,:) + 4*U(end-2,:) - U(end-3,:)) / (dx^2);
        dVdxx(end,:) = (2*V(end,:) - 5*V(end-1,:) + 4*V(end-2,:) - V(end-3,:)) / (dx^2);
        dUdyy(:,1) = (2*U(:,1) - 5*U(:,2) + 4*U(:,3) - U(:,4)) / (dy^2);
        dVdyy(:,1) = (2*V(:,1) - 5*V(:,2) + 4*V(:,3) - V(:,4)) / (dy^2);
        dUdyy(:,end) = (2*U(:,end) - 5*U(:,end-1) + 4*U(:,end-2) - U(:,end-3)) / (dy^2);
        dVdyy(:,end) = (2*V(:,end) - 5*V(:,end-1) + 4*V(:,end-2) - V(:,end-3)) / (dy^2);

        % 2nd order 2nd derivative central difference for interior points
        dUdxx(2:end-1,:) = (U(1:end-2,:) - 2*U(2:end-1,:) + U(3:end,:)) / (dx^2);
        dVdxx(2:end-1,:) = (V(1:end-2,:) - 2*V(2:end-1,:) + V(3:end,:)) / (dx^2);
        dUdyy(:,2:end-1) = (U(:,1:end-2) - 2*U(:,2:end-1) + U(:,3:end)) / (dy^2);
        dVdyy(:,2:end-1) = (V(:,1:end-2) - 2*V(:,2:end-1) + V(:,3:end)) / (dy^2);

    continuity = dUdx + dVdy;
    xmomentum = U .* dUdx + V .* dUdy + dPdx./rho - nu*(dUdxx + dUdyy);
    ymomentum = V .* dVdy + U .* dVdx + dPdy./rho - nu*(dVdxx + dVdyy);

    % gov eq. loss (MSE)
    physicsLoss = mean(continuity.^2,'all') ...
                + mean(xmomentum.^2,'all') ...
                + mean(ymomentum.^2,'all');
        
    % boundary condition loss (MSE)
    u = sqrt(U.^2 + V.^2);
    inletLoss = mean((U(1,:)-Uin).^2,'all');
    wallLoss = mean(u(2:end-1,1).^2,'all') + mean(u(2:end-1,end).^2,'all');
    outletLoss = mean(dUdx(end,:).^2,'all'); % dU/dx = 0 at outlet, implement later
    objectLoss = mean(u(dlX(:,:,2)==0).^2,'all');
    bcLoss = inletLoss + wallLoss + outletLoss + objectLoss;
    
    % data loss (MSE)
    dataLossUV = mean((Y(:,:,1:2) - dlY(:,:,1:2)).^2, 'all');
    dataLossP  = mean((Y(:,:,3) - dlY(:,:,3)).^2, 'all');

    % total loss and gradient
    loss = w_UV * dataLossUV ...
     + w_P  * dataLossP ...
     + w_bc     * bcLoss ...
     + w_phys   * physicsLoss;
    grad = dlgradient(loss,neuralNet.Learnables);
    lossDisp = [physicsLoss bcLoss dataLossUV dataLossP max(u,[],'all') max(continuity,[],'all')];
end