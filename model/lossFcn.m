function [loss,grad,cvg,lossDisp] = lossFcn(neuralNet,dlX,dlY,Nx,Ny,L,D,nu,Uin,w_phys,w_bc,w_UV,w_P)
%% compute the loss and its gradient for a given input

    % forward pass
    Y = forward(neuralNet, dlX); 
    U = Y(:,:,1);
    V = Y(:,:,2);
    P = Y(:,:,3);

    % prepare data
    U = min(max(U, -2.0), 2.0); % hard clamp to prevent residual explosion
    V = min(max(V, -2.0), 2.0);
    U_soft = tanh(U/2.5)*2.5; % soft clamp
    V_soft = tanh(V/2.5)*2.5;
    
    % compure parameters
    dx = L / (Nx - 1); % grid spacing
    dy = D / (Ny - 1);
    object = dlX(:,:,2)==0; % region masks
    fluid = dlX(:,:,2)==1;
    wall = dlX(:,:,2)==2;
    inlet = dlX(:,:,2)==3;
    outlet = dlX(:,:,2)==4;
    Re = Uin * D / nu; % Reynolds number

    % governing equation residuals
    [dUdx,dVdx,dUdy,dVdy,dUdxx,dVdxx,dUdyy,dVdyy,dPdx,dPdy] = derivativesUVP(U,V,P,Nx,Ny,dx,dy);

    continuity = (dUdx + dVdy).*fluid; % multiply by fluid (Nx x Ny boolean array) to remove object and walls
    xmomentum = (U .* dUdx + V .* dUdy + dPdx - (1/Re)*(dUdxx + dUdyy)).*fluid; 
    ymomentum = (V .* dVdy + U .* dVdx + dPdy - (1/Re)*(dVdxx + dVdyy)).*fluid;

    % gov eq. loss (MSE) and residuals
    % physicsLoss = mean(continuity.^2,'all') ...
    %             + mean(xmomentum.^2,'all') ...
    %             + mean(ymomentum.^2,'all');
    physicsLoss =  mean(continuity.^2,'all');
    div_mean = mean(abs(continuity),'all');
    div_max = max(abs(continuity),[],'all');
    mom_mean = sqrt(mean(xmomentum.^2 + ymomentum.^2,'all'));
        
    % boundary condition loss (MSE)
    u_res = sqrt(U_soft.^2 + V_soft.^2);
    inletLoss = mean((U_soft(inlet)-1).^2,'all'); % inlet velocity (normalized to Uin)
    wallLoss = mean(u_res(wall).^2,'all'); % no slip
    outletLoss = mean(dUdx(outlet).^2,'all'); % zero gradient
    objectLoss = mean(u_res(object).^2,'all');
    bcLoss = inletLoss + wallLoss + outletLoss + objectLoss; % prioritize no slip
    
    % data loss (MSE) and residuals
    dataLossUV = mean((Y(:,:,1:2) - dlY(:,:,1:2)).^2, 'all');
    % dataLossP  = mean((Y(:,:,3) - dlY(:,:,3)).^2, 'all');
    dataLossP = max(abs(Y(:,:,3) - dlY(:,:,3)),[],'all');
    U_L2 = sqrt(sum((Y(:,:,1:2) - dlY(:,:,1:2)).^2,'all')) / sqrt(sum(dlY(:,:,1:2).^2,'all')); % relative L2 (Frobenius) error norm of velocity error
    P_L2 = sqrt(sum((Y(:,:,3) - dlY(:,:,3)).^2,'all')) / sqrt(sum(dlY(:,:,3).^2,'all'));

    % total loss and gradient
    loss = w_UV*dataLossUV + w_P*dataLossP + w_bc*bcLoss + w_phys*physicsLoss;
    grad = dlgradient(loss,neuralNet.Learnables);
    cvg = [div_mean div_max mom_mean U_L2 P_L2];
    lossDisp = [physicsLoss bcLoss dataLossUV dataLossP max(u_res,[],'all')];
end