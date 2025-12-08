function avgDiv = avgDiv(dlX,dlY,Nx,Ny,Ns,L,D)
%% compute the average divergence of velocity (absolute value) fields across all training samples
    
    X = gather(extractdata(dlX));
    Y = gather(extractdata(dlY));
    U = Y(:,:,1,:); U = squeeze(U);
    V = Y(:,:,2,:); V = squeeze(V);
    P = Y(:,:,3,:); P = squeeze(P);
    fluid = X(:,:,2,:)==1; fluid = squeeze(fluid);
    dx = L / (Nx - 1);
    dy = D / (Ny - 1);
    div = zeros(Ns,1);

    for i = 1:Ns
        [dUdx,~,~,dVdy,~,~,~,~,~,~] = derivativesUVP(U(:,:,i),V(:,:,i),P(:,:,i),Nx,Ny,dx,dy);
        div(i) = mean(abs(dUdx + dVdy).*fluid(:,:,i),'all'); % mean divergence for X(i)
    end
    
    avgDiv = sum(div,'all') / Ns;
end