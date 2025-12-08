function avgDiv = avgDiv(X,Y,Nx,Ny,Ns,L,D)
%% compute the average divergence of velocity (absolute value) fields across all training samples
    if isa(X,'dlarray') == 1 && isa(Y,'dlarray') == 1 % convert to double if necessary
        X = gather(extractdata(X));
        Y = gather(extractdata(Y));
    end
    
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