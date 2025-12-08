function [dUdx,dVdx,dUdy,dVdy,dUdxx,dVdxx,dUdyy,dVdyy,dPdx,dPdy] = derivativesUVP(U,V,P,Nx,Ny,dx,dy)
%% compute the first and second derivatives of U,V, and P

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
end