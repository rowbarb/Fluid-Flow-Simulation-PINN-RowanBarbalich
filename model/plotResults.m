function plotResults(net, Xn, Yn, X, Y)
% Xn, Yn : normalized inputs (N x 1)
% X, Y   : original (un-normalized) grid (N x 1) for reshaping

dlX = dlarray(Xn.',"CB");
dlY = dlarray(Yn.',"CB");

pred = forward(net, cat(1,dlX,dlY));clc
u(~isfinite(u)) = 0;
v(~isfinite(v)) = 0;
p(~isfinite(p)) = 0;

u = extractdata(pred(1,:)).';
v = extractdata(pred(2,:)).';
p = extractdata(pred(3,:)).';

% reshape to grid
Nx = numel(unique(X));
Ny = numel(unique(Y));
U = reshape(u,Ny,Nx);
V = reshape(v,Ny,Nx);
P = reshape(p,Ny,Nx);
Xgrid = reshape(X,Ny,Nx);
Ygrid = reshape(Y,Ny,Nx);

figure;
subplot(1,2,1)
quiver(Xgrid, Ygrid, U, V)
title('PINN Velocity Field')
xlabel('x'); ylabel('y');

subplot(1,2,2)
contourf(Xgrid, Ygrid, P,20)
title('PINN Pressure Field')
colorbar
end
