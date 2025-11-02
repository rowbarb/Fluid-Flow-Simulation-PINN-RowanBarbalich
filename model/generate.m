
% Domain
Nx = 50; Ny = 50;     % grid resolution
Lx = 1; Ly = 1;       % domain size
[x, y] = meshgrid(linspace(0, Lx, Nx), linspace(0, Ly, Ny));

% Parameters
mu = 1.0;              % dynamic viscosity
dpdx = -1.0;           % pressure gradient driving the flow
H = Ly;                % channel height

% Analytical solution for steady Poiseuille flow
u = (1/(2*mu)) * dpdx * (y.^2 - y*H);   % velocity profile
v = zeros(size(u));                     % no vertical flow
p = -dpdx * x;                          % linear pressure drop

% Visuals
figure;
subplot(1,2,1)
quiver(x,y,u,v)
title('Velocity Field (u,v)')
xlabel('x'), ylabel('y')

subplot(1,2,2)
contourf(x,y,p,20)
title('Pressure Field (p)')
xlabel('x'), ylabel('y'), colorbar

% Step 5: Save data
save('flowData.mat', 'x', 'y', 'u', 'v', 'p')
