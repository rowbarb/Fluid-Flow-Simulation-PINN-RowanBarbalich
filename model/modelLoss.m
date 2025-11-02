function [loss,gradients] = modelLoss(net, dlX, dlY, dlTargets, params)
% modelLoss: computes total PINN loss + gradients

% forward pass
XY = cat(1, dlX, dlY);         % 2 x N
pred = forward(net, XY);       % 3 x N

u = pred(1,:);    % 1 x N
v = pred(2,:);    % 1 x N
p = pred(3,:);    % 1 x N

% ========= physics part =========
mu  = params.mu;
rho = params.rho;

% we need gradients wrt x and y
% tell MATLAB these are independent variables
x = dlX;
y = dlY;

% du/dx
dux = dlgradient(sum(u), x, 'EnableHigherDerivatives', true);
% du/dy
duy = dlgradient(sum(u), y, 'EnableHigherDerivatives', true);
% dv/dx
dvx = dlgradient(sum(v), x, 'EnableHigherDerivatives', true);
% dv/dy
dvy = dlgradient(sum(v), y, 'EnableHigherDerivatives', true);

% second derivs for viscous term
duxx = dlgradient(sum(dux), x, 'EnableHigherDerivatives', true);
duyy = dlgradient(sum(duy), y, 'EnableHigherDerivatives', true);

dvxx = dlgradient(sum(dvx), x, 'EnableHigherDerivatives', true);
dvyy = dlgradient(sum(dvy), y, 'EnableHigherDerivatives', true);

% pressure grads
dpx = dlgradient(sum(p), x, 'EnableHigherDerivatives', true);
dpy = dlgradient(sum(p), y, 'EnableHigherDerivatives', true);

% continuity: du/dx + dv/dy = 0
contResidual = dux + dvy;

% x-momentum (steady):
% u*du/dx + v*du/dy = -(1/rho) dp/dx + mu*(d2u/dx2 + d2u/dy2)
momX = u.*dux + v.*duy + (1/rho)*dpx - mu*(duxx + duyy);

% y-momentum (optional, but we include)
momY = u.*dvx + v.*dvy + (1/rho)*dpy - mu*(dvxx + dvyy);

% physics loss = MSE of residuals
L_cont = mse(contResidual);
L_momX = mse(momX);
L_momY = mse(momY);

physicsLoss = L_cont + L_momX + L_momY;

% ========= data loss (to match synthetic data) =========
% dlTargets is 3 x N: [u; v; p]
dataLoss = mse(pred - dlTargets);

% total loss (tune weights if needed)
loss = physicsLoss + 0.1*dataLoss;

% compute gradients wrt network learnables
gradients = dlgradient(loss, net.Learnables);
end

function out = mse(x)
out = mean(x.^2,'all');
end
