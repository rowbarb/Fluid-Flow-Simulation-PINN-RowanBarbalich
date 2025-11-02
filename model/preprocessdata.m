load('flowData.mat');  % loads x, y, u, v, p

% Reshape matrices into column vectors
X = x(:);
Y = y(:);
U = u(:);
V = v(:);
P = p(:);

% Normalize everything between -1 and 1
Xn = 2*(X - min(X)) / (max(X) - min(X)) - 1;
Yn = 2*(Y - min(Y)) / (max(Y) - min(Y)) - 1;
Un = 2*(U - min(U)) / (max(U) - min(U)) - 1;
Vn = 2*(V - min(V)) / (max(V) - min(V)) - 1;
Pn = 2*(P - min(P)) / (max(P) - min(P)) - 1;

% Combine inputs and outputs
inputs = [Xn, Yn];
targets = [Un, Vn, Pn];

% Save preprocessed data
save('preprocessedData.mat', 'inputs', 'targets');
