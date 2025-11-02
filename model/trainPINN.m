function net = trainPINN(net, inputs, targets)
% trainPINN: trains the PINN network for 2D steady channel flow
% inputs: N x 2   (normalized x,y)
% targets: N x 3  (normalized u,v,p)

% === Convert to dlarray, transpose to (features x batch) ===
X = inputs(:,1).';
Y = inputs(:,2).';
T = targets.';   % 3 x N

dlX = dlarray(X,"CB");   % C=features, B=batch
dlY = dlarray(Y,"CB");
dlT = dlarray(T,"CB");

% === Physical parameters ===
params.mu  = 1.0;
params.rho = 1.0;

% === Training hyperparameters ===
numEpochs = 2000;
learnRate = 5e-4;
gradDecay = 0.9;
sqGradDecay = 0.999;

trailingAvg = [];
trailingAvgSq = [];

fprintf('\n=== Starting PINN Training ===\n');

for epoch = 1:numEpochs

    % ---- Forward + backward pass ----
    [loss,grads] = dlfeval(@modelLoss, net, dlX, dlY, dlT, params);

    % ---- Handle NaN or Inf loss ----
    if ~isfinite(extractdata(loss))
        warning('⚠️  NaN or Inf loss detected at epoch %d. Resetting network...', epoch);
        net = createPINN();        % reset weights to avoid divergence
        continue
    end

    % ---- Gradient clipping (prevent explosion) ----
    for i = 1:numel(grads.Value)
        g = grads.Value{i};
        if any(isnan(g(:))) || any(~isfinite(g(:)))
            grads.Value{i} = zeros(size(g)); % skip invalid gradients
        else
            grads.Value{i} = max(min(g,1),-1);
        end
    end

    % ---- Parameter update ----
    [net,trailingAvg,trailingAvgSq] = adamupdate(net,grads, ...
        trailingAvg,trailingAvgSq,epoch,learnRate,gradDecay,sqGradDecay);

    % ---- Progress output ----
    if mod(epoch,100)==0
        fprintf('Epoch %d, Loss = %.6f\n', epoch, extractdata(loss));
    end
end

fprintf('✅ Training complete.\n');
end
