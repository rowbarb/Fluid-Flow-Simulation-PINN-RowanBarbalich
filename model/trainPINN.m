function [PINN,cvgOut] = trainPINN(PINN,Nx,Ny,Ns,dlX,dlY,numEpochs,szBatch,lossFcn,L,D,nu,Uin)
%% train PINN using adam optimizer and custom loss function

    % compute number of batches
    numBatches = floor(Ns / szBatch); % batches to cover all training data

    % adam optimizer parameters (adaptive moment estimation)
    averageGrad = [];
    averageSqGrad = [];

    % initialize training monitor and convergence array
    monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Epoch");
    cvgOut = zeros(numEpochs,5);
    
    % train neural network
    for epoch = 1:numEpochs
        if epoch <= 50 % weight and learning rate schedule
            w_UV = 1.0;  w_P = 0.05;  w_bc = 0.01;  w_phys = 0;
            learnRate = 1e-4;
        elseif epoch <= 150
            w_UV = 1.0;  w_P = 0.1;   w_bc = 0.1;   w_phys = 1e-4;
            learnRate = 1e-4;
        else
            w_UV = 1.0;  w_P = 0.15;  w_bc = 0.1;   w_phys = 1e-3;
            learnRate = 5e-5;
        end
        
        shuffle = randperm(Ns); % shuffle data each epoch

        for b = 1:numBatches % batch training    
            sumGrad = []; % reset each batch
            sumLoss = 0;
            batchStart = (b-1)*szBatch + 1; 
            batchEnd = min(b*szBatch, Ns); % last batch <= szBatch
            batchSize = batchEnd - batchStart + 1;
            for i = 1:batchSize
                samples = shuffle(batchStart + i - 1);
                Xsample = dlX(:,:,:,samples);
                Ysample = dlY(:,:,:,samples);
                [loss, grad, cvg, lossDisp] = dlfeval(lossFcn,PINN,Xsample,Ysample,...
                    Nx,Ny,L,D,nu,Uin,w_phys,w_bc,w_UV,w_P); % compute loss and gradients
                if i == 1 % sum sample losses and loss gradients
                    sumGrad = grad;
                    sumLoss = extractdata(loss);
                else
                    for layer = 1:size(grad, 1)
                        sumGrad.Value{layer} = sumGrad.Value{layer} + grad.Value{layer};
                    end
                    sumLoss = sumLoss + extractdata(loss);
                end
            end
            % average loss and gradients across batch
            for layer = 1:size(sumGrad, 1)
                sumGrad.Value{layer} = sumGrad.Value{layer} / batchSize;
            end
            avgLoss = sumLoss / batchSize;
            % update network with averaged gradients
            iteration = (epoch-1)*numBatches + b;
            [PINN, averageGrad, averageSqGrad] = adamupdate(PINN, sumGrad, ...
                averageGrad, averageSqGrad, iteration, learnRate);
        end
        % monitor training
        recordMetrics(monitor, epoch, Loss=avgLoss);
        updateInfo(monitor, 'Epoch', [num2str(epoch) ' of ' num2str(numEpochs)]);
        monitor.Progress = 100 * epoch/numEpochs;
        cvgOut(epoch,:) = cvg;
        if mod(epoch,10)==0 || epoch==1
            disp(['Epoch ' num2str(epoch)]);
            disp(['     Physics loss: ' num2str(lossDisp(1))]);
            disp(['     BC loss: ' num2str(lossDisp(2))]);
            disp(['     Velocity loss: ' num2str(lossDisp(3)) ', L2 error: ' num2str(cvg(4))]);
            disp(['     Pressure loss: ' num2str(lossDisp(4)) ', L2 error: ' num2str(cvg(5))]);
            disp(['     Max velocity: ' num2str(lossDisp(5))])
            disp(['     Max divergence: ' num2str(cvg(2)) ', mean: ' num2str(cvg(1))]);
            disp(['     Momentum residual: ' num2str(cvg(3))])
        end
    end
end