% load training/test data (from DeepCFD https://github.com/mdribeiro/DeepCFD) 
    
    % dataset dimensions
    Nx = 172; Ny = 79; Nc = 3; Ns = 981;

    % physical constants
    L = 0.260; % [m]
    D = 0.120; % [m]
    rho = 1000; % [kg/m^3]
    nu = 1e-4; % [m^2/s]
    Uin = 0.1; % [m/s]

    % check if data is already loaded
    % if ~exist('dataX','var') && ~exist('dataY','var')
        % check if training data is already in .mat format
        if isfile('../data/dataX.mat') && isfile('../data/dataY.mat')
            dataX = load('../data/dataX.mat'); dataX = dataX.dataX;
            dataY = load('../data/dataY.mat'); dataY = dataY.dataY;
        else % otherwise read .pkl files (from https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1)
            pickle = py.importlib.import_module('pickle'); % import pickle module (from python)
            X = py.open('../data/dataX.pkl','rb'); % open data
            Y = py.open('../data/dataY.pkl','rb');
            X_py = pickle.load(X); % returns numpy nd array
            Y_py = pickle.load(Y);
            X.close();
            Y.close();
            dataX = numpy2mat(X_py,Nx,Ny,Nc,Ns); % convert data to MATLAB array
            dataY = numpy2mat(Y_py,Nx,Ny,Nc,Ns); 
            save('../data/dataX.mat','dataX'); % save for future
            save('../data/dataY.mat','dataY');
        end
    % end

    % nondimensionalize data
    % Umax = max(dataY(:,:,1,:),[],'all');
    % Vmax = max(dataY(:,:,2,:),[],'all');
    P_dyn = rho * Uin^2;
    dataY(:,:,1,:) = dataY(:,:,1,:)/Uin;
    dataY(:,:,2,:) = dataY(:,:,2,:)/Uin;
    dataY(:,:,3,:) = dataY(:,:,3,:)/P_dyn;

    % setup training on GPU if available
    if canUseGPU % store data in gpuArray
        dlX = dlarray(gpuArray(dataX),'SSCB'); % dlarrays for automatic differentiation
        dlY = dlarray(gpuArray(dataY),'SSCB');
    else % otherwise normal dlarray
        dlX = dlarray(dataX,'SSCB');
        dlY = dlarray(dataY,'SSCB');
    end

    % visualize random sample from dataset
    random = ceil(Ns*rand);
    plotXY(dataX(:,:,:,random),dataY(:,:,:,random),Nx,Ny,['Sample #' num2str(random)]);

    % % compare output to sample
    % 
    % Y_out = forward(JRNY,dlX(:,:,:,random));
    % Y_out = extractdata(Y_out);
    % plotXY(dataX(:,:,:,random),squeeze(Y_out),Nx,Ny,'Output');

% create neural network architecture

    % input and output size
    szIn = [Nx Ny Nc]; % object SDF, masks, wall SDF fields
    szOut = [Nx Ny Nc]; % u, v, P fields

    % architecture
    layers = [
        imageInputLayer([Nx Ny Nc],Normalization="none")
        convolution2dLayer(3,64,"Padding","same")
        tanhLayer
        convolution2dLayer(3,128,"Padding","same")
        tanhLayer
        convolution2dLayer(3,512,"Padding","same")
        tanhLayer
        convolution2dLayer(3,128,"Padding","same")
        tanhLayer
        convolution2dLayer(3,64,"Padding","same")
        tanhLayer
        convolution2dLayer(1,3,"Padding","same")
    ];
    JRNY = dlnetwork(layers); % create network 
    if canUseGPU, JRNY = dlupdate(@gpuArray,JRNY); end % train on GPU
 
% train neural network

    % define hyperparameters
    szBatch = 8; % number of samples per batch (iteration)
    numEpochs = 300; % number of epochs
    numBatches = floor(Ns / szBatch); % batches to cover all training data

    % adam optimizer parameters (adaptive moment estimation)
    learnRate = 1e-4;
    averageGrad = [];
    averageSqGrad = [];

    % initialize training monitor
    monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Epoch");
    
    % train neural network
    for epoch = 1:numEpochs
        if epoch <= 20 % adjust weights as training progresses
            w_UV = 1e6;  w_P = 1e5;  w_bc = 1e4;   w_phys = 1e-8;
        elseif epoch <= 50
            w_UV = 1e5;  w_P = 1e4;  w_bc = 5e4;   w_phys = 1e-6;
        elseif epoch <= 100
            w_UV = 1e4;  w_P = 1e3;  w_bc = 1e5;   w_phys = 1e-4;
        elseif epoch <= 200
            w_UV = 5e3;  w_P = 5e2;  w_bc = 1e5;   w_phys = 1e-3;
        else
            w_UV = 1e3;  w_P = 1e2;  w_bc = 1e5;   w_phys = 1e-2; 
        end
        % learnRate = 1e-4 + 9e-4 * (epoch > 50);  % change to 1e-3 later
        for b = 1:numBatches
            batch = (b-1)*szBatch + (1:szBatch); % current batch indices
            for i = 1:length(batch)
                Xbatch = dlX(:,:,:,batch(i)); % get X and Y data
                Ybatch = dlY(:,:,:,batch(i)); 
                [loss,grad,lossDisp] = dlfeval(@lossFcn,JRNY,Xbatch,Ybatch,Nx,Ny,L,D,rho,nu,Uin,w_phys,w_bc,w_UV,w_P); % compute loss and gradients
                [JRNY,mp,vp] = adamupdate(JRNY,grad,averageGrad,averageSqGrad,epoch,learnRate); % update neural network
            end
        end
        disp(['Epoch ' num2str(epoch)]);
        disp(['     Physics loss: ' num2str(lossDisp(1))])
        disp(['     BC loss: ' num2str(lossDisp(2))])
        disp(['     Velocity loss: ' num2str(lossDisp(3))])
        disp(['     Pressure loss: ' num2str(lossDisp(4))])
        disp(['     Max velocity: ' num2str(lossDisp(5))])
        disp(['     Max divergence: ' num2str(lossDisp(6))])

        recordMetrics(monitor,epoch,Loss=loss);
        updateInfo(monitor,'Epoch',[num2str(epoch) ' of ' num2str(numEpochs)]);
        monitor.Progress = 100 * epoch/numEpochs;
        
    end

% compare output to sample

    Y_out = forward(JRNY,dlX(:,:,:,random));
    Y_out = extractdata(Y_out);
    plotXY(dataX(:,:,:,random),squeeze(Y_out),Nx,Ny,'Output');