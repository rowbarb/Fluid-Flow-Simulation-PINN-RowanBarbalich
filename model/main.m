% load training/test data (from DeepCFD https://github.com/mdribeiro/DeepCFD) 
    
    % dataset dimensions
    Nx = 172; Ny = 79; Nc = 3; Ns = 981;

    % physical constants
    L = 0.260; % [m]
    D = 0.120; % [m]
    rho = 1000; % [kg/m^3]
    nu = 1e-4; % [m^2/s]

    % check if data is already loaded
    if ~exist('dlX','var') && ~exist('dlY','var')

        % check if training data is already in .mat format
        if isfile('../data/dlX.mat') && isfile('../data/dlY.mat')
            dlX = load('../data/dlX.mat');
            dlY = load('../data/dlY.mat');
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
            dlX = dlarray(gpuArray(dataX),'SSCB'); % store in dl arrays for automatic differentiation
            dlY = dlarray(gpuArray(dataY),'SSCB'); % gpu arrays for faster training
        end
    end

    % visualize random sample from dataset
    random = ceil(Ns*rand)
    plotXY(dataX(:,:,:,random),dataY(:,:,:,random),Nx,Ny,['Sample #' num2str(random)]);

% create neural network architecture

    % input and output size
    szIn = [Nx Ny Nc]; % object SDF, masks, wall SDF fields
    szOut = [Nx Ny Nc]; % u, v, P fields

    % number of hidden units
    numHidden = 128;

    % architecture
    layers = [
        imageInputLayer(szIn, Normalization="none", Name="input")

        % ---- Block 1 ----
        convolution2dLayer(3,32,Padding="same",Name="conv1")
        reluLayer(Name="relu1")

        % ---- Block 2 ----
        convolution2dLayer(3,64,Padding="same",Name="conv2")
        reluLayer(Name="relu2")

        % ---- Bottleneck ----
        convolution2dLayer(3,64,Padding="same",Name="conv3")
        reluLayer(Name="relu3")

        % ---- Block 3 ----
        convolution2dLayer(3,32,Padding="same",Name="conv4")
        reluLayer(Name="relu4")

        % ---- Output ----
        convolution2dLayer(1,3,Padding="same",Name="output")  % (u, v, p)
    ];

    JRNY = dlnetwork(layers);
    JRNY = dlupdate(@gpuArray,JRNY); % train on GPU
 
% train neural network

    % define hyperparameters
    szBatch = 8; % number of samples per batch;
    numEpochs = 500; % number of epochs
    numBatches = floor(Ns / szBatch);

    % ADAM optimizer parameters (stochastic gradient descent with momentum)
    initialLR = 0.001; % initial learning rate
    mp = []; % mean
    vp = []; % variance
    
    % train neural network
    for epoch = 1:numEpochs
        for b = 1:numBatches
    
            batch = (b-1)*szBatch + (1:szBatch);
            
            Xbatch = dlX(:,:,:,batch);
            Ybatch = dlY(:,:,:,batch);
    
            [loss,grad] = dlfeval(@lossFcn,JRNY,Xbatch,Ybatch,L,D,rho,nu,Nx,Ny); % compute loss and gradients
            [JRNY,mp,vp] = adamupdate(JRNY,grad,mp,vp,epoch,initialLR); % update neural network
        end
        
        disp(['Epoch ' num2str(epoch) ': loss = ' num2str(extractdata(loss))]);
    end

    Y_out = forward(JRNY,dlX(:,:,:,random));
    Y_out = extractdata(Y_out);
    plotXY(dataX(:,:,:,random),Y_out(:,:,:,1),Nx,Ny,'Output');