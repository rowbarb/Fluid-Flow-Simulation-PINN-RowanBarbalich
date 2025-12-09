% R. Barbalich, Y. Huang, N. Lao, and J. Porco
% ME635 Modeling and Simulation
% Stevens Institute of Technology
% 2D Steady Incompressible Flow PINN
% December 9, 2025

%% load training/test data (from DeepCFD https://github.com/mdribeiro/DeepCFD) 
    
    % dataset dimensions and test samples
    Nx = 172; Ny = 79; Nc = 3; Ns = 981;

    % reserve ~25% of training set for validation
    nTest = floor(Ns*0.25);

    % physical constants
    L = 0.260; % [m]
    D = 0.120; % [m]
    rho = 1000; % [kg/m^3]
    nu = 1e-4; % [m^2/s]
    Uin = 0.1; % [m/s]

    % check if data is already loaded
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

    % nondimensionalize data
    P_dynamic = rho * Uin^2;
    dataY(:,:,1,:) = dataY(:,:,1,:)/Uin; % normalize to inlet velocity
    dataY(:,:,2,:) = dataY(:,:,2,:)/Uin;
    dataY(:,:,3,:) = dataY(:,:,3,:)/P_dynamic; % normalize to dynamic pressure

    % setup training on GPU if available
    if canUseGPU % store data in gpuArray
        dlX = dlarray(gpuArray(dataX),'SSCB'); % dlarrays for automatic differentiation
        dlY = dlarray(gpuArray(dataY),'SSCB');
    else % otherwise normal dlarray
        dlX = dlarray(dataX,'SSCB');
        dlY = dlarray(dataY,'SSCB');
    end

    % remove nTest samples from dataset for validation
    testIdx = randperm(Ns); testIdx = testIdx(1:nTest); % take first nTest samples
    trainIdx = true(1,Ns); trainIdx(testIdx) = false; % training indices
    testDataX = dlX(:,:,:,testIdx); % take samples from training data
    testDataY = dlY(:,:,:,testIdx); % take outputs for comparison
    dlX = dlX(:,:,:,trainIdx); % remove samples from training data
    dlY = dlY(:,:,:,trainIdx);
    Ns = Ns - nTest;

%% initialize PINN

    PINN = initializePINN(Nx,Ny,Nc);

%% train neural network

    numEpochs = 250; % number of passes o
    szBatch = 32; % samples per batch
    [PINN,cvg] = trainPINN(PINN,Nx,Ny,Ns,dlX,dlY,numEpochs,szBatch,@lossFcn,L,D,nu,Uin);

%% validation
    
    % use model on test data and plot 10 comparisons
    testDataOut = zeros(Nx,Ny,Nc,nTest);
    testDataX = gather(extractdata(testDataX)); % convert from GPUarray for plotting
    testDataY = gather(extractdata(testDataY));
    for i = 1:nTest
        testDataOut(:,:,:,i) = forward(PINN,testDataX(:,:,:,i)); % run model for each test sample
        if i <= 10 % plot first 10 and compare to ground truth
            plotXY(testDataX(:,:,:,i),testDataY(:,:,:,i),Nx,Ny, ...
                ['Test Sample ' num2str(i) ' (' num2str(testIdx(i)) '): Ground Truth Data'])
            plotXY(testDataX(:,:,:,i),testDataOut(:,:,:,i),Nx,Ny, ...
                ['Test Sample ' num2str(i) ' (' num2str(testIdx(i)) '): Model Output'])
        end
    end

    % compute average divergence of training and predicted velocity fields
    trainingDiv = avgDiv(dlX,dlY,Nx,Ny,Ns,L,D)
    testDiv = avgDiv(testDataX,testDataOut,Nx,Ny,nTest,L,D)

    % plot error heatmap for sample 5 and average across all samples
    uTest = sqrt(testDataY(:,:,1,:).^2 + testDataY(:,:,2,:).^2);
    uOut = sqrt(testDataOut(:,:,1,:).^2 + testDataOut(:,:,2,:).^2);
    uError = abs(uTest - uOut); % = abs res. velocity error / Uin
    uErrorAvg = mean(uError,4); % average error at each cell across test set
    uError5 = uError(:,:,5); % cell-wise error for sample 5

    % average error plot
    figure()
    imagesc(uErrorAvg')
    title('Average Error Heatmap')
    xlabel('Nx')
    ylabel('Ny')
    axis equal
    xlim([0 Nx])
    ylim([0 Ny])
    clim([0 0.15])
    cb = colorbar;
    cb.Label.String = '\deltav / U_{in}';

    % sample 5 error plot
    figure()
    imagesc(uError5')
    title('Sample 5 Error Heatmap')
    xlabel('Nx')
    ylabel('Ny')
    axis equal
    xlim([0 Nx])
    ylim([0 Ny])
    clim([0 0.15])
    cb = colorbar;
    cb.Label.String = '\deltav / U_{in}';

    % convergence plot
    figure()
    title('Error vs. Training Epoch')
    xlabel('Epoch')
    yyaxis left
    plot(1:1:numEpochs,cvg(:,4))
    ylabel('Velocity L_2 Error')
    yyaxis right
    plot(1:1:numEpochs,cvg(:,1))
    yline(trainingDiv,'LineStyle','--','Color','red', ...
        'Label','Average Training Set Divergence (absolute)')
    ylabel('Mean Divergence')
