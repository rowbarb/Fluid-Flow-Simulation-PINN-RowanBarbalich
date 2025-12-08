function PINN = initializePINN(Nx,Ny,Nc)
%% initialize PINN (U-net architecture with skip connections)
    % encoder
    enc1 = [
        imageInputLayer([Nx Ny Nc],'Name','input','Normalization','none')
        convolution2dLayer(3,64,'Padding','same','Name','enc1_conv1')
        tanhLayer('Name','enc1_tanh1')
        convolution2dLayer(3,64,'Padding','same','Name','enc1_conv2')
        tanhLayer('Name','enc1_tanh2')
        maxPooling2dLayer(2,'Stride',2,'Name','enc1_pool')
        ];
    
    enc2 = [
        convolution2dLayer(3,128,'Padding','same','Name','enc2_conv1')
        tanhLayer('Name','enc2_tanh1')
        convolution2dLayer(3,128,'Padding','same','Name','enc2_conv2')
        tanhLayer('Name','enc2_tanh2')
        maxPooling2dLayer(2,'Stride',2,'Name','enc2_pool')
        ];
    
    enc3 = [
        convolution2dLayer(3,256,'Padding','same','Name','enc3_conv1')
        tanhLayer('Name','enc3_tanh1')
        convolution2dLayer(3,256,'Padding','same','Name','enc3_conv2')
        tanhLayer('Name','enc3_tanh2')
        maxPooling2dLayer(2,'Stride',2,'Name','enc3_pool')
        ];
    
    % bottleneck
    bottleneck = [
        convolution2dLayer(3,512,'Padding','same','Name','bottleneck_conv1')
        tanhLayer('Name','bottleneck_tanh1')
        convolution2dLayer(3,512,'Padding','same','Name','bottleneck_conv2')
        tanhLayer('Name','bottleneck_tanh2')
        ];
    
    % decoder path with resize layers
    up3 = [
        resize2dLayer('OutputSize',[43 19],'Name','up3')
        convolution2dLayer(3,256,'Padding','same','Name','up3_conv')
        ];
    concat3 = depthConcatenationLayer(2,'Name','concat3');
    dec3 = [
        convolution2dLayer(3,256,'Padding','same','Name','dec3_conv1')
        tanhLayer('Name','dec3_tanh1')
        convolution2dLayer(3,256,'Padding','same','Name','dec3_conv2')
        tanhLayer('Name','dec3_tanh2')
        ];
    
    up2 = [
        resize2dLayer('OutputSize',[86 39],'Name','up2')
        convolution2dLayer(3,128,'Padding','same','Name','up2_conv')
        ];
    concat2 = depthConcatenationLayer(2,'Name','concat2');
    dec2 = [
        convolution2dLayer(3,128,'Padding','same','Name','dec2_conv1')
        tanhLayer('Name','dec2_tanh1')
        convolution2dLayer(3,128,'Padding','same','Name','dec2_conv2')
        tanhLayer('Name','dec2_tanh2')
        ];
    
    up1 = [
        resize2dLayer('OutputSize',[172 79],'Name','up1')
        convolution2dLayer(3,64,'Padding','same','Name','up1_conv')
        ];
    concat1 = depthConcatenationLayer(2,'Name','concat1');
    dec1 = [
        convolution2dLayer(3,64,'Padding','same','Name','dec1_conv1')
        tanhLayer('Name','dec1_tanh1')
        convolution2dLayer(3,64,'Padding','same','Name','dec1_conv2')
        tanhLayer('Name','dec1_tanh2')
        convolution2dLayer(1,3,'Padding','same','Name','output_conv')
        ];
    
    % assemble layer graph
    lgraph = layerGraph(enc1);
    lgraph = addLayers(lgraph, enc2);
    lgraph = addLayers(lgraph, enc3);
    lgraph = addLayers(lgraph, bottleneck);
    lgraph = addLayers(lgraph, up3);
    lgraph = addLayers(lgraph, concat3);
    lgraph = addLayers(lgraph, dec3);
    lgraph = addLayers(lgraph, up2);
    lgraph = addLayers(lgraph, concat2);
    lgraph = addLayers(lgraph, dec2);
    lgraph = addLayers(lgraph, up1);
    lgraph = addLayers(lgraph, concat1);
    lgraph = addLayers(lgraph, dec1);
    
    % connect encoder
    lgraph = connectLayers(lgraph,'enc1_pool','enc2_conv1');
    lgraph = connectLayers(lgraph,'enc2_pool','enc3_conv1');
    lgraph = connectLayers(lgraph,'enc3_pool','bottleneck_conv1');
    
    % connect decoder with skip connections
    lgraph = connectLayers(lgraph,'bottleneck_tanh2','up3');
    lgraph = connectLayers(lgraph,'up3_conv','concat3/in1');
    lgraph = connectLayers(lgraph,'enc3_tanh2','concat3/in2');
    lgraph = connectLayers(lgraph,'concat3','dec3_conv1');
    
    lgraph = connectLayers(lgraph,'dec3_tanh2','up2');
    lgraph = connectLayers(lgraph,'up2_conv','concat2/in1');
    lgraph = connectLayers(lgraph,'enc2_tanh2','concat2/in2');
    lgraph = connectLayers(lgraph,'concat2','dec2_conv1');
    
    lgraph = connectLayers(lgraph,'dec2_tanh2','up1');
    lgraph = connectLayers(lgraph,'up1_conv','concat1/in1');
    lgraph = connectLayers(lgraph,'enc1_tanh2','concat1/in2');
    lgraph = connectLayers(lgraph,'concat1','dec1_conv1');

    PINN = dlnetwork(lgraph);
    if canUseGPU, PINN = dlupdate(@gpuArray,PINN); end % train on GPU
end