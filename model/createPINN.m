function net = createPINN()
% createPINN: 2-input (x,y), 3-output (u,v,p)

layers = [
    featureInputLayer(2,"Name","input")
    fullyConnectedLayer(20,"Name","fc1")
    tanhLayer("Name","tanh1")
    fullyConnectedLayer(20,"Name","fc2")
    tanhLayer("Name","tanh2")
    fullyConnectedLayer(20,"Name","fc3")
    tanhLayer("Name","tanh3")
    fullyConnectedLayer(3,"Name","output")  % u, v, p
];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);
end
