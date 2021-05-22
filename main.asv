%% clear everything
clc
clear
close all


%% read data set
load('mnist.mat');


% to see image
% X = trainX;
% im3 = reshape(X(3,:), 28, 28)';
% imshow(i3);

%% vars
alpha = 0.1; % learning rate
layers = [512,512]; % number of nodes in each hidden layer
epochs = 20;
px = 28; % each image is 28*28
Nlayers = numel(layers);

trainX = double(trainX');
testX = double(testX');
trainY = double(trainY);
testY = double(testY);
trainN = size(trainX,2);
testN = size(testX,2);

[XTrain,YTrain] = digitTrain4DArrayData;
[XTest,YTest] = digitTest4DArrayData;

trainHists = zeros(32,trainN);
testHists = zeros(32,testN);


%% feature extraction
for i=1:trainN
   image = reshape(trainX(:,i), px, px)';
   size(image);
   tmp = [];
   for j=1:px/4:px
       for k=1:px/4:px
           h = imhist(image(j:j+6,k:k+6));
           tmp = [tmp,h(1),h(end)];
       end
   end
   trainHists(:,i) = tmp';
end

for i=1:testN
   image = reshape(testX(:,i), px, px)';
   size(image);
   tmp = [];
   for j=1:px/4:px
       for k=1:px/4:px
           h = imhist(image(j:j+6,k:k+6));
           tmp = [tmp,h(1),h(end)];
       end
   end
   testHists(:,i) = tmp';
end



%% MLP without softmax

net = feedforwardnet(layers);
net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'radbas';
net.layers{2}.transferFcn = 'softmax';
% net.layers{3}.transferFcn = 'purelin';
net = train(net,trainHists,trainY); % todo: mini batch size
yHatMLP = net(testHists);
% yHatMLP = predict(net,testHists);
view(net)

yy = discretize(testY, 0:9, 'categorical');
yh = discretize(yHatMLP, 0:9, 'categorical');

figure
plotconfusion(categorical(round(testY)),categorical(round(yHatMLP)));


%% CNN
NNLayers = [...
    imageInputLayer([28,28,1])
    convolution2dLayer(3,64)
    reluLayer
    maxPooling2dLayer(4)
    convolution2dLayer(3,32)
    reluLayer
    maxPooling2dLayer(4)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
   
opts = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'ValidationData',{XTest,YTest}, ...
    'ValidationFrequency',50, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,NNLayers,opts);

%% naive bayes
nb = fitcnb(X(idxtt,:),Y(idxtt),'Weights',W(idxtt));
nb = train(nb,x,labels);


%% KNN
knn = fitcknn(X,Y,'NumNeighbors',3);


%% SAE ( pretrained MLP with auto encoder )


%% SVM

