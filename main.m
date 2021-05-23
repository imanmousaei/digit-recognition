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
   imTrain{i} = image;
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
   imTest{i} = image;
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



%% MLP without softmax -> 38%

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


%% CNN -> 99.16%
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

CNNnet = trainNetwork(XTrain,YTrain,NNLayers,opts);

YCNN=classify(CNNnet,XTest);
plotconfusion(YTest,YCNN);


%% naive bayes -> 0%
nb = fitcnb(trainX,trainY);
Ynb = predict(nb,testX);
% cannot work NB with MNIST(cuz variance=0)


%% KNN -> 97.1%
knn = fitcknn(trainX',trainY','NumNeighbors',3);
Yknn = predict(knn,testX');
plotconfusion(categorical(testY'),categorical(Yknn));


%% discriminant classifier -> 87.3%
discriminant = fitcdiscr(trainX',trainY','discrimType', 'pseudoLinear');
Ydiscriminant = predict(discriminant,testX');
plotconfusion(categorical(testY'),categorical(Ydiscriminant));


%% decision tree -> 87.8%
tree = fitctree(trainX',trainY');
Ytree = predict(tree,testX');
plotconfusion(categorical(testY'),categorical(Ytree));


%% SAE ( pretrained MLP with auto encoder weights )
[xTrainImages,tTrain] = digitTrainCellArrayData;
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
view(autoenc1)
features1 = encode(autoenc1,xTrainImages);

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(features1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
view(autoenc2)
features2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(features2,tTrain,'MaxEpochs',400);
view(softnet)

view(autoenc1)
view(autoenc2)
view(softnet)

stackednet = stack(autoenc1,autoenc2,softnet);
view(stackednet)

% Get the number of pixels in each image
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Load the test images
[xTestImages,tTest] = digitTestCellArrayData;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

y = stackednet(xTest);
plotconfusion(tTest,y);

% fine tuning
stackednet = train(stackednet,xTrain,tTrain);
y = stackednet(xTest);
plotconfusion(tTest,y);




%% multiclass SVM
svm = fitcecoc(trainX',trainY');
Ysvm = predict(svm,testX');
plotconfusion(categorical(testY'),categorical(Ysvm));
