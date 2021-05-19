%% clear everything
clc
clear
close all


%% read data set
D=xlsread('dataset.xlsx');
x = D
D = D';
D(:,1)
X = trainX;

% to see image
% im3 = reshape(X(3,:), 28, 28)';
% imshow(i3);

%% vars
alpha = 0.1; % learning rate
layers = [size(D,2)-1, 10, 1]; % number of nodes in each layer
epochs = 20;
Nlayers = numel(layers);
maxL = max(size(D,2)-1,10);
N = size(D,1);
R = 0.8;
Ntrain = round(R*N);


%% MLP

net = feedforwardnet([4 6 1]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,x,labels);
view(net)

%% naive bayes
nb = fitcnb(X(idxtt,:),Y(idxtt),'Weights',W(idxtt));
nb = train(nb,x,labels);


%% KNN
knn = fitcknn(X,Y,'NumNeighbors',3);


%% SAE ( pretrained MLP with auto encoder )



