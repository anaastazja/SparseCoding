function ConvSparse_vgg_MNIST_simple(varargin)


close all
clc;


%%  SET DIRECTORY
addpath(genpath('./utils/MNIST'));
addpath('./dictionary');
addpath(genpath('./data/MNIST'));
addpath(genpath('./MNIST'));
addpath('./CNN');
run(fullfile(fileparts(mfilename('fullpath')), './MatConvNet/vl_setupnn.m')) ;


%%% original data path
MNIST       = 'D:\Studia\Praca magisterska\MatLab\data\MNIST';


%%% sava data path
modelPath           = './model';
dataPath            = './data';
dictionaryPath      = './dictionary';
resultPath          = './result';
featurePath         = './feature';

%%% processing dataset
dataDir             = MNIST;
datasetName         = 'MNIST';
patchNameSize       = '28';

%%% number of classes
numClass            = 10;     


%%% 
if gpuDeviceCount
    gpud = gpuDevice();
    fprintf('success loading GPU.\n');
end

%%%       
convsparsetrain            = 1;        % 1, learn 1st layer decoders      % 1, initialize 1st layer weight with unsupervised learned features

InitMNISTparams


%% PRE-PROCESSING DATA
load './data/MNIST/mnist.mat'
xtrain = single(training.images);
ytrain = single(training.labels);
ytrain = transpose(ytrain);

if gpuDeviceCount
    xtrain = reshape(xtrain, 784, 60000);
    xtrain = gpuArray(xtrain);
    ytrain = gpuArray(ytrain);
end
acttype    = 'max'; 
imgsize    = [28 28 1];

%Prepare train/test data

cv = cvpartition(size(xtrain,2),'HoldOut',0.1);
idx = cv.test;
xtest = xtrain(:, idx);
ytest = ytrain(:, idx);

numTrains = size(xtrain,2);

%% convolutional sparse learning first layer feature
str = sprintf('covkernel_%s_%s_imgsz%s_mdsz%s-%s_ksz%s-%s_pool%s-%s_vneigb%s', acttype, datasetName, patchNameSize,...
        num2str(numfeatures1), num2str(numfeatures2), num2str(kernelsize1), num2str(kernelsize2),...
        num2str(poolsize1), num2str(poolsize2), num2str(vneighbors1));
 

if convsparsetrain
    ConvSparseLearningMNIST;
else
    load([dictionaryPath filesep str]);
end

fprintf("Done ConvSparseLearning");

test_MNIST(xtest, ytest, str, vneighbors1, acttype, 'imgsize', [28 28 1]);
