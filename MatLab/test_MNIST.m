function test_MNIST(xtest, ytest, str, vneighbors, acttype, varargin)

%making custom colormap

n = 256;               %// number of colors
B = linspace(1, 0, n);  %// Blue
R = ones(size(B));  %// Red
G = linspace(1, 0, n);  %// Green
mymap = [R(:), G(:), B(:)];

InitMNISTparams;

test_mode = 1;
disp(newline + "TESTING MODE" + newline);

%%% sava data path
modelPath           = './model';
dataPath            = './data';
dictionaryPath      = './dictionary';
resultPath          = './result';
featurePath         = './feature';

load([dictionaryPath filesep str]);

opts.batchSize = 256 ;
opts.useGpu = false ;
opts.errorType = 'multiclass' ;
opts.expDir = 'data/exp' ;
opts.test = [] ;
opts.imgsize = [];

opts.conserveMemory = true ;
opts.sync = true ;
opts.prefetch = false ;


opts = vl_argparse(opts, varargin) ;

imgsize     = opts.imgsize;

tiedflag                = 1;
poolstride              = 1;

tiedparams.lambda       = 1.5;    % sparsity regularization
tiedparams.alpha        = 0;     % L1 decay
tiedparams.beta         = 0;      % L2 decay
tiedparams.epsilonw     = 1e-3;  %-4
tiedparams.epsilonb     = 1e-4;  %-5
tiedparams.epsilono     = 1e-4;  %-5
tiedparams.momentum     = 0.5;
tiedparams.momentumf    = 0.99;
params                  = tiedparams;
params.winc             = zeros(size(kernels));


if ~exist(opts.expDir), mkdir(opts.expDir) ; end
if isempty(opts.test), opts.test = 1:length(ytest) ; end
if isnan(opts.test), opts.test = [] ; end

%saving path
basic_path          = "D:\Studia\PracaMagisterska\MatLab\CSAEs-master\MNIST\images\"
save_path           = "D:\Studia\PracaMagisterska\MatLab\CSAEs-master\MNIST\images\" + kernelsize1 + "x" + kernelsize2 + "x" + numfeatures1 + "\lmbda_" + tiedparams.lambda';

%Adding headers to csv file
data_headers = {'number', 'dict_size', 'lmbda', 'sparsity_count', 'percent'};
data_names = {'number', 'mean_absolute_error'};
if ~exist(basic_path + kernelsize1 + "x" + kernelsize2 + ".csv")
    writecell(data_headers, basic_path + kernelsize1 + "x" + kernelsize2 + ".csv");
end
writecell(data_names, save_path + "\data_lmbda_" + tiedparams.lambda + ".csv");

num = 0;
count = 0;
nloop = size(ytest, 2);
for loop = 1:nloop
    count_sparsity = 0;

    count = count + 1;
    img = reshape(xtest(:,loop), imgsize);  
    numSamples = 1;

    [kernels, hbias, obias, params, ri, enc, error]  = TiedRecstConvNets(img, acttype, kernels,...
            hbias, obias, params, poolstride, test_mode, tiedflag, vneighbors);

    if ytest(:, loop) == num
        figure(1); subplot(121), imshow(img,[]);  title 'Oryginał'%imagesc(img); title 'Original image'
        subplot(122); imshow(ri, []);  title 'Rekonstrukcja'%imagesc(img+E); title 'Reconstructed image'
        figure(2); montage(enc); colormap(mymap) ; colorbar; title("Mapa współczynników dla cyfry " + num)
        saveas(figure(1), save_path + "\rekonstrukcja" + num + ".png"); 
        saveas(figure(2), save_path + "\mapa" + num + ".png"); 
        
        %counting sparsity

        pixels = size(enc, 1) * size(enc,2) * size(enc,3);

        for i = 1:size(enc,1)
            for j = 1:size(enc,2)
                for k = 1:size(enc,3)
                    if enc(i, j, k) < 1e-6    
                        count_sparsity = count_sparsity + 1;
                    end
                end
            end
        end
    
        percent = count_sparsity/pixels*100;
        
        %saving data to csv
        dat = [num numfeatures1 tiedparams.lambda count_sparsity percent];
        writematrix(dat, basic_path + kernelsize1 + "x" + kernelsize2 + ".csv", 'WriteMode','append');
        num = num + 1;
    end
    
    %calculating mean absolute error
    error = abs(ri - img);
    mae = sum(error, 'all')/(size(error,1)*size(error,2));
    data = [ytest(:, loop) mae];
    writematrix(data, save_path + "\data_lmbda_" + tiedparams.lambda + ".csv", 'WriteMode','append');
end
