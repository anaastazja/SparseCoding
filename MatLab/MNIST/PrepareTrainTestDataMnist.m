clabel 		= unique(ytrain);
numtrdata 	= 0;
numtsdata 	= 0;
nval      	= 1;

trdata      = [];
vldata      = [];
tsdata      = [];
trlabel     = [];
vllabel     = [];
tslabel     = [];

trindex     = [];
vlindex     = [];
tsindex     = [];
rng('shuffle');

for j = 1:numClass

    idx_label = find(ytrain == clabel(j)); %returns an number of index/ array of indexes when ytrain is equal to a unique value of ytrain
    num 		= length(idx_label);
    idx_rand 	= randperm(num); %returns array is random positions of numbers 1:num
    tr_idx 		= idx_label(idx_rand(1:num));
    ts_idx  	= idx_label(idx_rand(trNum+1:min(end,trNum+50)));
    
    % training data
    trdata 		= horzcat(trdata, xtrain(:,tr_idx(1:end-nval)));
    trlabel 	= horzcat(trlabel, ytrain(:,tr_idx(1:end-nval)));
        
    % validataion data
    A = tr_idx(end-nval+1:end);
    vldata      = horzcat(vldata, xtrain(:,tr_idx(end-nval+1:end)));
    vllabel 	= horzcat(vllabel, ytrain(:,tr_idx(end-nval+1:end)));
    
    % testing data
    tsdata 		= horzcat(tsdata, xtrain(:,ts_idx));
    tslabel 	= horzcat(tslabel, ytrain(:,ts_idx));

    numtrdata 	= numtrdata + length(tr_idx);
    numtsdata 	= numtsdata + length(ts_idx);
    
    trindex = [trindex tr_idx(1:end-nval)];
    vlindex = [vlindex tr_idx(end-nval+1:end)];
    tsindex = [tsindex ts_idx];
end

fprintf('Training number: %d\n', numtrdata);
fprintf('Testing number:  %d\n', numtsdata);
