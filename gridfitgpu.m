% gridfitgpu.m
%
% operates over large datasets, too big to fit in GPU RAM and have overhead
% for computations, to search for which model among a large set of models
% (MODEL) best predicts measured data (DATA). Can fit many dimensions of
% data simultaneously (e.g., voxels). Fits linear model, returns betas for
% best-fitting model, sum-squared-error (minimum), and the index of the
% best model
%
% General problem is that, when we want to compare a very large number of
% model predictions against a very large set of data (entire brain fMRI
% time series), we can't solve the linear system in the GPU RAM, and doing
% so on CPUs can take a fairly long time, even on modern hardware. When the
% problem can fit on a GPU, things can be *very* fast - 10-50x faster. But
% large datasets often cannot fit on the GPU, especially because many
% fitting problems require some intermediate values to be stored (see
% gpuregress.m). Accordingly, this function determines how much memory is
% needed, compares that with how much is available, and chops up the
% problem into lots of shorter pieces that run very very quickly. In
% principle, can parallelize this over multiple GPUs quite easily. 
%
% INPUTS:
% data: n_measurements (tpts) x n_dimensions (vox)
% model: n_measurements x n_predictors x n_models
% trunc_neg_fits: binary; determines whether to allow the first beta term
% to be negative (if true, exclude negative fits; otherwise, allow them)
% - this is a carryover from vistasoft, and so only (for now) operates on
%   first dimension; can easily extend to others in updates
% gpu_overhead: 0-1, %age of GPU memory to leave open for OS tasks (should
% be ~0.1-0.3, higher if the GPU also handles graphics, lower if
% compute-only)
%
% OUTPUTS:
% bf_idx: 1 for each dimension (vox)
% bf_b:   1 for each dimension (vox), predictor
% sse:    1 for each dimension (vox)
%
%
% Tommy Sprague, NYU, Feb 17, 2017. tsprague@nyu.edu or
% tommy.sprague@gmail.com if you have questions/problems
%
% WORK IN PROGRESS, tested to work as of date above, but can always be
% improved.
%



function [bf_idx,bf_b,bf_sse] = gridfitgpu(data, model,trunc_neg_fits,gpu_overhead)

% if not otherwise specified, don't allow for negative fits (to align w/
% vistasoft gridfit procedures)
if nargin < 3 || isempty(trunc_neg_fits)
    trunc_neg_fits = 1;
end


% reset the GPU and get info about its memory
gg = gpuDevice;
%fprintf('GPU index %i\n',gg.Index);
if nargin < 4 || isempty(gpu_overhead)
    gpu_overhead = 0.125; % let's shoot for using 90% of GPU RAM, we can make this higher if necessary
end

total_mem_available = gg.AvailableMemory; % in bytes

n_measurements = size(data,1);
n_dimensions = size(data,2); % vox


% solve for most models we can run at a time across all dimensions
% (eventually: check to make sure this is >=1)

n_predictors = size(model,2);
n_models = size(model,3);    % total # of models

tmpm = 1:n_models;
total_mem_load = max(4 * (n_dimensions.*tmpm.*n_measurements + n_measurements.*n_predictors.*tmpm  + n_measurements.*n_dimensions + n_dimensions.*tmpm.*n_predictors), 4*(2*n_dimensions .* tmpm .* n_measurements + n_dimensions.*tmpm));
clear tmpm;


% 
n_models_per_iter = find((total_mem_load./total_mem_available)<=(1-gpu_overhead),1,'last');
clear total_mem_load;

% one entry for each measurement (voxel; reconstruction; etc)
bf_idx = single(nan(n_dimensions,1));
bf_sse = single(inf(n_dimensions,1));
bf_b   = single(nan(n_dimensions,n_predictors));

niter = ceil(n_models/n_models_per_iter);

fprintf('Beginning loop over model blocks\n%i models per block, %i blocks total\n\n',n_models_per_iter,niter);

% can this stay outside loop or does it have to be inside, after a
% gpuReset?
dg = gpuArray(data);
tic;
for ii = 1:niter
    if mod(ii,100)==0
        fprintf('Starting GPU fold %i\n',ii);
    end
    
    
    % which models are we fitting?
    which_models = ((ii-1)*n_models_per_iter+1):(min(n_models,ii*n_models_per_iter));
    
    
    % set up GPU arrays for this iteration
    %dg = gpuArray(data
    mg = gpuArray(model(:,:,which_models));
    
    
    [myXg,myPredg] = gpuregress(mg,dg);
    myX = gather(myXg); clear myXg;
    
    mysse = gpusse(myPredg,dg);
    
    % get rid of negative fits?
    if trunc_neg_fits == 1
        mysse(myX(1,:,:)<0) = inf('single');
    end
    
    % we also need to look for the best model
    [newmin,minidx] = min(mysse,[],3);
    idx_to_replace = find((newmin<bf_sse.'));
    
    bf_sse(idx_to_replace) = gather(newmin(idx_to_replace));
    bf_idx(idx_to_replace) = which_models(minidx(idx_to_replace)); % which model?

    %  (below: 1.43 s); 500 models: 3.73s, 3.65 s, 3.83 s
    % ----- using a for loop --------- (about 5-10% faster than repmat,
    % below)
    for bb = 1:size(myX,1)
        bf_b(idx_to_replace,bb) = myX(sub2ind(size(myX),bb*ones(length(idx_to_replace),1),idx_to_replace,minidx(idx_to_replace))); % this one's a bit harder.... sub2ind (X: n_predictors x n_dimensions x n_models)
    end



    clear mg mysse myPredg idx_to_replace newmin minidx;
    
end
toc;


return
