% gridfitgpu_par.m
%
% parallelization wrapper over gridfitgpu.m - inputs, etc identical, so see
% that documentation (copied below) for usage info. 
%
% If you have more than one GPU installed in a compute system, you can use
% them simultaneously. In practice, speedup is often a bit less than nx,
% but certainly > 1. Speedup will be larger for bigger problems
% (parallelizes over # of entities fit, like voxels). There's overhead to
% setting up the parallelization, so only use if necessary (and be sure to
% compare performance to single-GPU version).
%
% NOTE: does not delete parallel pool! so this will need to be killed after
% the script...  (important to prevent increased overhead for ROI-wise
% model fitting, etc).
%
% from gridfitgpu.m:
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
%
% OUTPUTS:
% bf_idx: 1 for each dimension (vox)
% bf_b:   1 for each dimension (vox), predictor
% sse:    1 for each dimension (vox)
%
%
% Tommy Sprague, UCSB, Aug 22, 2019. tsprague@ucsb.edu or
% tommy.sprague@gmail.com if you have questions/problems
%



function [bf_idx,bf_b,bf_sse] = gridfitgpu_par(data, model,trunc_neg_fits)

% if not otherwise specified, don't allow for negative fits (to align w/
% vistasoft gridfit procedures)
if nargin < 3
    trunc_neg_fits = 1;
end

ngpus = gpuDeviceCount;

% split up voxels evenly over GPUs (TODO: there should be a threshold for
% doing this, to minimize overhead cost....)
data_gpu = cell(ngpus,1);
ndimspercell = ceil(size(data,2)/ngpus);
paridx = zeros(size(data,2),ngpus); % indices of data processed on each parpool worker
for ii = 1:ngpus
    thisidx = (1:ndimspercell)+(ii-1)*ndimspercell;
    thisidx = thisidx(thisidx<=size(data,2));
    paridx(thisidx,ii) = 1;
    data_gpu{ii} = data(:,thisidx);
end

% TODO: if already created w/ ngpus workers, use that; otherwise, make a
% new one
if isempty(gcp('nocreate'))
    maxNumCompThreads(floor(feature('numcores')/ngpus)-1); % leave some CPU overhead
    myp = parpool(ngpus);
    
end

% set up cell arrays to retreive bf_idx, bf_b, bf_sse
bf_idx_cell = cell(ngpus,1);
bf_b_cell   = cell(ngpus,1);
bf_sse_cell = cell(ngpus,1);

parfor ii = 1:ngpus
    [bf_idx_cell{ii},bf_b_cell{ii},bf_sse_cell{ii}] = gridfitgpu(data_gpu{ii},model,trunc_neg_fits);
end

%delete(myp);
% reset to normal # of computational threads per worker
maxNumCompThreads('automatic');

% put the bf_idx_cell, etc, datapoints into bf_idx, bf_b using paridx
bf_idx = nan(size(data,2),1);
bf_b   = nan(size(data,2),size(model,2));
bf_sse = nan(size(data,2),1);

for ii = 1:size(paridx,2)
    bf_idx(paridx(:,ii)==1)   = bf_idx_cell{ii};
    bf_b  (paridx(:,ii)==1,:) = bf_b_cell{ii};
    bf_sse(paridx(:,ii)==1)   = bf_sse_cell{ii};
end



return