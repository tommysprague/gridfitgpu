% run_gridfitgpu.m
%
%
% example script for loading data & predictions and running a gridfit on
% GPU
%
% gridFit_vars.mat is a small subset of an example dataset that contains
% measured voxel timeseries during a vRF mapping procedure (data), as
% well as run-wise trends (just constant offset), and a subset of predicted
% voxel responses, given a set of vRF parameters (not included). 

%% for running a gridfit using pre-defined predictions, data (e.g., from mrVista)
load gridFit_vars.mat;

% build model (concatenate trends; or a constant)
model = single(nan(360,4,size(prediction,2)));
model(:,1,:) = prediction;
model(:,2:4,:) = repmat(trends,1,1,size(prediction,2)); % incl 3 trends, model is ~3 GB

% data is in the loaded file

if ~exist('models_to_test','var')
    models_to_test = 500;
end

[a1,b1,c1] = gridfitgpu(data,model(:,:,(1:models_to_test)));


% if more than 1 GPU is available, also run the model split across GPUs.
% note that performnce may be slower than a single GPU if a small problem
% is tested - but when problems are large (take > 10-30 s on a singe GPU),
% often there is substantial time savings. but - always check your
% particular problem to see if it makes sense to parallelize or not!
if gpuDeviceCount > 1
    tic;
    [a2,b2,c2] = gridfitgpu_par(data,model(:,:,(1:models_to_test)));
    toc;
    delete(gcp('nocreate'));
end
