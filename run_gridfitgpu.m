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

[a,b,c] = gridfitgpu_test(data,model(:,:,(1:models_to_test)));

