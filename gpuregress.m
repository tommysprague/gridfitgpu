function [X, pred] = gpuregress(model,data)
% solves multiple linear regression for models MODEL given data DATA
%
% Ideal for "gridfit" type problems: uses gpu's pagefun to rapidly iterate
% over models & fit simultaneously to each voxel. Works optimally when
% fitting lots of voxels on a few models; should loop over this function
% with different slices of models. 
%
% INPUTS:
% - MODEL should be n_measurements x n_predictors x n_models (gpuArray,
%   single to save on RAM)
% - DATA  should be n_measurements x n_dimensions (e.g., voxels)
%
% n_dimensions is the number of signal dimensions being fit simultaneously
% (like voxels for a RF model). n_measurements is the number of measured
% signal values for each measured signal dimension (e.g., # of timepoints)
%
% RETURNS:
% - X:    best-fit betas to each predictor (DIM 2 of model); 
%          n_predictors x n_dimensions x n_models
% - pred: best-fit model's prediction of data (for use with GPUSSE)
%          n_measurements x n_dimensions x n_models
%
% should be ~20x speed or greater than similar computation on CPU
% (depending on parallelization). tested on GeForce GTX 670 and Titan Black
%
% TCS, 8/30/2016, tsprague@nyu.edu or tommy.sprague@gmail.com
%
%


% best-fit betas; uses manual pagefun implementation of pinv
% (model'*model)*model'*data
X = pagefun(@mtimes,pagefun(@mtimes,pagefun(@inv,pagefun(@mtimes,pagefun(@transpose,model),model)),pagefun(@transpose,model)),data);

% predicted responses 
pred = pagefun(@mtimes,model,X);


return