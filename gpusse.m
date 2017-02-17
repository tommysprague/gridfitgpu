% gpusse.m
%

function sse = gpusse(pred,data)
%
% computes sum squared error between a set of model predictions PRED and
% data DATA. optimized w/ GPU
%
% INPUTS:
% - PRED: best-fit model's prediction of data (from GPUREGRESS)
%          n_measurements x n_dimensions x n_models
% - DATA: actual data  (all measurements for each measurement dimension)
%          n_measurements x n_dimensions
%
% RETURNS:
% - SSE:  sum squared error for each model, dimension
%          1 x n_dimensions x n_models
%
% Tommy Sprague 8/30/2016 (created)
% tsprague@nyu.edu, tommy.sprague@gmail.com

sse = sum(bsxfun(@(a,b) (a-b).^2,pred,data),1);

return