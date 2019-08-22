% misc_test.m
%
% miscellaneous tests tried when implementing parallel GPU utilization
%
% TCS 8/22/2019
%
% NOTE: do not use! all this functionality has been baked into
% gridfitgpu_par.m



load gridFit_vars.mat;

% build model (concatenate trends; or a constant)
model = single(nan(360,4,size(prediction,2)));
model(:,1,:) = prediction;
model(:,2:4,:) = repmat(trends,1,1,size(prediction,2)); % incl 3 trends, model is ~3 GB

% data is in the loaded file

if ~exist('models_to_test','var')
    models_to_test = 500;
end


% try multiple GPUs (across voxels)
myp = parpool(gpuDeviceCount);
tmpmodel = model(:,:,(1:models_to_test));
tmpdata = cell(5,1);
for ii = 1:5
    tmpdata{ii} = data(:,(1:200)+(ii-1)*200);
end
tmpdata = repmat(tmpdata,500,1);
fprintf('all GPUs (parfor), 0.5 million vox\n');
tic;
parfor ii = 1:length(tmpdata)
    gridfitgpu(tmpdata{ii},tmpmodel);
end
fprintf('final time: ');
toc;
delete(myp);

% test a subset of voxels
fprintf('Single GPU, 200 vox\n');
[a,b,c] = gridfitgpu(data(:,1:200),model(:,:,(1:models_to_test)));



% now do it like real - ~16k voxels per 'slice', 471312 models
bigdata = repmat(data,1,16); 
bigmodel = repmat(single(model),1,1,471);

% first, like usual: (one Titan V on tcs-compute-1: 861 s)
fprintf('Single GPU, 16k vox, 417k models\n');
[a,b,c] = gridfitgpu(bigdata,bigmodel);

% now, make it run via parfor
% (3 Titan V on tcs-compute-1: 453 s)
bigdatacell = cell(gpuDeviceCount,1);
nvoxpercell = ceil(size(bigdata,2)/gpuDeviceCount);
for ii = 1:gpuDeviceCount
    thisidx = (1:nvoxpercell)+(ii-1)*nvoxpercell;
    thisidx = thisidx(thisidx<=size(bigdata,2));
    bigdatacell{ii} = bigdata(:,thisidx);
    clear thisidx;
end

fprintf('all GPUs, split over GPUs\n');
maxNumCompThreads(10);
myp = parpool(gpuDeviceCount);
tic;
parfor ii = 1:length(bigdatacell)
    gridfitgpu(bigdatacell{ii},bigmodel);
end
toc;
delete(myp);
maxNumCompThreads('automatic');


% now, the automatic version of that:
fprintf('Multi GPU, gridfitgpu_par.m\n');
tic;
[a,b,c] = gridfitgpu_par(bigdata,bigmodel);
toc;
delete(gcp('nocreate'));