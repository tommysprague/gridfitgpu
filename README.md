# gridfitgpu
routines for performing rapidly-parallelized linear regression on nvidia GPUs

Rapidly determine the best model predictions to account for each of many measurements simultaneously. Requires a compatible NVidia GPU (see MATLAB documentation), which requires Windows or Linux. 

Approximately 6-20x speedup, depending on compute capabilities. Identical results to using similar functions on CPU. 

Required for GPU-accelerated functions in vistasoft (https://github.com/tommysprague/vistasoft_ts/tree/sprague_gridfit_updates)
