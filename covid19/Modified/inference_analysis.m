load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

load('inference') % run inference_run.m first!
size(zpost)

param_names = ['beta', 'mu', 'theta', 'Z', 'alpha', 'D'];
nparams = length(param_names);
nrows = 2; ncols = 3;
figure;

