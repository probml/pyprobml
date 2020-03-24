
load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

 % run inference_run.m first!

num_ens = 100;
num_iter = 5;
seed = 42;
fname = fprintf('results_E%d_I%d_S%d', num_ens, num_iter, seed);
load(fname) % zpost, ppost
size(zpost)
size(ppost) 

param_names = {'\beta', '\mu', '\theta', 'Z', '\alpha', 'D'};
nparams = length(param_names);

% Figure S4
nrows = 2; ncols = 3;
figure;
for i=1:nparams
    subplot(nrows, ncols, i);
    nbins = 10;
    histogram(ppost(i,:,end,end), nbins) % final set of samples
    title(param_names{i})
end

% Figure S6
nrows = 6; ncols = 1;
figure;
plt=1;
for p=1:nparams
    subplot(nrows, ncols, p);
    samples = squeeze(ppost(p,:,end,:));
    boxplot(samples) 
    ylabel(param_names{p})
end
