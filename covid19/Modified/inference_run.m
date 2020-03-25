load('../Data/M.mat') %load mobility
load('../Data/pop.mat') %load population
load('../Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

num_ens = 300;
num_iter = 10; %~1 minute per iteration
seed = 42;
legacy = true;

rng(seed); 
tic
[theta, ppost, zpost] = inference_refactored(M, pop, obs_truth,  num_ens, num_iter, legacy);
toc

% zpost: num_states * num_ens * num_times * num_iter
% num_states = num_loc * 5 = 375*5 = 1875
% ppost: num_params * num_ens  * num_times * num_iter
% num_params = 6

data_dir = '~/covid19/Results';
fname = sprintf('%s/leg-results-E%d-I%d-S%d.mat', data_dir, num_ens, num_iter, seed);
save(fname, 'theta', 'zpost','ppost');
