load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

num_ens = 100;
num_iter = 5;
seed = 42;

rng(seed); 
tic
[ppost, zpost] = inference_refactored(M, pop, obs_truth,  num_ens, num_iter);
toc

% zpost: num_states * num_ens * num_times * num_iter
% num_states = num_loc * 5 = 375*5 = 1875
% ppost: num_params * num_ens  * num_times * num_iter
% num_params = 6

fname = fprintf('results_E%d_I%d_S%d', num_ens, num_iter, seed);
save(fname,'zpost','ppost');
