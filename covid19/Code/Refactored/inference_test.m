
load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

[~, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

num_ens = 5;
num_iter = 2;
seed = 2;

num_times = 3; % can reduce num time steps to < 14 for faster debugging
obs_truth = obs_truth(:, 1:num_times);

legacy = false; %true;

rng(seed); 
disp('inference')
[theta, para_post1, zpost1] = inference_refactored(M, pop, obs_truth,  num_ens, num_iter, legacy);

rng(seed); 
disp('inference_orig')
[theta, para_post0, zpost0] = inference_modified(M, pop, obs_truth, num_ens, num_iter, legacy);

assert(approxeq(zpost0, zpost1))
assert(approxeq(para_post0, para_post1))
