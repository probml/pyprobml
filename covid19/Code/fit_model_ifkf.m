load('../Data/M.mat') %load mobility
load('../Data/pop.mat') %load population
load('../Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

num_ens = 100;
num_iter = 5; %~1 minute per iteration
seed = 42;
add_noise = false;
nsteps = 1;
name = sprintf('ens%d-iter%d-seed%d-noise%d-steps%d', ...
    num_ens, num_iter, seed, add_noise, nsteps);
rng(seed); 
tic
[theta, ppost, zpost] = IFKF(M, pop, obs_truth,  num_ens, num_iter, ...
    add_noise, nsteps);
toc

% zpost: num_states * num_ens * num_times * num_iter
% num_states = num_loc * 5 = 375*5 = 1875
% ppost: num_params * num_ens  * num_times * num_iter
% num_params = 6

data_dir = '~/covid19/Results';
fname = sprintf('%s/params-%s.mat', data_dir, name);
save(fname, 'theta', 'zpost','ppost');
