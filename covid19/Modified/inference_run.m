load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

num_ens = 5;
num_iter = 3;
%num_times = 4; % can reduce num time steps to < 14 for faster debugging
seed = 42;
obs_truth = obs_truth(:, 1:num_times);

%observed error variance
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(4,obs_truth(l,t)^2/4);
    end
end

Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers

legacy = false; %true;

rng(seed); 
tic
[ppost, zpost] = inference_refactored(M, pop, obs_truth, OEV, num_ens, num_iter, num_times, gam_rnds, legacy);
toc

% num_states = num_loc * 5 = 375*5 = 1875
% zpost: num_states * num_ens * num_times * num_iter
% num_params = 6
% ppost: num_params * num_ens  * num_times * num_iter
save('inference','zpost','ppost');