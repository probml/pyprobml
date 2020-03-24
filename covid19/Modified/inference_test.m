
load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

num_ens = 4;
num_iter = 2;
num_times = 2; % can reduce num time steps to < 14 for faster debugging
seed = 42;
obs_truth = obs_truth(:, 1:num_times);

%set observed error variance
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(4,obs_truth(l,t)^2/4);
    end
end


Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generage gamma random numbers

legacy = true;

rng(seed); 
disp('inference2')
[theta2, para_post2, xpost2] = inference2(M, pop, obs_truth, OEV, num_ens, num_iter, num_times, gam_rnds, legacy);

rng(seed); 
disp('inference_orig')
[theta0, para_post0, xpost0] = inference_modified(M, pop, obs_truth, OEV, num_ens, num_iter, num_times, gam_rnds, legacy);

rng(seed); 
disp('inference1')
[theta1, para_post1, xpost1] = inference1(M, pop, obs_truth, OEV, num_ens, num_iter, num_times, gam_rnds, legacy);

assert(approxeq(xpost1, xpost2))
assert(approxeq(theta1, theta2))
assert(approxeq(para_post1, para_post2))

assert(approxeq(xpost0, xpost1))
assert(approxeq(theta0, theta1))
assert(approxeq(para_post0, para_post1))
