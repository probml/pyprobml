
load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

num_ens = 4;
num_iter = 2;
num_times = 3; % can reduce this to < 14 for debugging
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


rng(seed); 
[theta1, para_post1, xpost1] = inference1(M, pop, obs_truth, OEV, ...
    num_ens, num_iter, num_times, gam_rnds);
% xpost is 1881 x num_ens x num_times x num_iter
% where 1881 = 1875 + 6, 1875 = 375*5

rng(seed); 
[theta2, para_post2, xpost2] = inference2(M, pop, obs_truth, OEV, ...
    num_ens, num_iter, num_times, gam_rnds);

assert(approxeq(xpost1, xpost2))
assert(approxeq(theta1, theta2))
assert(approxeq(para_post1, para_post2))
