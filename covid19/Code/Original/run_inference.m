load('../../Data/Mobility.mat')
load('../../Data/pop.mat')
load('../../Data/incidence.mat')
Iter = 5;
num_ens = 20;
seeds = 1:5;
tic
for seedi = 1:length(seeds)
    seed = seeds(seedi);
    fprintf('\n\nrunning seed %d\n', seed)
    % para_post: (num_para,num_ens,num_times,Iter);
    % theta: (num_para,Iter+1)
    % params are stored in this order: beta, mu, theta, Z, alpha, Dlow
    [para_post,theta] = inference2(M, pop, incidence, Iter, num_ens);
    disp(theta) % final iteration of method
end
toc