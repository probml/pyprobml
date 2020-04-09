load('../../Data/Mobility.mat') %load mobility
load('../../Data/pop.mat') %load population
load('../../Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

legacy =  true;
num_ens = 100;
num_iter = 5; %~1 minute per iteration
seeds = 1:5;
nseeds = length(seeds);
rnd_init = false;

param_dist = zeros(6, nseeds);
tic
for seedi = 1:nseeds
    seed = seeds(seedi);
    fprintf('\n\nrunning seed %d\n', seed)
    % para_post: (num_para,num_ens,num_times,Iter);
    % theta: (num_para,Iter+1)
    % params are stored in this order: beta, mu, theta, Z, alpha, D
    [theta] = inference_refactored(M, pop, obs_truth, num_ens, num_iter, legacy, rnd_init);
    disp(theta) % final iteration of method
    param_dist(:,seedi) = theta(:,end);
end
toc 

names = {'beta', 'mu', 'theta', 'Z', 'alpha', 'D'};
figure; 
for i=1:6
    subplot(2,3,i)
    dist = param_dist(i,:);
    boxplot(dist);
    q = quantile(dist, [0.025 0.5 0.975]);
    title(sprintf('%s %5.3f (%4.2f-%4.2f)', names{i}, q(2), q(1), q(3)));
end
suptitle(sprintf('Refactored MLEs: legacy %d, rndinit %d, %d seeds, %d samples, %d iter',  legacy, rnd_init, nseeds, num_ens, num_iter));
fname = sprintf('param_boxplot_refactored_%dlegacy, %drndinit_%dseeds_%dsamples_%diter', legacy, rnd_init, nseeds, num_ens, num_iter);
print(gcf, fname, '-dpng');
