load('../../Data/Mobility.mat')
load('../../Data/pop.mat')
load('../../Data/incidence.mat')
num_iter = 2;
num_ens = 10;
seeds = 1:2;
nseeds = length(seeds);
param_dist = zeros(6, nseeds);
tic
for seedi = 1:nseeds
    seed = seeds(seedi);
    fprintf('\n\nrunning seed %d\n', seed)
    % para_post: (num_para,num_ens,num_times,Iter);
    % theta: (num_para,Iter+1)
    % params are stored in this order: beta, mu, theta, Z, alpha, D
    [para_post,theta] = inference2(M, pop, incidence, num_iter, num_ens);
    disp(theta) % final iteration of method
    param_dist(:,seedi) = theta(:,end);
end
toc  % 10 minutes for 5 seeds, 5 iters, 100 samples
fname = sprintf('param_dist_%dseeds_%dsamples_%diter', nseeds, num_ens, num_iter);
save(fname, 'param_dist')

names = {'beta', 'mu', 'theta', 'Z', 'alpha', 'D'};
figure; 
for i=1:6
    subplot(2,3,i)
    dist = param_dist(i,:);
    boxplot(dist);
    q = quantile(dist, [0.025 0.5 0.975]);
    disp(median(dist))
    title(sprintf('%s %5.3f (%4.2f-%4.2f)', names{i}, q(2), q(1), q(3)));
end
suptitle(sprintf('Original MLEs over %d seeds, %d samples, %d iter',  nseeds, num_ens, num_iter));
fname = sprintf('param_boxplot_%dseeds_%dsamples_%diter', nseeds, num_ens, num_iter);
print(gcf, fname, '-dpng')



