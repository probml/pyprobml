
function run_inference()

load('../../Data/Mobility.mat')
load('../../Data/pop.mat')
load('../../Data/incidence.mat')
num_iter = 2;
num_ens = 10;
seeds = 1:2;
nseeds = length(seeds);
param_dist = zeros(6, nseeds);
rnd_init = true;
legacy = true;
obs_truth=incidence';
for seedi = 1:nseeds
    seed = seeds(seedi);

    fprintf('\n\nrunning seed %d\n', seed)
    %[~,theta] = inference2(M, pop, incidence, num_iter, num_ens);
    [theta] = inference_refactored(M, pop, obs_truth, num_ens, num_iter, legacy, rnd_init);
    %disp(theta) % final iteration of method
    param_dist(:,seedi) = theta(:,end);
    fname = sprintf('param-dist-seeds-1to%d-samples%d-iter%d', seedi, num_ens, num_iter);
    save(fname, 'param_dist')

    make_plot(param_dist(:, 1:seedi), fname);
end

end

function make_plot(param_dist, fname)
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
suptitle(sprintf('Sampling distribution for MLEs (refactored code)\n%s', fname))
fname = sprintf('%s_boxplot', fname);
print(gcf, fname, '-dpng');
end

