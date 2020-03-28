
function plot_samples()

load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
fig_folder = '~/covid19/Figures';

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)
wuhan = 170;

obs_truth_wuhan = obs_truth(wuhan,:);
obs_truth_all = sum(obs_truth); % sum over all locations
max_count = max(obs_truth_all(:)); % max over time to set scale


rng(42);
num_ens = 100;

delays = [false, true];
for j=1:length(delays)
    add_delay = delays(j);
    

param_ndx = 1;
params = set_params(param_ndx);
[obs_samples, state_samples] = sample_data(params, M, pop, num_ens, add_delay);
    
obs_samples_all =  squeeze(sum(obs_samples,1));
obs_samples_wuhan = squeeze(obs_samples(wuhan,:,:));

truth_list = {obs_truth_all, obs_truth_wuhan};
samples_list = {obs_samples_all, obs_samples_wuhan};
%name_list = {'all', 'wuhan'};

truth_list = {obs_truth_wuhan};
samples_list = {obs_samples_wuhan};
name_list = {'wuhan'};

for i=1:length(name_list)
    truth = truth_list{i};
    samples = samples_list{i};
    name = name_list{i};
    figure;
    plot(truth, 'kx', 'markersize', 10)
    hold on
    boxplot(samples)
    ylabel('reported cases')
    xlim([0 num_times+1]);
    ylim([-10 max_count+10])
    
    [mse, mae, nll] =  evaluate_preds(truth, samples);
    title(sprintf('delay=%d, loc=%s, mse=%5.3f, mae=%5.3f, nll=%5.3f',...
        add_delay, name, mse, mae, nll))
    fname = sprintf('%s/predictions_%s_%d', fig_folder, name, add_delay);
    print(fname, '-dpng');
    
end

end

end

function [mse, mae, nll] =  evaluate_preds(truth, samples)
ntimes = length(truth);
nll = 0;
mse = 0;
mae = 0;
for t=1:ntimes
    dist = fitdist(samples(:,t), 'kernel');
    nll = nll + -1*log(pdf(dist, truth(t)));
    mae = mae + abs(median(dist) - truth(t));
    mse = mse + (mean(dist) - truth(t))^2;
end
mse = mse/ntimes;
mae = mae/ntimes;
end


