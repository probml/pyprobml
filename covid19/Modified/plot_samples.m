
function plot_samples()

load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
fig_folder = '~/covid19/Figures';

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)
wuhan = 170;

params = set_params();
obs_truth_sumloc = sum(obs_truth);
max_count = max(obs_truth_sumloc(:));

figure;
plot_counts_loc(obs_truth_sumloc, max_count);
title('total over all locations')

figure;
plot_counts_loc(obs_truth(wuhan,:), max_count);
title('wuhan')

num_ens = 100;
%[obs_samples, state_samples] = sample_data(params, M, pop, num_ens);

end

function plot_counts_loc(counts, max_count)
num_times = length(counts);
plot(counts, 'rx', 'markersize', 8);
xlim([0 num_times+1]);
if nargin >= 2
    ylim([-10 max_count+10]);
end
ylabel('reported cases')
end
