
function plot_samples()

load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
fig_folder = '~/covid19/Figures';

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

%set observed error variance
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(4,obs_truth(l,t)^2/4);
    end
end

rng(42);
num_ens = 300;
sample_from_prior = false;
rnd_init = true;

Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers

wuhan = 170;
global all_loc
all_loc = num_loc+1; % special index

params = set_params();
obs_truth_sumloc = sum(obs_truth);
global max_count
max_count = max(obs_truth_sumloc(:));


if sample_from_prior
    add_delay = false;
    [obs_samples, state_samples] = sample_data(params, M, pop, num_ens, add_delay);
else
    z0_ens = initialize_state(pop, num_ens, M, rnd_init);
    param0_ens = params * ones(1,num_ens);
    inflation_factor = 1.1;
    legacy = false;
    [zpost, ppost, obs_samples] = ensembleKF1(z0_ens, param0_ens, ...
        M, pop, obs_truth, OEV, inflation_factor, gam_rnds, legacy, true);
end
    

figure;
plot_true_counts(obs_truth, all_loc);
title('all locations')
hold on
plot_pred_counts(obs_samples, all_loc);
fname = sprintf('%s/samples_all', fig_folder); print(fname, '-dpng');


figure;
plot_true_counts(obs_truth, wuhan);
title('wuhan')
hold on
plot_pred_counts(obs_samples, wuhan);
fname = sprintf('%s/samples_wuhan', fig_folder); print(fname, '-dpng');


end



function plot_pred_counts(obs_samples, loc)
global all_loc max_count
if loc == all_loc
    samples = squeeze(sum(obs_samples,1));
else
    samples = squeeze(obs_samples(loc,:,:));
end
[num_ens num_times] = size(samples);
boxplot(samples);
xlim([0 num_times+1]);
ylim([-10 max_count+10])
end


function plot_true_counts(obs_counts, loc)
global all_loc max_count
if loc == all_loc
    counts = sum(obs_counts,1);
else
    counts = obs_counts(loc, :);
end
num_times = length(counts);
plot(counts, 'kx', 'markersize', 10);
xlim([0 num_times+1]);
ylim([-10 max_count+10])
ylabel('reported cases')
end


