load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
data.obs_truth=incidence'; % obs(l,t)
data.M = M;
data.pop = pop;
fig_folder = '~/covid19/Figures';
model = [];

model.params = set_params(1);
model.add_delay = true;
model.name = 'params=paper';
rng(42);
num_ens = 100;
model.loss = mc_objective(model, data,  num_ens);
obs_samples = sample_data(model, data, num_ens);
plot_samples(obs_samples, data.obs_truth, model.name, fig_folder)
model_paper = model;


%{
model.params = set_params(2);
model.add_delay = true;
model.name = 'params=MIF'
rng(42);
num_ens = 100;
loss = mc_objective(model, obs_truth,  num_ens)
plot_samples(model, obs_truth, num_ens, fig_folder)
%}

seeds = 1:3;
ntrials = length(seeds);
models = cell(1, ntrials);
for i = 1:ntrials
    seed = seeds(i);
    rng(seed);
    num_ens = 100;
    max_iter = 50;
    model.params = initialize_params();
    model.name = sprintf('params=NM,seed=%d,iter=%d', seed, max_iter);
    model.loss_init = mc_objective(model, data,  num_ens);
    [model, loss] = fit_model_nelder_mead(model, data, num_ens, max_iter);
    model.loss = mc_objective(model, data,  num_ens);
    %plot_samples(model, data, num_ens, fig_folder);
    models{i} = model;
end


model = models{3}; % seed 3
model.name = 'params=NM,seed=3,iter=50'
rng(42);
num_ens = 100;
loss = mc_objective(model, data,  num_ens)
plot_samples(model, data, num_ens, fig_folder)


