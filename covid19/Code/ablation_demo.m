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
model.nsteps = 4;
model.add_noise = true;
rng(42);
num_ens = 10;
%model.loss = mc_objective(model, data,  num_ens);
obs_samples = sample_data(model, data, num_ens, 1);
%plot_samples(obs_samples, data.obs_truth, model.name, fig_folder)

%{
rng(42);
obs_samples2 = sample_data(model, data, num_ens, 2);
%plot_samples(obs_samples, data.obs_truth, model.name, fig_folder)
assert(approxeq(obs_samples, obs_samples2))
%}