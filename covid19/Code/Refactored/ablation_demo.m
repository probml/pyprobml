function ablation_demo()

load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
data.obs_truth=incidence'; % obs(l,t) 
data.M = M;
data.pop = pop;
global fig_folder results_folder
fig_folder = '~/covid19/Figures';
results_folder  = '~/covid19/Results';


model = [];
model.params = set_params(1);
model.add_delay = true;
model.num_integration_steps = 4;
model.add_noise = true;
rng(42);
num_ens = 100;
tic
obs_samples = sample_data(model, data, num_ens);
toc
model.name = sprintf('paper-noise=1=steps=4');
plot_samples(obs_samples, data.obs_truth)
suptitle(model.name)


rng(42);
model.add_noise = false;
model.num_integration_steps = 4;
tic
obs_samples = sample_data(model, data, num_ens);
toc
model.name = sprintf('paper-noise=0=steps=4');
plot_samples(obs_samples, data.obs_truth)
suptitle(model.name)

rng(42);
model.add_noise = false;
model.num_integration_steps = 1;
tic
obs_samples = sample_data(model, data, num_ens);
toc
model.name = sprintf('paper-noise=0=steps=1');
plot_samples(obs_samples, data.obs_truth)
suptitle(model.name)

%{
num_ens = 100;
num_iter = 5; 
seed = 42;
add_noise = false;
nsteps = 1;
make_plot(data, num_ens, num_iter, seed, add_noise, nsteps);
%}

end

function make_plot(data, num_ens, num_iter, seed, add_noise, nsteps)
global fig_folder results_folder
name = sprintf('ens%d-iter%d-seed%d-noise%d-steps%d', ...
    num_ens, num_iter, seed, add_noise, nsteps);
fname = sprintf('%s/params-%s.mat', results_folder, name);
tmp = load(fname); % 'theta', 'zpost','ppost');
model = [];
model.add_delay = true;
model.params = tmp.theta(:,end);
model.add_noise = add_noise;
model.nsteps = nsteps;
rng(seed)
obs_samples = sample_data(model, data, num_ens);
rng(seed)
plot_samples(obs_samples, data.obs_truth);
suptitle(name)
end

