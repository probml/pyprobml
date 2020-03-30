function fit_demo()

load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
obs_truth=incidence'; % obs(l,t)
input_data_test.M = M;
input_data.pop = pop;
output_obs_test = obs_truth;
ntrain = 10;
output_obs
[num_locs, num_times] = size(data.obs_truth);
train_data = data;
ntrain = 10;
train_data.obs_truth = data.obs_truth(:, 1:ntrain);
fig_folder = '~/covid19/Figures';

%{
model = [];
model.params = set_params(1);
model.add_delay = true;
model.nsteps = 1;
model.add_noise = false;
model.name = sprintf('paper-noise%d-nsteps%d', model.add_noise, model.nsteps);
rng(42);
num_ens = 100;
%model.loss = mc_objective(model, data,  num_ens);
samples = sample_data(model, data, num_ens);
model.loss = plot_samples(samples, data.obs_truth, model.name, fig_folder, 2)
model_paper = model;
%}


seeds = 1:4;
ntrials = length(seeds);
models = cell(1, ntrials);
loss_list = zeros(1, ntrials);
for i = 1:ntrials
    model = [];
    model.seed = seeds(i);
    rng(model.seed);
    num_ens = 100;
    model.max_iter = 20;
    model.method = 'nelder-mead';
    model = [];
    model.params = initialize_params(1, true);
    model.add_noise = false;
    model.add_delay = true;
    model.num_integration_steps = 1;
    model.ntrain = ntrain;
    model.name = sprintf('NM-ntrain%d-seed%d-iter=%d-noise%d-nsteps%d', ...
        model.ntrain, model.seed, model.max_iter, model.add_noise, ...
        model.num_integration_steps);
    
    function loss = objective(params)
      model.params = params;
      loss = mc_objective(model, data_train, num_ens, model.ntrain);
    end
    %[model, loss] = fit_model_nelder_mead(model, data, num_ens, max_iter);
    options = optimset('Display','iter','MaxIter',model.max_iter);
    [params, loss] = fminsearch(@objective, model.params, options);
    model.params = params;
    model.loss_test = mc_objective(model, data_test,  num_ens);
    model.loss_train = loss;
    
    models{i} = model;
    loss_list(i) = loss;
end


[loss, ndx] = min(loss_list);
model = models{ndx}; % seed 3
rng(42);
num_ens = 100;
samples = sample_data(model, data, num_ens);
plot_samples(samples, data.obs_truth, 2)
%loss_test = mc_objective(model, data_test,  num_ens);
suptitle(sprintf('%s mae_train=%5.3f mae_test=%5.3f', ...
    model.name, model.loss_train, model.loss_test);
fname = sprintf('%s/predictions-%s-%s', fig_folder, model.name);
print(fname, '-dpng');
    
end



