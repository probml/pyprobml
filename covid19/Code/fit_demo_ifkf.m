load('../Data/M.mat') %load mobility
load('../Data/pop.mat') %load population
load('../Data/incidence.mat') %load observation

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)

fig_folder = '~/covid19/Figures';

input_data_test.M = M;
input_data_test.pop = pop;
output_data_test = obs_truth;

ntrain = 10;
input_data_train.M = M(:,:,1:ntrain);
input_data_train.pop = pop;
output_data_train = obs_truth(:, 1:ntrain);

model = [];
model.method = 'ifkf';
model.num_ens = 100;
model.max_iter = 5;
model.seed = 42;
model.add_delay = true;
model.update_given_nbrs = false;
model.add_noise = false;
model.num_integration_steps = 1;
model.ntrain = ntrain;
model.name = sprintf('%s-nbrs%d-ntrain%d-seed%d-iter=%d-noise%d-nsteps%d', ...
        model.method, model.update_given_nbrs, ...
        model.ntrain, model.seed, model.max_iter, model.add_noise, ...
        model.num_integration_steps);
rng(model.seed); 
tic
[model, loss] = fit_model_ifkf(...
        model, input_data_train, output_data_train, num_ens, model.max_iter);
toc
model.loss_train = loss;
model.loss_test = mc_objective(model, input_data_test, output_data_test, num_ens);

rng(42);
num_ens = 100;
samples = sample_data(model, input_data_test);
plot_samples(samples, output_data_test)
%loss_test = mc_objective(model, data_test,  num_ens);
suptitle(sprintf('%s mae-train=%5.3f mae-test=%5.3f', ...
    model.name, model.loss_train, model.loss_test));
fname = sprintf('%s/predictions-%s-%s', fig_folder, model.name);
print(fname, '-dpng');


data_dir = '~/covid19/Results';
fname = sprintf('%s/params-%s.mat', data_dir, model.name);
save(fname, 'model');
