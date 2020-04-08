load('../Data/Mobility.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
load('../Data/city_names.mat') % cities{l}
obs_truth=incidence'; % obs(l,t)
fig_folder = '~/tmp/Figures';


input_data.M = M;
input_data.pop = pop;

model.params = set_params;
model.add_noise = false;
model.add_delay = true;
model.rnd_delay = false;
model.num_ens = 1; % single sample
model.num_integration_steps = 1;


pop0 = pop;
z0 = initialize_state_deterministic(pop0, M);
[beta, mu, theta, Z, alpha, D] = unpack_params(model.params);

T = 5;
[obs_trace, ztrace,  delta_trace] = sample_data(model, input_data, T, z0);
%plot_nonzero_states(ztrace{t}, t, thresh); suptitle('z')

obs = squeeze(obs_trace(:,1,:)); % single sample
plot_time_series(obs);


%%%%%%%%%%%%%%%
model.num_ens = 3;
z0samples = initialize_state(pop0, model.num_ens, M);
for i=1:model.num_ens
    plot_nonzero_states(z0samples(:,i), 0, 0); 
    suptitle(sprintf('Z0  sample %d', i))
end

[obs_trace_sample, ztrace_sample, delta_trace_sample] = ...
    sample_data(model, input_data, T);
for t=[1:1]
    for s=[1,2,3]
        %plot_nonzero_states(delta_trace_sample(:,s,t), t, 1);
        plot_time_series(squeeze(obs_trace_sample(:,s,:)));
        suptitle(sprintf('z t %d sample %d', t, s))
    end
end

%%%%%%%%%
input_data_test.M = M;
input_data_test.pop = pop;
output_data_test = obs_truth;

ntrain = 10;
input_data_train.M = M(:,:,1:ntrain);
input_data_train.pop = pop;
output_data_train = obs_truth(:, 1:ntrain);

Osum = sum(obs_truth,2); % sum over time
[Osort, loc_ndx_all] = sort(Osum, 'descend');
loc_ndx_train = loc_ndx_all(3:end); % skip top 2 cities
    
model.params = set_params;
model.add_noise = false;
model.add_delay = true;
model.rnd_delay = true;
model.num_ens = 3; % single sample
model.num_integration_steps = 1;

z0 = initialize_state(pop0, 3, M);
[S,E,IR,IU,O] = unpack_states(z0(:,3));
t=0;
for s=[1,2,3]
    thresh = 0;
    plot_nonzero_states(z0(:,s), t, thresh);
    suptitle(sprintf('z t %d sample %d', t, s))
end

[z1, d1] = sample_from_dynamics(z0,  params, pop0, M(:,:,1), false, 1);
[S,E,IR,IU,O] = unpack_states(z1(:,3));
O(:)'

[Otrace, Ztrace, Dtrace, Otrace_instant] = sample_data(model, input_data_train);

t=1;
for s=[1,2,3]
    thresh = 1;
    plot_nonzero_states(Dtrace(:,s,t), t, thresh);
    suptitle(sprintf('z t %d sample %d', t, s))
end


for s=[1,2,3]
    plot_time_series(squeeze(Otrace(:,s,:)));
    suptitle(sprintf('O sample %d',  s))
end
   
model.num_ens = 100;
model.rnd_delay = true;
[loss,loss_ex] = mc_objective(model, input_data_train, output_data_train, loc_ndx_train);


