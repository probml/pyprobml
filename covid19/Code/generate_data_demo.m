
load('../Data/Mobility.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)
obs_truth=incidence'; % obs(l,t)
fig_folder = '~/tmp/Figures';


input_data.M = M;
input_data.pop = pop;

model.params = set_params;
model.add_noise = false;
model.add_delay = false;
model.num_ens = 1; % single sample
model.num_integration_steps = 1;
model.rounding = true;


pop0 = pop;
z0 = initialize_state_deterministic(pop0, M, model.rounding);
[beta, mu, theta, Z, alpha, D] = unpack_params(model.params);

T = 5;

ztrace = cell(1,T);
delta_trace = cell(1,T);
pop_trace = cell(1,T);
rounding = true;
for t=1:T
    if t==1
        [ztrace{t}, delta_trace{t}] = deterministic_dynamics(...
            z0, model.params, pop0, M(:,:,1));
        pop_trace{t} = update_pop(pop0, M(:,:,t), theta, pop0);
    else
        [ztrace{t}, delta_trace{t}] = deterministic_dynamics(...
            ztrace{t-1}, model.params, pop_trace{t-1}, M(:,:,t));
         pop_trace{t} = update_pop(pop_trace{t-1}, M(:,:,t), theta, pop0);
    end

end

thresh = 1;
for t=1:1
    plot_nonzero_states(ztrace{t}, t, thresh); suptitle('z')
    plot_nonzero_states(delta_trace{t}, t, thresh); suptitle('delta')
end

[obs_trace, ztrace2, pop_trace2, delta_trace2] = sample_data(model, input_data, T, z0);

a = []; for t=1:T,  a(t) = approxeq(delta_trace2(:,:,t), delta_trace{t}); end; disp(a)
a = []; for t=1:T,  a(t) = approxeq(ztrace2(:,:,t), ztrace{t}); end; disp(a)
a = []; for t=1:T,  a(t) = approxeq(pop_trace2(:,:,t), pop_trace{t}); end; disp(a)

obs = squeeze(obs_trace(:,1,:));

plot_time_series(obs);
