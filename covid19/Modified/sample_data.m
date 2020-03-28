function [obs_pred_locs_ens_times, z_locs_ens_times] = sample_data(model, data, num_ens)

params = model.params;
add_delay = model.add_delay;
mobility_locs_times = data.M;
pop_locs = data.pop;

rnd_init = true;
legacy = false; %true;

[num_loc, num_locs2, num_times] = size(mobility_locs_times);
z_locs_ens_0 = initialize_state(pop_locs, num_ens, mobility_locs_times, rnd_init);
[beta, mu, theta, Z, alpha, D] = unpack_params(params); 
params_ens_0 = params * ones(1,num_ens);
inflation_factor=1.1;

Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers


%observation operator: obs=Hz
Hz=zeros(num_loc,5*num_loc);
for i=1:num_loc
    Hz(i,(i-1)*5+5)=1;
end

pop_locs_ens_0 = pop_locs * ones(1,num_ens);
pop_locs_ens_t = pop_locs_ens_0;
obs_pred_locs_ens_times = zeros(num_loc,num_ens,num_times);
num_var = num_loc*5;
z_locs_ens_times = zeros(num_var, num_ens,num_times);
z_locs_ens_t = z_locs_ens_0;

for t=1:num_times
     %fprintf('timestep %d\n', t)  
     z_locs_ens_t = inflate(z_locs_ens_t, inflation_factor);
     z_locs_ens_t = checkbound_states(z_locs_ens_t, pop_locs_ens_t);

     % integrate forward
     Mt = mobility_locs_times(:,:,t);
    [z_locs_ens_t] = integrate_ODE_onestep(z_locs_ens_t, params_ens_0, pop_locs_ens_t, Mt, legacy);
    z_locs_ens_times(:,:,t)=z_locs_ens_t;
        
    % compute new predicted population
    pop_new = pop_locs_ens_t + sum(Mt,2)*theta - sum(Mt,1)'*theta;  % eqn 5
    minfrac=0.6;
    ndx = find(pop_new < minfrac*pop_locs_ens_0);
    pop_new(ndx)=pop_locs_ens_0(ndx)*minfrac;
    pop_locs_ens_t = pop_new;
    
    % generate observaions
     pred_cnt = Hz * z_locs_ens_t; % predicted counts
     if add_delay
        obs_pred_locs_ens_times = add_delayed_obs(obs_pred_locs_ens_times, t, pred_cnt, gam_rnds);
     else
        obs_pred_locs_ens_times(:,:,t) = pred_cnt;
     end
end

end

function y = inflate(x, inflation)
    N = size(x,2);
    m = mean(x,2);
    y = m*ones(1,N) + inflation*(x - m*ones(1,N));
end
