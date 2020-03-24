function x_locs_ens_times = process_trajectory6(x_locs_ens_0, mobility_locs_times, ...
    pop_locs, obs_truth_locs_times, obs_var_locs_times, inflation, gam_rnds, legacy)

[z0, p0] = unpack_x(x_locs_ens_0);
[z_locs_ens_times, p_ens_times] = helper(z0, p0, mobility_locs_times, ...
    pop_locs, obs_truth_locs_times, obs_var_locs_times, inflation, gam_rnds, legacy);
x_locs_ens_times = pack_x(z_locs_ens_times, p_ens_times);
end


function [z_locs_ens_times, p_ens_times] = helper(z_locs_ens_0, p_ens_0, ...
    mobility_locs_times, pop_locs, obs_truth_locs_times, obs_var_locs_times, ...
    inflation, gam_rnds, legacy)

[num_var, num_ens] = size(z_locs_ens_0);
[num_loc, num_times] = size(obs_truth_locs_times);
num_params = size(p_ens_0,1);

%observation operator: obs=Hz
Hz=zeros(num_loc,5*num_loc);
for i=1:num_loc
    Hz(i,(i-1)*5+5)=1;
end
G = sum(mobility_locs_times,3); % connectivity structure of graph

pop_locs_ens_0 = pop_locs * ones(1,num_ens);
pop_locs_ens_t = pop_locs_ens_0;
obs_pred_locs_ens_times = zeros(num_loc,num_ens,num_times);
z_locs_ens_times = zeros(num_var, num_ens,num_times);
p_ens_times = zeros(num_params, num_ens,num_times);
z_locs_ens_t = z_locs_ens_0;
p_ens_t = p_ens_0;
for t=1:num_times
     fprintf('timestep %d\n', t)  
    %inflation
    z_locs_ens_t = inflate(z_locs_ens_t, inflation);
    z_locs_ens_t = checkbound_states(z_locs_ens_t, pop_locs_ens_t);
    p_ens_t = inflate(p_ens_t, inflation);
    p_ens_t = checkbound_params(p_ens_t);
    
    %integrate state forward one step
    Mt = mobility_locs_times(:,:,t);
    [z_locs_ens_t] = integrate_ODE_onestep(z_locs_ens_t, p_ens_t, pop_locs_ens_t, Mt, legacy);
    
    % compute new predicted population
    [beta, mu, theta, Z, alpha, D] = unpack_params(p_ens_t); % each param is 1xnum_ens
    pop_new = pop_locs_ens_t + sum(Mt,2)*theta - sum(Mt,1)'*theta;  % eqn 5
    minfrac=0.6;
    ndx = find(pop_new < minfrac*pop_locs_ens_0);
    pop_new(ndx)=pop_locs_ens_0(ndx)*minfrac;
    pop_locs_ens_t = pop_new;
   
    pred_cnt = Hz * z_locs_ens_t; % predicted counts
    obs_pred_locs_ens_times = add_delayed_obs(obs_pred_locs_ens_times, t, pred_cnt, gam_rnds);
    obs_pred_locs_ens_t=obs_pred_locs_ens_times(:,:,t); % (l,e)
    %absorb observation at each location sequentially
    for l=1:num_loc
        neighbors=union(find(G(:,l)>0),find(G(l,:)>0));
        nbrs=[neighbors;l];%add location l      
        [dz, dp] = compute_update_given_lt(z_locs_ens_t, p_ens_t, ...
            obs_pred_locs_ens_t(l,:), ...
            obs_truth_locs_times(l,t), obs_var_locs_times(l,t), nbrs);
        z_locs_ens_t = z_locs_ens_t + dz;
        z_locs_ens_t = checkbound_states(z_locs_ens_t, pop_locs_ens_t);
        p_ens_t = p_ens_t + dp;
        p_ens_t = checkbound_params(p_ens_t);
    end
    z_locs_ens_times(:,:,t)=z_locs_ens_t;
    p_ens_times(:,:,t)=p_ens_t;
end

end

function y = inflate(x, inflation)
    N = size(x,2);
    m = mean(x,2);
    y = m*ones(1,N) + inflation*(x - m*ones(1,N));
end
    
function obs_post_ens_lt = compute_obs_post_lt(obs_prior_ens_lt, obs_value_lt, obs_var_lt)
% Get predictive distribution of single observed variable from ensemble
prior_var = var(obs_prior_ens_lt);
post_var = prior_var*obs_var_lt/(prior_var+obs_var_lt);
% sigma_post = sigma_prior * sigma_obs / (sigma_prior + sigma_obs)
if prior_var==0%if degenerate
    post_var=1e-3;
    prior_var=1e-3;
end
prior_mean = mean(obs_prior_ens_lt);
% p7 first eqn first 2  terms
post_mean = post_var*(prior_mean/prior_var + obs_value_lt/obs_var_lt);
% p7 first eqn last term
alpha = (obs_var_lt/(obs_var_lt+prior_var)).^0.5;
obs_post_ens_lt = post_mean + alpha*(obs_prior_ens_lt-prior_mean); % 1 x nens
end


function rr = compute_correlation_states_to_obs(z_prior_samples_t, obs_prior_samples_lt,  nbrs)
num_var = size(z_prior_samples_t, 1);
%Loop over each state variable (connected to location l)
rr=zeros(1,num_var);
for i=1:length(nbrs)
    idx=nbrs(i);
    for j=1:5
        A=cov(z_prior_samples_t((idx-1)*5+j,:), obs_prior_samples_lt);
        rr((idx-1)*5+j)=A(2,1);
    end
end
end

function rr = compute_correlation_params_to_obs(p_prior_samples_t, obs_prior_samples_lt)
num_param = size(p_prior_samples_t, 1);
rr=zeros(1,num_param);
for i=1:num_param
    A=cov(p_prior_samples_t(i,:), obs_prior_samples_lt);
    rr(i)=A(2,1);
end
end

function [dz, dp] = compute_update_given_lt(z_locs_ens_t, p_ens_t, obs_prior_ens_lt, obs_truth_lt, obs_var_lt, nbrs)
prior_var = var(obs_prior_ens_lt);
if prior_var==0%if degenerate
    prior_var=1e-3;
end
obs_post_ens_lt = compute_obs_post_lt(obs_prior_ens_lt, obs_truth_lt, obs_var_lt);
dy = obs_post_ens_lt - obs_prior_ens_lt; % nens * 1

rr_z = compute_correlation_states_to_obs(z_locs_ens_t, obs_prior_ens_lt,  nbrs); % 1 * nz
rr_p = compute_correlation_params_to_obs(p_ens_t, obs_prior_ens_lt); % 1 * nparams

rr_z = rr_z/prior_var;
rr_p = rr_p/prior_var;
dz = rr_z' * dy;
dp = rr_p' * dy;
end