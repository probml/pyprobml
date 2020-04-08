function x_locs_ens_times = process_trajectory5(x_locs_ens_0, mobility_locs_times, ...
    pop_locs, obs_truth_locs_times, obs_var_locs_times, inflation)

[num_var, num_ens] = size(x_locs_ens_0);
[num_loc, num_times] = size(obs_truth_locs_times);

%observation operator: obs=Hx
Hx=zeros(num_loc,5*num_loc+6);
for i=1:num_loc
    Hx(i,(i-1)*5+5)=1;
end
Hz=zeros(num_loc,5*num_loc);
for i=1:num_loc
    Hz(i,(i-1)*5+5)=1;
end
G = sum(mobility_locs_times,3); % connectivity structure of graph


pop_locs_ens_0 = pop_locs * ones(1,num_ens);
pop_locs_ens_t = pop_locs_ens_0;
obs_pred_locs_ens_times = zeros(num_loc,num_ens,num_times);
x_locs_ens_times = zeros(num_var,num_ens,num_times);
x_locs_ens_t = x_locs_ens_0;
for t=1:num_times
     fprintf('timestep %d\n', t)  
    %inflation
    %x_locs_ens_t=mean(x_locs_ens_t,2)*ones(1,num_ens)+inflation*(x_locs_ens_t-mean(x_locs_ens_t,2)*ones(1,num_ens));
    %x_locs_ens_t=checkbound(x_locs_ens_t, pop_locs_ens_t);
    [z_locs_ens_t, p_ens_t] = unpack_x(x_locs_ens_t);
    z_locs_ens_t = inflate(z_locs_ens_t, inflation);
    z_locs_ens_t = checkbound_states(z_locs_ens_t, pop_locs_ens_t);
    p_ens_t = inflate(p_ens_t, inflation);
    p_ens_t = checkbound_params(p_ens_t);
    x_locs_ens_t = pack_x(z_locs_ens_t, p_ens_t);
    
    %integrate forward
    %[x_locs_ens_t,pop_locs_ens_t]=SEIR_refactored(x_locs_ens_t,mobility_locs_times,pop_locs_ens_t,t,pop_locs_ens_0);
    [z_locs_ens_t, p_ens_t] = unpack_x(x_locs_ens_t);
    Mt = mobility_locs_times(:,:,t);
    [z_locs_ens_t] = integrate_ODE_onestep(z_locs_ens_t, p_ens_t, pop_locs_ens_t, Mt, false);
    x_locs_ens_t = pack_x(z_locs_ens_t, p_ens_t);
        
    [beta, mu, theta, Z, alpha, D] = unpack_params(p_ens_t); % each param is 1xnum_ens
    pop_new = pop_locs_ens_t + sum(Mt,2)*theta - sum(Mt,1)'*theta;  % eqn 5
    minfrac=0.6;
    ndx = find(pop_new < minfrac*pop_locs_ens_0);
    pop_new(ndx)=pop_locs_ens_0(ndx)*minfrac;
    pop_locs_ens_t = pop_new;
    
    pred_cnt = Hx * x_locs_ens_t; % predicted number of new infections at current time
    pred_cnt2 = Hz * z_locs_ens_t;
    assert(isequal(pred_cnt, pred_cnt2))
    
    obs_pred_locs_ens_times = add_delayed_obs(obs_pred_locs_ens_times, t, pred_cnt);
    obs_pred_locs_ens_t=obs_pred_locs_ens_times(:,:,t); % (l,e)
    %loop through local observations
    for l=1:num_loc
        neighbors=union(find(G(:,l)>0),find(G(l,:)>0));
        nbrs=[neighbors;l];%add location l
        obs_truth_lt = obs_truth_locs_times(l,t);
        obs_var_lt = obs_var_locs_times(l,t);
        obs_pred_ens_lt = obs_pred_locs_ens_t(l,:);
        dx = compute_state_update_given_lt(x_locs_ens_t, obs_pred_ens_lt, obs_truth_lt, obs_var_lt, nbrs);
        
        [z_locs_ens_t, p_ens_t] = unpack_x(x_locs_ens_t);
        [dz, dp] = compute_update_given_lt(z_locs_ens_t, p_ens_t, obs_pred_ens_lt, obs_truth_lt, obs_var_lt, nbrs);
        dx2 = [dz; dp];
        assert(approxeq(dx, dx2))
        
        x_locs_ens_t = x_locs_ens_t+dx;
        x_locs_ens_t = checkbound(x_locs_ens_t,pop_locs_ens_t);
    end
    x_locs_ens_times(:,:,t)=x_locs_ens_t;
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
num_param = 6;
rr=zeros(1,num_param);
for i=1:num_param
    A=cov(p_prior_samples_t(i,:), obs_prior_samples_lt);
    rr(i)=A(2,1);
end
end

function dx = compute_state_update_given_lt(x_locs_ens_t, obs_prior_ens_lt, obs_truth_lt, obs_var_lt, nbrs)
prior_var = var(obs_prior_ens_lt);
if prior_var==0%if degenerate
    prior_var=1e-3;
end
obs_post_ens_lt = compute_obs_post_lt(obs_prior_ens_lt, obs_truth_lt, obs_var_lt);
dy = obs_post_ens_lt - obs_prior_ens_lt; % error in prediction

[z, p] = unpack_x(x_locs_ens_t);
rr_z = compute_correlation_states_to_obs(z, obs_prior_ens_lt,  nbrs);
rr_p = compute_correlation_params_to_obs(p, obs_prior_ens_lt);
rr = [rr_z rr_p];

rr = rr/prior_var;
dx=rr'*dy;
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