function x_post = process_trajectory3(x, M, pop, obs_truth, OEV, lambda)


[num_var, num_ens] = size(x);
[num_loc, num_times] = size(obs_truth);

pop0=pop*ones(1,num_ens);

%observation operator: obs=Hx
H=zeros(num_loc,5*num_loc+6);
for i=1:num_loc
    H(i,(i-1)*5+5)=1;
end

G = sum(M,3); % connectivity structure of graph

pop=pop0;
pred_obs_seq=zeros(num_loc,num_ens,num_times);
x_post=zeros(num_var,num_ens,num_times);
for t=1:num_times
     fprintf('timestep %d\n', t)  
    %inflation
    x=mean(x,2)*ones(1,num_ens)+lambda*(x-mean(x,2)*ones(1,num_ens));
    x=checkbound(x,pop);
    %integrate forward
    [x,pop]=SEIR_refactored(x,M,pop,t,pop0);
    obs_cnt=H*x; % predicted number of new infections
    pred_obs_seq = add_delayed_obs(pred_obs_seq, t, obs_cnt);
    pred_obs=pred_obs_seq(:,:,t); % (l,e)
    %loop through local observations
    for l=1:num_loc
        neighbors=union(find(G(:,l)>0),find(G(l,:)>0));
        nbrs=[neighbors;l];%add location l
        obs_truth_lt = obs_truth(l,t);
        obs_var_lt = OEV(l,t);
        obs_prior_samples_lt = pred_obs(l,:);
        dx = compute_state_update_lt(x, obs_prior_samples_lt, obs_truth_lt, obs_var_lt, nbrs);
        x=x+dx;
        x = checkbound(x,pop);
    end
    x_post(:,:,t)=x;
end

end

function obs_post_samples = compute_obs_post_lt(obs_prior_samples, obs_value, obs_var)
% Get predictive distribution of single observed variable from ensemble
prior_var = var(obs_prior_samples);
post_var = prior_var*obs_var/(prior_var+obs_var);
% sigma_post = sigma_prior * sigma_obs / (sigma_prior + sigma_obs)
if prior_var==0%if degenerate
    post_var=1e-3;
    prior_var=1e-3;
end
prior_mean = mean(obs_prior_samples);
% p7 first eqn first 2  terms
post_mean = post_var*(prior_mean/prior_var + obs_value/obs_var);

% p7 first eqn last term
alpha = (obs_var/(obs_var+prior_var)).^0.5;
obs_post_samples = post_mean + alpha*(obs_prior_samples-prior_mean); % 1 x nens
end

function rr = compute_correlation_with_obs(x, pred_obs_samples,  nbrs)
num_var = size(x, 1);
num_param = 6;
num_loc = (num_var - num_param)/5;
%Loop over each state variable (connected to location l)
rr=zeros(1,num_var);
for i=1:length(nbrs)
    idx=nbrs(i);
    for j=1:5
        A=cov(x((idx-1)*5+j,:), pred_obs_samples);
        rr((idx-1)*5+j)=A(2,1);
    end
end
% loop over each parameter
for i=num_loc*5+1:num_loc*5+6
    A=cov(x(i,:), pred_obs_samples);
    rr(i)=A(2,1);
end
end

function dx = compute_state_update_lt(x, obs_prior_samples_lt, obs_truth_lt, obs_var_lt, nbrs)
prior_var = var(obs_prior_samples_lt);
if prior_var==0%if degenerate
    prior_var=1e-3;
end
obs_post_samples_lt = compute_obs_post_lt(obs_prior_samples_lt, obs_truth_lt, obs_var_lt);
dy = obs_post_samples_lt - obs_prior_samples_lt; % error in prediction
rr = compute_correlation_with_obs(x, obs_prior_samples_lt,  nbrs);
rr = rr/prior_var;
dx=rr'*dy;
end


