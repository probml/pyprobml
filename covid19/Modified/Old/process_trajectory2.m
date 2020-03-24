function x_post = process_trajectory2(x, M, pop, obs_truth, OEV, lambda)


[num_var, num_ens] = size(x);
[num_loc, num_times] = size(obs_truth);

pop0=pop*ones(1,num_ens);

%observation operator: obs=Hx
H=zeros(num_loc,5*num_loc+6);
for i=1:num_loc
    H(i,(i-1)*5+5)=1;
end

pop=pop0;
obs_temp=zeros(num_loc,num_ens,num_times);%records of reported cases
x_post=zeros(num_var,num_ens,num_times);%posterior
for t=1:num_times
     fprintf('timestep %d\n', t)  
    %inflation
    x=mean(x,2)*ones(1,num_ens)+lambda*(x-mean(x,2)*ones(1,num_ens));
    x=checkbound(x,pop);
    %integrate forward
    [x,pop]=SEIR_refactored(x,M,pop,t,pop0);
    obs_cnt=H*x; % predicted number of new infections
    obs_temp = add_delayed_obs(obs_temp, t, obs_cnt);
    obs_ens=obs_temp(:,:,t);%observation at t
    %loop through local observations
    for l=1:num_loc
        x = compute_state_update(x, obs_ens, obs_truth, OEV, M, l, t, pop);
    end
    x_post(:,:,t)=x;
end

end

function obs_post = compute_predicted_obs(obs_ens, obs_truth, OEV, l, t)
% Get predictive distribution of observed variable from ensemble
%Get the variance of the ensemble for
obs_var = OEV(l,t);
prior_var = var(obs_ens(l,:));
post_var = prior_var*obs_var/(prior_var+obs_var);
% sigma_post = sigma_prior * sigma_obs / (sigma_prior + sigma_obs)
if prior_var==0%if degenerate
    post_var=1e-3;
    prior_var=1e-3;
end
prior_mean = mean(obs_ens(l,:));
% p7 first eqn first 2  terms
post_mean = post_var*(prior_mean/prior_var + obs_truth(l,t)/obs_var);

% p7 first eqn last term
alpha = (obs_var/(obs_var+prior_var)).^0.5;
obs_post = post_mean + alpha*(obs_ens(l,:)-prior_mean); % 1 x nens
end

function rr = compute_correlation_with_obs(x, obs_ens,  M, l)
num_loc = size(M,1);
num_var = size(x, 1);
%Loop over each state variable (connected to location l)
rr=zeros(1,num_var);
neighbors=union(find(sum(M(:,l,:),3)>0),find(sum(M(l,:,:),3)>0));
neighbors=[neighbors;l];%add location l
for i=1:length(neighbors)
    idx=neighbors(i);
    for j=1:5
        A=cov(x((idx-1)*5+j,:),obs_ens(l,:));
        rr((idx-1)*5+j)=A(2,1);
    end
end
for i=num_loc*5+1:num_loc*5+6
    A=cov(x(i,:),obs_ens(l,:));
    rr(i)=A(2,1);
end
end

function x = compute_state_update(x, obs_ens, obs_truth, OEV, M, l, t, pop)
prior_var = var(obs_ens(l,:));
if prior_var==0%if degenerate
    prior_var=1e-3;
end
obs_post = compute_predicted_obs(obs_ens, obs_truth, OEV, l, t);
dy = obs_post - obs_ens(l,:); % error in prediction
rr = compute_correlation_with_obs(x, obs_ens,  M, l);
rr = rr/prior_var;
dx=rr'*dy;
x=x+dx;
x = checkbound(x,pop);
end


