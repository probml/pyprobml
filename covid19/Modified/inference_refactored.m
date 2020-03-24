function [para_post, z_post] = inference_refactored(M, pop_static, obs_truth, OEV, ...
    num_ens, num_iter, num_times, gam_rnds, legacy)

num_loc = size(M,1);%number of locations
[param0_ens, paramax,paramin]=initialize_params(num_ens);
num_para = size(param0_ens,1); % beta,mu,theta,Z,alpha,D
num_states = num_loc*5; % for each locn, S, E, IR, IU, O
%num_var = num_states + num_para;
pop0_ens = pop_static*ones(1,num_ens);

theta=zeros(num_para, num_iter+1);
para_post=zeros(num_para,num_ens,num_times,num_iter);
z_post=zeros(num_states,num_ens,num_times,num_iter);

var_shrinkage_factor = 0.9; 
SIG=(paramax-paramin).^2/4; %initial covariance of parameters
inflation_factor=1.1;

for n=1:num_iter
    fprintf('iteration %d\n', n)
    sig=var_shrinkage_factor^(n-1);
    Sigma=diag(sig^2*SIG);
    if (n==1)
        %first guess of state space
        [param0_ens, ~, ~] = initialize_params(num_ens);
        z0_ens = initialize_state(pop_static, num_ens, M);
        theta(:,1)=mean(param0_ens,2);%mean parameter
    else
        z0_ens = initialize_state(pop_static, num_ens, M);
        param0_ens=mvnrnd(theta(:,n)',Sigma,num_ens)';%generate parameters
    end
    param0_ens = checkbound_params(param0_ens); 
    z0_ens = checkbound_states(z0_ens, pop0_ens);
    [z_post_iter, p_post_iter] = ensembleKF1(z0_ens, param0_ens, ...
        M, pop_static, obs_truth, OEV, inflation_factor, gam_rnds, legacy);
    z_post(:,:,:,n) = z_post_iter;
    para_post(:,:,:,n) = p_post_iter;
    temp=squeeze(mean(p_post_iter,2));%average over ensemble members
    theta(:,n+1)=mean(temp,2);%average over time
end

end
