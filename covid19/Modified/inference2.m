function [theta, para_post, x_post] = inference2(M, pop, obs_truth, OEV, ...
    num_ens, num_iter, num_times, gam_rnds, legacy)

num_loc=size(M,1);%number of locations
%observation operator: obs=Hx
H=zeros(num_loc,5*num_loc+6);
for i=1:num_loc
    H(i,(i-1)*5+5)=1;
end

pop0=pop*ones(1,num_ens);
[x,paramax,paramin]=initialize(pop0,num_ens);%get parameter range
num_var=size(x,1);%number of state variables
num_para=size(paramax,1);%number of parameters

theta=zeros(num_para, num_iter+1);%mean parameters at each iteration
para_post=zeros(num_para,num_ens,num_times,num_iter);%posterior parameters
x_post=zeros(num_var,num_ens,num_times,num_iter);

sig=zeros(1, num_iter);%variance shrinking parameter
alp=0.9;%variance shrinking rate
SIG=(paramax-paramin).^2/4;%initial covariance of parameters
lambda=1.1;%inflation parameter to aviod divergence within each iteration


for n=1:num_iter
    fprintf('iteration %d\n', n)
    sig(n)=alp^(n-1);
    %generate new ensemble members using multivariate normal distribution
    Sigma=diag(sig(n)^2*SIG);
    if (n==1)
        %first guess of state space
        [x,~,~]=initialize(pop0,num_ens);
        para=x(end-5:end,:);
        theta(:,1)=mean(para,2);%mean parameter
    else
        [x,~,~]=initialize(pop0,num_ens);
        para=mvnrnd(theta(:,n)',Sigma,num_ens)';%generate parameters
        x(end-5:end,:)=para;
    end
    x=checkbound(x,pop0);
    
    x_post_iter = process_trajectory6(x, M, pop, obs_truth, OEV, lambda, gam_rnds, legacy);
    para_post_iter = x_post_iter(end-5:end, :, :);
    x_post(:,:,:,n) = x_post_iter;
    para_post(:,:,:,n) = para_post_iter;
    para=para_post(:,:,:,n);
    temp=squeeze(mean(para,2));%average over ensemble members
    theta(:,n+1)=mean(temp,2);%average over time
end

end
