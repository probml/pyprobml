function [x_post, theta] = inference2(M, pop, incidence, num_ens, num_iter, num_times, gam_rnds)

disp('inference2')

num_loc=size(M,1);%number of locations
%observation operator: obs=Hx
H=zeros(num_loc,5*num_loc+6);
for i=1:num_loc
    H(i,(i-1)*5+5)=1;
end

%num_times=size(incidence,1);
obs_truth=incidence';
%set OEV
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(4,obs_truth(l,t)^2/4);
    end
end

pop0=pop*ones(1,num_ens);
[x,paramax,paramin]=initialize(M, pop0,num_ens);%get parameter range
num_var=size(x,1);%number of state variables
num_para=size(paramax,1);%number of parameters
theta=zeros(num_para, num_iter+1);%mean parameters at each iteration
para_post=zeros(num_para,num_ens,num_times,num_iter);%posterior parameters
sig=zeros(1, num_iter);%variance shrinking parameter
alp=0.9;%variance shrinking rate
SIG=(paramax-paramin).^2/4;%initial covariance of parameters
lambda=1.1;%inflation parameter to aviod divergence within each iteration

x_post=zeros(num_var,num_ens,num_times,num_iter);

for n=1:num_iter
    fprintf('iteration %d\n', n)
    sig(n)=alp^(n-1);
    %generate new ensemble members using multivariate normal distribution
    Sigma=diag(sig(n)^2*SIG);
    if (n==1)
        %first guess of state space
        [x,~,~]=initialize(M, pop0,num_ens);
        para=x(end-5:end,:);
        theta(:,1)=mean(para,2);%mean parameter
    else
        [x,~,~]=initialize(M, pop0,num_ens);
        para=mvnrnd(theta(:,n)',Sigma,num_ens)';%generate parameters
        x(end-5:end,:)=para;
    end
    %check rnd params are in range
    %x=checkbound_ini(x,pop0);
    x=checkbound(x,pop0);
    
    x_post(:,:,:,n) = process_trajectory(x, M, pop, num_ens, obs_truth, OEV, lambda, gam_rnds, num_times);
    para=x_post(end-5:end,:,:,n);
    temp=squeeze(mean(para,2));%average over ensemble members
    theta(:,n+1)=mean(temp,2);%average over time
end

end
