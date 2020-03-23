function [x_post_iter, theta] = inference1(M, pop, incidence, num_ens, Iter, num_times, gam_rnds)

disp('inference1')

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
theta=zeros(num_para,Iter+1);%mean parameters at each iteration
para_post=zeros(num_para,num_ens,num_times,Iter);%posterior parameters
sig=zeros(1,Iter);%variance shrinking parameter
alp=0.9;%variance shrinking rate
SIG=(paramax-paramin).^2/4;%initial covariance of parameters
lambda=1.1;%inflation parameter to aviod divergence within each iteration

x_post_iter=zeros(num_var,num_ens,num_times,Iter);

for n=1:Iter
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
    %correct lower/upper bounds of the parameters
    %x=checkbound_ini(x,pop0);
    x=checkbound(x,pop0);
    %Begin looping through observations
    x_prior=zeros(num_var,num_ens,num_times);%prior
    x_post=zeros(num_var,num_ens,num_times);%posterior
    pop=pop0;
    obs_temp=zeros(num_loc,num_ens,num_times);%records of reported cases
    for t=1:num_times
        fprintf('timestep %dn\n', t)
        %inflation
        x=mean(x,2)*ones(1,num_ens)+lambda*(x-mean(x,2)*ones(1,num_ens));
        x=checkbound(x,pop);
        %integrate forward
        [x,pop]=SEIR(x,M,pop,t,pop0);
        obs_cnt=H*x;%new infection

        %{
        %add reporting delay
        for k=1:num_ens
            for l=1:num_loc
                if obs_cnt(l,k)>0
                    rnd=datasample(gam_rnds,obs_cnt(l,k));
                    for h=1:length(rnd)
                        if (t+rnd(h)<=num_times)
                            obs_temp(l,k,t+rnd(h))=obs_temp(l,k,t+rnd(h))+1;
                        end
                    end
                end
            end
        end
        obs_ens=obs_temp(:,:,t);%observation at t
        %}
        
        obs_ens = obs_cnt; % predicted observed counts
        
        x_prior(:,:,t)=x;%set prior
        %loop through local observations
        for l=1:num_loc
            %Get the variance of the ensemble
            obs_var = OEV(l,t);
            prior_var = var(obs_ens(l,:));
            post_var = prior_var*obs_var/(prior_var+obs_var);
            if prior_var==0%if degenerate
                post_var=1e-3;
                prior_var=1e-3;
            end
            prior_mean = mean(obs_ens(l,:));
            post_mean = post_var*(prior_mean/prior_var + obs_truth(l,t)/obs_var);
            %%%% Compute alpha and adjust distribution to conform to posterior moments
            alpha = (obs_var/(obs_var+prior_var)).^0.5;
            dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);
            %Loop over each state variable (connected to location l)
            rr=zeros(1,num_var);
            neighbors=union(find(sum(M(:,l,:),3)>0),find(sum(M(l,:,:),3)>0));
            neighbors=[neighbors;l];%add location l
            for i=1:length(neighbors)
                idx=neighbors(i);
                for j=1:5
                    A=cov(x((idx-1)*5+j,:),obs_ens(l,:));
                    rr((idx-1)*5+j)=A(2,1)/prior_var;
                end
            end
            for i=num_loc*5+1:num_loc*5+6
                A=cov(x(i,:),obs_ens(l,:));
                rr(i)=A(2,1)/prior_var;
            end
            %Get the adjusted variable
            dx=rr'*dy;
            x=x+dx;
            %Corrections to DA produced aphysicalities
            x = checkbound(x,pop);
        end
        x_post(:,:,t)=x;
        para_post(:,:,t,n)=x(end-5:end,:);
    end
    x_post_iter(:,:,:,n) = x_post;
    
    para=x_post(end-5:end,:,1:num_times);
    temp=squeeze(mean(para,2));%average over ensemble members
    theta(:,n+1)=mean(temp,2);%average over time
end

end
