function para_post = inference()
%Inference for the metapopulation SEIR model
%Programmed by Sen Pei (contact:sp3449@cumc.columbia.edu)

rng(42);
load M %load mobility
load pop %load population
Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
rnds=ceil(gamrnd(a,b,1e4,1));%pre-generage gamma random numbers
num_loc=size(M,1);%number of locations
%observation operator: obs=Hx
H=zeros(num_loc,5*num_loc+6);
for i=1:num_loc               
    H(i,(i-1)*5+5)=1;
end
load incidence %load observation
num_times=size(incidence,1);
obs_truth=incidence';
%set OEV
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(4,obs_truth(l,t)^2/4);
    end
end
num_ens=300;%number of ensemble
pop0=pop*ones(1,num_ens);
[x,paramax,paramin]=initialize(pop0,num_ens);%get parameter range
num_var=size(x,1);%number of state variables
%IF setting
Iter=3; %10;%number of iterations
num_para=size(paramax,1);%number of parameters
theta=zeros(num_para,Iter+1);%mean parameters at each iteration
para_post=zeros(num_para,num_ens,num_times,Iter);%posterior parameters
sig=zeros(1,Iter);%variance shrinking parameter
alp=0.9;%variance shrinking rate
SIG=(paramax-paramin).^2/4;%initial covariance of parameters
lambda=1.1;%inflation parameter to aviod divergence within each iteration
%start iteration for Iter round
for n=1:Iter
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
    %correct lower/upper bounds of the parameters
    x=checkbound_ini(x,pop0);
    %Begin looping through observations
    x_prior=zeros(num_var,num_ens,num_times);%prior
    x_post=zeros(num_var,num_ens,num_times);%posterior
    pop=pop0;
    obs_temp=zeros(num_loc,num_ens,num_times);%records of reported cases
    for t=1:num_times
    fprintf('inference_original: iter %d time %d\n', n, t)
        %inflation
        x=mean(x,2)*ones(1,num_ens)+lambda*(x-mean(x,2)*ones(1,num_ens));
        x=checkbound(x,pop);
        %integrate forward
        [x,pop]=SEIR(x,M,pop,t,pop0);
        obs_cnt=H*x;%new infection
        %add reporting delay
        for k=1:num_ens
            for l=1:num_loc
                if obs_cnt(l,k)>0
                    rnd=datasample(rnds,obs_cnt(l,k));
                    for h=1:length(rnd)
                        if (t+rnd(h)<=num_times)
                            obs_temp(l,k,t+rnd(h))=obs_temp(l,k,t+rnd(h))+1;
                        end
                    end
                end
            end
        end
        obs_ens=obs_temp(:,:,t);%observation at t
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
    para=x_post(end-5:end,:,1:num_times);
    temp=squeeze(mean(para,2));%average over ensemble members
    theta(:,n+1)=mean(temp,2);%average over time
end

save('inference_E100_I3_S42.mat','para_post');
end

function x = checkbound_ini(x,pop)
%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D
betalow=0.8;betaup=1.5;%transmission rate
mulow=0.2;muup=1.0;%asymptomatic factor
thetalow=1;thetaup=1.75;%movement factor
Zlow=2;Zup=5;%incubation time
alphalow=0.02;alphaup=1.0;%symptomatic rate
Dlow=2;Dup=5;%infectious time
xmin=[betalow;mulow;thetalow;Zlow;alphalow;Dlow];
xmax=[betaup;muup;thetaup;Zup;alphaup;Dup];
num_loc=size(pop,1);
for i=1:num_loc
    %S
    x((i-1)*5+1,x((i-1)*5+1,:)<0)=0;
    x((i-1)*5+1,x((i-1)*5+1,:)>pop(i,:))=pop(i,x((i-1)*5+1,:)>pop(i,:));
    %E
    x((i-1)*5+2,x((i-1)*5+2,:)<0)=0;
    %Ir
    x((i-1)*5+3,x((i-1)*5+3,:)<0)=0;
    %Iu
    x((i-1)*5+4,x((i-1)*5+4,:)<0)=0;
    %obs
    x((i-1)*5+5,x((i-1)*5+5,:)<0)=0;
end
for i=1:6
    temp=x(end-6+i,:);
    index=(temp<xmin(i))|(temp>xmax(i));
    index_out=find(index>0);
    index_in=find(index==0);
    %redistribute out bound ensemble members
    x(end-6+i,index_out)=datasample(x(end-6+i,index_in),length(index_out));
end
end

function x = checkbound(x,pop)
%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D
betalow=0.8;betaup=1.5;%transmission rate
mulow=0.2;muup=1.0;%asymptomatic factor
thetalow=1;thetaup=1.75;%movement factor
Zlow=2;Zup=5;%incubation time
alphalow=0.02;alphaup=1.0;%symptomatic rate
Dlow=2;Dup=5;%infectious time
xmin=[betalow;mulow;thetalow;Zlow;alphalow;Dlow];
xmax=[betaup;muup;thetaup;Zup;alphaup;Dup];
num_loc=size(pop,1);
for i=1:num_loc
    %S
    x((i-1)*5+1,x((i-1)*5+1,:)<0)=0;
    x((i-1)*5+1,x((i-1)*5+1,:)>pop(i,:))=pop(i,x((i-1)*5+1,:)>pop(i,:));
    %E
    x((i-1)*5+2,x((i-1)*5+2,:)<0)=0;
    %Ir
    x((i-1)*5+3,x((i-1)*5+3,:)<0)=0;
    %Iu
    x((i-1)*5+4,x((i-1)*5+4,:)<0)=0;
    %obs
    x((i-1)*5+5,x((i-1)*5+5,:)<0)=0;
end
for i=1:6
    x(end-6+i,x(end-6+i,:)<xmin(i))=xmin(i)*(1+0.1*rand(sum(x(end-6+i,:)<xmin(i)),1));
    x(end-6+i,x(end-6+i,:)>xmax(i))=xmax(i)*(1-0.1*rand(sum(x(end-6+i,:)>xmax(i)),1));
end
end