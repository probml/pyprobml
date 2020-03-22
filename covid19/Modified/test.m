load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

seed = 42;
num_ens = 3;
Iter=2;%number of iterations

num_loc=size(M,1);%number of locations
num_times=size(incidence,1);
obs_truth=incidence';
%set OEV
OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
        OEV(l,t)=max(4,obs_truth(l,t)^2/4);
    end
end

pop0=pop*ones(1,num_ens);

% OLD
rng(seed);
[x0,paramax,paramin]=initialize(pop, num_ens); %ok
num_para=size(paramax,1);%number of parameters
% x has size (375*5 + 6, num_ens) = (1875 + 6, 2) = 1881, 2
%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D

t=1;
rng(seed);
[x1,pop1]=SEIR(x0,M,pop,t,pop0);
 
rng(seed);
[x1new,pop1new]=SEIR_onestep(x0,M,pop,t,pop0);
assert(approxeq(x1, x1new))
assert(approxeq(pop1, pop1new))

%%%%%%

% NEW
rng(seed);
[states0] = initialize_state(pop, num_ens, M); % (1875, 2)
[params0, paramax2, paramin2]=initialize_params(num_ens);

[x0states, x0params] = unpack_x(x0); 
%x0states = x0(1:1875,:);
%x0params = x0(1876:1881,:);
assert(approxeq(x0params, params0))
assert(approxeq(x0states, states0))

%%%%%%%
% first iteration

% OLD
theta=zeros(num_para,Iter+1);%mean parameters at each iteration
[x,~,~]=initialize(pop0,num_ens);
para=x(end-5:end,:);
theta(:,1)=mean(para,2);%mean parameter
x=checkbound_ini(x,pop0);

% NEW
pop0 = pop*ones(1,num_ens);
mean_params_per_iter = zeros(num_para, Iter+1);
states = initialize_state(pop0, num_ens, M);
[params0, paramax, paramin]=initialize_params(num_ens);
states0 = checkbound_states(states0, pop0);
params0 = checkbound_params_init(params0);
