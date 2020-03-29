load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

seed = 42;
num_ens = 3;


pop0=pop*ones(1,num_ens);

rng(seed);
[x0,paramax,paramin] = initialize(pop, num_ens);
[states0, params0] = unpack_x(x0);
x0_checked =checkbound_ini(x0, pop0);
[states0_checked, params0_checked] = unpack_x(x0_checked);

states0_new = checkbound_states(states0, pop0);
params0_new = checkbound_params(params0, true);
assert(approxeq(states0_checked, states0_new))
assert(approxeq(params0_checked, params0_new))

t=1;
[x1,pop1]=SEIR(x1,M,pop,t,pop0);
[states1, params1] = unpack_x(x1);
x1_checked = checkbound(x1,pop1);
[states1_checked, params1_checked] = unpack_x(x1_checked);

states1_new = checkbound_states(states1, pop1);
params1_new = checkbound_params(params1, false);
assert(approxeq(states1_checked, states1_new))
assert(approxeq(params1_checked, params1_new))