function [x_new, pop_new] = SEIR_refactored(x, M, pop, ts, pop0, legacy)

% This is a clean rewrite of SEIR.m
% if legacy=true, we emulate the original buggy matlab code
% (bug confirmed by author)

if nargin < 6
    legacy = false;
end

[states, params] = unpack_x(x);
Mt = M(:,:,ts);
[states_new] = integrate_ODE_onestep(states, params, pop, Mt, legacy);

[beta, mu, theta, Z, alpha, D] = unpack_params(params); % each param is 1xnum_ens
pop_new = pop + sum(Mt,2)*theta - sum(Mt,1)'*theta;  % eqn 5
minfrac=0.6;
ndx = find(pop_new < minfrac*pop0);
pop_new(ndx)=pop0(ndx)*minfrac;

x_new = pack_x(states_new, params);

end

