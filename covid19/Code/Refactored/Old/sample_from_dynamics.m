function [states_new, prob] = sample_from_dynamics(states_t, pop_t, model, data, t, debug)
% Input:
% states(l,s), l=1:nloc, s=1:nsamples
% pop(l,s)
% model.params, model.add_delay, model.nsteps
% data.pop, data.M (ignores data.obs_truth)
% Output:
% states_new(l,s)
% prob(l,s)

params0 = model.params;
add_delay = model.add_delay;
Mt = data.M(:,:,t);

[num_loc, num_ens] = size(states_t);
params = params0 * ones(1,num_ens);

[Sold, Eold, IRold, IUold, ~] = unpack_states(states_t); 
S = Sold; E = Eold; IR = IRold; IU = IUold;

delta_coefs = [1/2, 1/2, 1, 1];
for step=1:model.nsteps
    rates = compute_poisson_rates(S, E, IR, IU, Mt, pop_t, params);
    if model.add_noise
        noise = sample_poisson_noise(rates);
        increment = noise;
    else
        increment = rates;
    end
    [Sdelta{step}, Edelta{step}, IRdelta{step}, IUdelta{step}, Odelta{step}] = ...
        compute_deltas(increment);
    S = Sold + delta_coefs(step)*Sdelta{step};
    E = Eold + delta_coefs(step)*Edelta{step};
    IR = IRold + delta_coefs(step)*IRdelta{step};
    IU = IUold + delta_coefs(step)*IUdelta{step};
    
    if debug && (step <= 2)
    fprintf('step %d\n', step);
        rates(1:3,1)'
    Sdelta{step}(1:3,1)'
    S(1:3,1)'
    end
end

if model.nsteps==1
    Snew = Sold + round(Sdelta{1});
    Enew = Eold + round(Edelta{1});
    IRnew = IRold + round(IRdelta{1});
    IUnew = IUold + round(IUdelta{1});
    Onew = round(Odelta{1});
else
    Snew=Sold + round(Sdelta{1}/6+Sdelta{2}/3+Sdelta{3}/3+Sdelta{4}/6);
    Enew=Eold + round(Edelta{1}/6+Edelta{2}/3+Edelta{3}/3+Edelta{4}/6);
    IRnew=IRold + round(IRdelta{1}/6+IRdelta{2}/3+IRdelta{3}/3+IRdelta{4}/6);
    IUnew=IUold + round(IUdelta{1}/6+IUdelta{2}/3+IUdelta{3}/3+IUdelta{4}/6);
    Onew = round(Odelta{1}/6+Odelta{2}/3+Odelta{3}/3+Odelta{4}/6);
end
states_new = pack_states(Snew, Enew, IRnew, IUnew, Onew);
prob = 1;

end

function rates = compute_poisson_rates(S, E, IR, IU, Mt, pop, params)

[num_loc, num_ens] = size(S);
[beta, mu, theta, Z, alpha, D] = unpack_params(params); % each param is 1xnum_ens

popp = pop - IR;

U3 =(ones(num_loc,1)*theta).*(Mt*(S./popp)); %ESenter
U4 =(ones(num_loc,1)*theta).*(S./popp).*(sum(Mt)'*ones(1,num_ens)); %ESleft
U4 = min(U4, S);
U7=(ones(num_loc,1)*theta).*(Mt*(E./popp)); %EEenter
U8=(ones(num_loc,1)*theta).*(E./popp).*(sum(Mt)'*ones(1,num_ens)); %EEleft
U8 = min(U8, E);
U11=(ones(num_loc,1)*theta).*(Mt*(IU./popp)); % EIaenter
U12 = (ones(num_loc,1)*theta).*(IU./popp).*(sum(Mt)'*ones(1,num_ens)); % EIaleft 
U12 = min(U12, IU);

U1=(ones(num_loc,1)*beta).*S.*IR./pop;
U2=(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*S.*IU./pop;
U5=(ones(num_loc,1)*alpha).*E./(ones(num_loc,1)*Z);
U6=(ones(num_loc,1)*(1-alpha)).*E./(ones(num_loc,1)*Z);
U9=IR./(ones(num_loc,1)*D);
U10=IU./(ones(num_loc,1)*D);

rates = pack_stats(U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10);;
rates = max(rates, 0);
end


function samples = sample_poisson_noise(rates)
[nloc, nens, nstat] = size(rates);
samples = zeros(nloc, nens, nstat);
for i=1:nstat
    samples(:,:,i) = poissrnd(rates(:,:,i));
end
%{
sz = size(stats);
stats = reshape(stats, [prod(sz), 1]);
samples = poissrnd(stats);
samples = reshape(samples, sz);
%}
end

function [Sdelta, Edelta, IRdelta, IUdelta, Odelta] = compute_deltas(stats)
% each delta is nloc*nens
[U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10] = unpack_stats(stats);
%[U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12] = unpack_stats_ordered(stats);
Sdelta = -U1-U2+U3-U4;
Edelta = U1+U2-U5-U6+U7-U8;
IRdelta = U5-U9;
IUdelta = U6-U10+U11-U12;
Odelta = U5;
end
    

function stats=pack_stats(U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10)
    % stats is num_loc * num_ens * num_stats
    num_stats = 12; 
    % we choose this ordering to match the order of sampling
    % in the original code, to ensure results are identical
    [num_loc, num_ens] = size(U3);
    stats = zeros(num_loc, num_ens, num_stats);
    stats(:,:,1)=U3;  
    stats(:,:,2)=U4; 
    stats(:,:,3)=U7; 
    stats(:,:,4)=U8; 
    stats(:,:,5)=U11; 
    stats(:,:,6)=U12; 
    stats(:,:,7)=U1; 
    stats(:,:,8)=U2; 
    stats(:,:,9)=U5; 
    stats(:,:,10)=U6; 
    stats(:,:,11)=U9; 
    stats(:,:,12)=U10; 
end

function [U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10] = unpack_stats(stats)
    U3 = stats(:,:,1);
    U4 = stats(:,:,2);
    U7 = stats(:,:,3);
    U8 = stats(:,:,4);
    U11 = stats(:,:,5);
    U12 = stats(:,:,6);
    U1 = stats(:,:,7);
    U2 = stats(:,:,8);
    U5 = stats(:,:,9);
    U6 = stats(:,:,10);
    U9 = stats(:,:,11);
    U10 = stats(:,:,12);
end
