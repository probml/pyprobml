function [x_new, pop_new] = SEIR_onestep(x,M,pop,ts,pop0)

[states, params] = unpack_x(x);
Mt = M(:,:,ts);
[states_new] = integrate_ODE_onestep(states, params, pop, Mt);

[beta, mu, theta, Z, alpha, D] = unpack_params(params); % each param is 1xnum_ens
pop_new = pop + sum(Mt,2)*theta - sum(Mt,1)'*theta;  % eqn 5
minfrac=0.6;
ndx = find(pop_new < minfrac*pop0);
pop_new(ndx)=pop0(ndx)*minfrac;

x_new = pack_x(states_new, params);

end

function [states_new] = integrate_ODE_onestep(states, params, pop, Mt)
% Integrates the ODE eqns 1-4 for one time step using RK4 method

[S, E, Is, Ia, ~] = unpack_states(states); % nloc * nens

% first step of RK4
Ts0=S;
Te0=E;
TIs0=Is; 
Tia0=Ia;
first_step = true;

stats = compute_stats(Ts0, Te0, TIs0, Tia0, Mt, pop, params, first_step);
stats = sample_stats(stats);
[sk1, ek1, Isk1, iak1, ik1i] = compute_deltas(stats);

%second step
Ts1=S+sk1/2;
Te1=E+ek1/2;
TIs1=Is+Isk1/2;
Tia1=Ia+iak1/2;
first_step = false;

stats = compute_stats(Ts1, Te1, TIs1, Tia1, Mt, pop, params, first_step);
stats = sample_stats(stats);
[sk2, ek2, Isk2, iak2, ik2i] = compute_deltas(stats); 

%third step
Ts2=S+sk2/2;
Te2=E+ek2/2;
TIs2=Is+Isk2/2;
Tia2=Ia+iak2/2;

stats = compute_stats(Ts2, Te2, TIs2, Tia2, Mt, pop, params, first_step);
stats = sample_stats(stats);
[sk3, ek3, Isk3, iak3, ik3i] = compute_deltas(stats); 
%fourth step
Ts3=S+sk3;
Te3=E+ek3;
TIs3=Is+Isk3;
Tia3=Ia+iak3;

stats = compute_stats(Ts3, Te3, TIs3, Tia3, Mt, pop, params, first_step);
stats = sample_stats(stats);
[sk4, ek4, Isk4, iak4, ik4i] = compute_deltas(stats); 


%%%%% Compute final states
S_new=S+round(sk1/6+sk2/3+sk3/3+sk4/6);
E_new=E+round(ek1/6+ek2/3+ek3/3+ek4/6);
Is_new=Is+round(Isk1/6+Isk2/3+Isk3/3+Isk4/6);
Ia_new=Ia+round(iak1/6+iak2/3+iak3/3+iak4/6);
Incidence_new=round(ik1i/6+ik2i/3+ik3i/3+ik4i/6);
obs_new=Incidence_new;
states_new = pack_states(S_new, E_new, Is_new, Ia_new, obs_new);

end


function [ESenter, ESleft, EEenter, EEleft, EIaenter, EIaleft, Eexps, Eexpa, Einfs, Einfa, Erecs, Ereca] = unpack_stats(stats)
    ESenter = stats(:,:,1);
    ESleft = stats(:,:,2);
    EEenter = stats(:,:,3);
    EEleft = stats(:,:,4);
    EIaenter = stats(:,:,5);
    EIaleft = stats(:,:,6);
    Eexps = stats(:,:,7);
    Eexpa = stats(:,:,8);
    Einfs = stats(:,:,9);
    Einfa = stats(:,:,10);
    Erecs = stats(:,:,11);
    Ereca = stats(:,:,12);
end

function stats=pack_stats(ESenter, ESleft, EEenter, EEleft, EIaenter, EIaleft, Eexps, Eexpa, Einfs, Einfa, Erecs, Ereca)
    % stats is num_loc * num_ens * num_stats
    num_stats = 12; % U1...U12 in paper
    [num_loc, num_ens] = size(ESenter);
    stats = zeros(num_loc, num_ens, num_stats);
    stats(:,:,1)=ESenter; % U3 
    stats(:,:,2)=ESleft; % U4
    stats(:,:,3)=EEenter; % U7
    stats(:,:,4)=EEleft; % U8
    stats(:,:,5)=EIaenter; % U11
    stats(:,:,6)=EIaleft; % U12
    stats(:,:,7)=Eexps; % U1 
    stats(:,:,8)=Eexpa; % U2 
    stats(:,:,9)=Einfs; % U5
    stats(:,:,10)=Einfa; % U6
    stats(:,:,11)=Erecs; % U9
    stats(:,:,12)=Ereca; % U10
end
    


function stats = compute_stats(Ts, Te, TIs, Tia, Mt, pop, params, step1)
[num_loc, num_ens] = size(Ts);
[beta, mu, theta, Z, alpha, D] = unpack_params(params); % each param is 1xnum_ens

if step1
    ESenter=(ones(num_loc,1)*theta).*(Mt*(Ts./(pop-Tia)));
    ESleft=min((ones(num_loc,1)*theta).*(Ts./(pop-Tia)).*(sum(Mt)'*ones(1,num_ens)),Ts);
    EEenter=(ones(num_loc,1)*theta).*(Mt*(Te./(pop-Tia)));
    EEleft=min((ones(num_loc,1)*theta).*(Te./(pop-Tia)).*(sum(Mt)'*ones(1,num_ens)),Te);
    EIaenter=(ones(num_loc,1)*theta).*(Mt*(Tia./(pop-Tia)));
    EIaleft=min((ones(num_loc,1)*theta).*(Tia./(pop-Tia)).*(sum(Mt)'*ones(1,num_ens)),Tia);
else
    ESenter=(ones(num_loc,1)*theta).*(Mt*(Ts./(pop-TIs)));
    ESleft=min((ones(num_loc,1)*theta).*(Ts./(pop-TIs)).*(sum(Mt)'*ones(1,num_ens)),Ts);
    EEenter=(ones(num_loc,1)*theta).*(Mt*(Te./(pop-TIs)));
    EEleft=min((ones(num_loc,1)*theta).*(Te./(pop-TIs)).*(sum(Mt)'*ones(1,num_ens)),Te);
    EIaenter=(ones(num_loc,1)*theta).*(Mt*(Tia./(pop-TIs)));
    EIaleft=min((ones(num_loc,1)*theta).*(Tia./(pop-TIs)).*(sum(Mt)'*ones(1,num_ens)),Tia);
end

Eexps=(ones(num_loc,1)*beta).*Ts.*TIs./pop;
Eexpa=(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*Ts.*Tia./pop;
Einfs=(ones(num_loc,1)*alpha).*Te./(ones(num_loc,1)*Z);
Einfa=(ones(num_loc,1)*(1-alpha)).*Te./(ones(num_loc,1)*Z);
Erecs=TIs./(ones(num_loc,1)*D);
Ereca=Tia./(ones(num_loc,1)*D);

stats = pack_stats(ESenter, ESleft, EEenter, EEleft, EIaenter, EIaleft, Eexps, Eexpa, Einfs, Einfa, Erecs, Ereca);
stats = max(stats, 0);
end

function samples = sample_stats(stats)
[nloc, nens, nstat] = size(stats);
samples = zeros(nloc, nens, nstat);
for i=1:nstat
    samples(:,:,i) = poissrnd(stats(:,:,i));
end
%{
sz = size(stats);
stats = reshape(stats, [prod(sz), 1]);
samples = poissrnd(stats);
samples = reshape(samples, sz);
%}
end

function [sk, ek, Isk, iak, ik] = compute_deltas(stats)
% each delta is nloc*nens
[ESenter, ESleft, EEenter, EEleft, EIaenter, EIaleft, Eexps, Eexpa, Einfs, Einfa, Erecs, Ereca] = unpack_stats(stats);
sk=-Eexps-Eexpa+ESenter-ESleft;
ek=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;
Isk=Einfs-Erecs;
iak=Einfa-Ereca+EIaenter-EIaleft;
ik=Einfs;
end
    