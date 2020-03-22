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

[S, E, IR, IU, O] = unpack_states(states); % nloc * nens
% S = suspectible (original name Ts)
% E = exposed (original name Te)
% IR = infected reported (original name TIs)
% IU = infected unreported (original name Tia)

% first step of RK4
first_step = true;

stats = compute_stats(S, E, IR, IU, Mt, pop, params, first_step);
stats = sample_stats(stats);
[S1delta, E1delta, IR1delta, IU1delta, O1delta] = compute_deltas(stats);

S1=S+S1delta/2;
E1=E+E1delta/2;
IR1=IR+IR1delta/2;
IU1=IU+IU1delta/2;


%second step
first_step = false;
stats = compute_stats(S1, E1, IR1, IU1, Mt, pop, params, first_step);
stats = sample_stats(stats);
[S2delta, E2delta, IR2delta, IU2delta, O2delta] = compute_deltas(stats);


S2=S+S2delta/2;
E2=E+E2delta/2;
IR2=IR+IR2delta/2;
IU2=IU+IU2delta/2;


%third step

stats = compute_stats(S2, E2, IR2, IU2, Mt, pop, params, first_step);
stats = sample_stats(stats); 
[S3delta, E3delta, IR3delta, IU3delta, O3delta] = compute_deltas(stats); 

S3=S+S3delta;
E3=E+E3delta;
IR3=IR+IR3delta;
IU3=IU+IU3delta;


%fourth step
stats = compute_stats(S3, E3, IR3, IU3, Mt, pop, params, first_step);
stats = sample_stats(stats); 
[S4delta, E4delta, IR4delta, IU4delta, O4delta] = compute_deltas(stats); 


%%%%% Compute final states
S_new=S+round(S1delta/6+S2delta/3+S3delta/3+S4delta/6);
E_new=E+round(E1delta/6+E2delta/3+E3delta/3+E4delta/6);
IR_new=IR+round(IR1delta/6+IR2delta/3+IR3delta/3+IR4delta/6);
IU_new=IU+round(IU1delta/6+IU2delta/3+IU3delta/3+IU4delta/6);
Incidence_new=round(O1delta/6+O2delta/3+O3delta/3+O4delta/6);
obs_new=Incidence_new;
states_new = pack_states(S_new, E_new, IR_new, IU_new, obs_new);

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
    

function stats = compute_stats(S, E, IR, IU, Mt, pop, params, step1)
[num_loc, num_ens] = size(S);
[beta, mu, theta, Z, alpha, D] = unpack_params(params); % each param is 1xnum_ens

if step1
    ESenter=(ones(num_loc,1)*theta).*(Mt*(S./(pop-IU)));
    ESleft=min((ones(num_loc,1)*theta).*(S./(pop-IU)).*(sum(Mt)'*ones(1,num_ens)),S);
    EEenter=(ones(num_loc,1)*theta).*(Mt*(E./(pop-IU)));
    EEleft=min((ones(num_loc,1)*theta).*(E./(pop-IU)).*(sum(Mt)'*ones(1,num_ens)),E);
    EIaenter=(ones(num_loc,1)*theta).*(Mt*(IU./(pop-IU)));
    EIaleft=min((ones(num_loc,1)*theta).*(IU./(pop-IU)).*(sum(Mt)'*ones(1,num_ens)),IU);
else
    ESenter=(ones(num_loc,1)*theta).*(Mt*(S./(pop-IR)));
    ESleft=min((ones(num_loc,1)*theta).*(S./(pop-IR)).*(sum(Mt)'*ones(1,num_ens)),S);
    EEenter=(ones(num_loc,1)*theta).*(Mt*(E./(pop-IR)));
    EEleft=min((ones(num_loc,1)*theta).*(E./(pop-IR)).*(sum(Mt)'*ones(1,num_ens)),E);
    EIaenter=(ones(num_loc,1)*theta).*(Mt*(IU./(pop-IU)));
    EIaleft=min((ones(num_loc,1)*theta).*(IU./(pop-IR)).*(sum(Mt)'*ones(1,num_ens)),IU);
end

Eexps=(ones(num_loc,1)*beta).*S.*IR./pop;
Eexpa=(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*S.*IU./pop;
Einfs=(ones(num_loc,1)*alpha).*E./(ones(num_loc,1)*Z);
Einfa=(ones(num_loc,1)*(1-alpha)).*E./(ones(num_loc,1)*Z);
Erecs=IR./(ones(num_loc,1)*D);
Ereca=IU./(ones(num_loc,1)*D);

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

function [Sdelta, Edelta, IRdelta, IUdelta, Odelta] = compute_deltas(stats)
% each delta is nloc*nens
[ESenter, ESleft, EEenter, EEleft, EIaenter, EIaleft, Eexps, Eexpa, Einfs, Einfa, Erecs, Ereca] = unpack_stats(stats);
Sdelta = -Eexps-Eexpa+ESenter-ESleft;
Edelta = Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;
IRdelta = Einfs-Erecs;
IUdelta = Einfa-Ereca+EIaenter-EIaleft;
Odelta = Einfs;
end
    

function [sk, ek, Isk, iak, ik] = compute_deltas2(stats)
% each delta is nloc*nens
[ESenter, ESleft, EEenter, EEleft, EIaenter, EIaleft, Eexps, Eexpa, Einfs, Einfa, Erecs, Ereca] = unpack_stats(stats);
sk=-Eexps-Eexpa+ESenter-ESleft;
ek=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;
Isk=Einfs-Erecs;
iak=Einfa-Ereca+EIaenter-EIaleft;
ik=Einfs;
end