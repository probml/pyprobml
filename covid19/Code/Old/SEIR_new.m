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


%{
Original code used these names

    % U3 = ESenter
    % U4 = ESleft
     % U7 = EEenter
    % U8 = EEleft
    % U11 = EIaenter
     % U12 = EIaleft
     % U1 = Eexpds
    % U2  = Eexpa
   % U5 = Einfs
     % U6 = Einfa
     % U9 = Erecs
     % U10 = Ereca
%}

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

function stats=pack_stats_ordered(U1, U2, U3, U4, U5, U6, U7, U8,U9, U10, U11, U12)
    % stats is num_loc * num_ens * num_stats
    num_stats = 12; 
    [num_loc, num_ens] = size(U1);
    stats = zeros(num_loc, num_ens, num_stats);
    stats(:,:,1)=U1;  
    stats(:,:,2)=U2; 
    stats(:,:,3)=U3; 
    stats(:,:,4)=U4; 
    stats(:,:,5)=U5; 
    stats(:,:,6)=U6; 
    stats(:,:,7)=U7; 
    stats(:,:,8)=U8; 
    stats(:,:,9)=U9; 
    stats(:,:,10)=U10; 
    stats(:,:,11)=U11; 
    stats(:,:,12)=U12; 
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


function [U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12] = unpack_stats_ordered(stats)
    U1 = stats(:,:,1);
    U2 = stats(:,:,2);
    U3 = stats(:,:,3);
    U4 = stats(:,:,4);
    U5 = stats(:,:,5);
    U6 = stats(:,:,6);
    U7 = stats(:,:,7);
    U8 = stats(:,:,8);
    U9 = stats(:,:,9);
    U10 = stats(:,:,10);
    U11 = stats(:,:,11);
    U12 = stats(:,:,12);
end
    

function stats = compute_stats(S, E, IR, IU, Mt, pop, params, step1)
[num_loc, num_ens] = size(S);
[beta, mu, theta, Z, alpha, D] = unpack_params(params); % each param is 1xnum_ens

if step1
    U3=(ones(num_loc,1)*theta).*(Mt*(S./(pop-IU)));
    U4=min((ones(num_loc,1)*theta).*(S./(pop-IU)).*(sum(Mt)'*ones(1,num_ens)),S);
    U7=(ones(num_loc,1)*theta).*(Mt*(E./(pop-IU)));
    U8=min((ones(num_loc,1)*theta).*(E./(pop-IU)).*(sum(Mt)'*ones(1,num_ens)),E);
    
    U11=(ones(num_loc,1)*theta).*(Mt*(IU./(pop-IU)));
    % According to the paper, U11 should have IR in denominator: 
    %U11=(ones(num_loc,1)*theta).*(Mt*(IU./(pop-IR)));
    % However, according to the original code, for step 1, U11 is given by
    %EIaenter=dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Ia(:,:,tcnt)./(pop-Ia(:,:,tcnt))));
    % where Ia=IU and Is=IR
    
    U12 = (ones(num_loc,1)*theta).*(IU./(pop-IU)).*(sum(Mt)'*ones(1,num_ens));
    U12 = min(U12, IU);
    %EIaleft=min(dt1*(ones(num_loc,1)*theta).*(Ia(:,:,tcnt)./(pop-Ia(:,:,tcnt))).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Ia(:,:,tcnt));
     
else
    U3=(ones(num_loc,1)*theta).*(Mt*(S./(pop-IR)));
    U4=min((ones(num_loc,1)*theta).*(S./(pop-IR)).*(sum(Mt)'*ones(1,num_ens)),S);
    U7=(ones(num_loc,1)*theta).*(Mt*(E./(pop-IR)));
    U8=min((ones(num_loc,1)*theta).*(E./(pop-IR)).*(sum(Mt)'*ones(1,num_ens)),E);
    
    U11=(ones(num_loc,1)*theta).*(Mt*(IU./(pop-IR)));
    % U11(i,e) = theta(i,e) * sum_j M(i,j) U(j,e)/pop(j,e)-R(j,e)
    %EIaenter=dt1*(ones(num_loc,1)*theta).*(M(:,:,ts)*(Tia1./(pop-Tis1)));
     
    U12 = (ones(num_loc,1)*theta).*(IU./(pop-IR)).*(sum(Mt)'*ones(1,num_ens));
    U12 = min(U12, IU);
    % According to the paper, U12 should have IU in denominator:
    % U12=min((ones(num_loc,1)*theta).*(IU./(pop-IU)).*(sum(Mt)'*ones(1,num_ens)),IU);
    % However, according to the original code, for step >1, U12 is given by
    %  EIaleft=min(dt1*(ones(num_loc,1)*theta).*(Tia1./(pop-Tis1)).*(sum(M(:,:,ts))'*ones(1,num_ens)),dt1*Tia1);
end

U1=(ones(num_loc,1)*beta).*S.*IR./pop;
U2=(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*S.*IU./pop;
U5=(ones(num_loc,1)*alpha).*E./(ones(num_loc,1)*Z);
U6=(ones(num_loc,1)*(1-alpha)).*E./(ones(num_loc,1)*Z);
U9=IR./(ones(num_loc,1)*D);
U10=IU./(ones(num_loc,1)*D);

stats = pack_stats(U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10);
%stats = pack_stats_ordered(U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12);
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
[U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10] = unpack_stats(stats);
%[U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12] = unpack_stats_ordered(stats);
Sdelta = -U1-U2+U3-U4;
Edelta = U1+U2-U5-U6+U7-U8;
IRdelta = U5-U9;
IUdelta = U6-U10+U11-U12;
Odelta = U5;
end
    