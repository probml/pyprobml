function [states_new, states_delta] = sample_from_dynamics(...
    states_old,  params, pop, Mt, add_noise, nsteps, legacy, rounding)
%function [states_new] = sample_from_dynamics(states_old,  model, pop, Mt)
% Input:
% states(l*5,s), l=1:nloc, s=1:nsamples
% pop(l,s)
% model.params, model.add_delay, model.nsteps
% data.pop, data.M (ignores data.obs_truth)
% Output:
% states_new(l*5,s)

if nargin < 7, legacy = true; end
if nargin < 8, rounding = true; end


num_loc = size(Mt, 1);
num_ens = size(states_old, 2);
num_comp = 5;

%params = model.params * ones(1,num_ens);
[Sold, Eold, IRold, IUold, Oold] = unpack_states(states_old);
components_old = pack_components(Sold, Eold, IRold, IUold, Oold);
components_old(:,:,5) = 0; % old observations are not carried over time
components_delta  = zeros(num_loc, num_ens, num_comp);
components_intermediate = components_old;

delta_weights = [2,2,1,1];
rk_weights = [6,3,3,6]; % runge kutte integration weights
%nsteps = model.nsteps;
%add_noise = model.add_noise;

if nsteps==1
    rates = compute_poisson_rates(components_intermediate, Mt, pop, params, 1, legacy);
    if add_noise
        increment = sample_poisson_noise(rates);
    else
        increment = rates;
    end
    components_delta = compute_component_deltas(increment);
    if rounding
        components_delta = round(components_delta);
    end
else
    deltas_step = cell(1, nsteps);
    for stepnum=1:nsteps
        rates = compute_poisson_rates(components_intermediate, Mt, pop, params, stepnum, legacy);
        if add_noise
            increment = sample_poisson_noise(rates);
        else
            increment = round(rates);
        end
        deltas = compute_component_deltas(increment);
        components_intermediate = components_old + deltas / delta_weights(stepnum);
        deltas_step{stepnum} = deltas;
    end
end

if nsteps>1
    for stepnum=1:nsteps
       components_delta = components_delta + deltas_step{stepnum} / rk_weights(stepnum);
    end
    if rounding
        components_delta = round(components_delta);
    end
end
        

components_new = components_old + components_delta;
[S,E,IR,IU,O]  = unpack_components(components_new);
O = round(O);
states_new = pack_states(S,E,IR,IU,O);

[SD,ED,IRD,IUD,OD]  = unpack_components(components_delta);
states_delta = pack_states(SD,ED,IRD,IUD,OD);

end

function components = pack_components(S, E, IR, IU, O)
num_comp = 5;
[num_loc, num_ens] = size(S);
components = zeros(num_loc, num_ens, num_comp);
components(:,:,1) = S;
components(:,:,2) = E;
components(:,:,3) = IR;
components(:,:,4) = IU;
components(:,:,5) = O;
end

function [S, E, IR, IU, O] = unpack_components(components)
S = components(:,:,1);
E = components(:,:,2);
IR = components(:,:,3);
IU = components(:,:,4);
O = components(:,:,5);
end

function rates = compute_poisson_rates(components, Mt, pop, params, step, legacy)
[S, E, IR, IU, ~] = unpack_components(components);
[num_loc, num_ens] = size(S);
[beta, mu, theta, Z, alpha, D] = unpack_params(params); % each param is 1xnum_ens

if legacy
    if (step==1)
        % Incorreclty uses Ia=IU in denominator
        popp = pop - IU;
    else
        % correctl uses Tis=IR in denominator
        popp = pop - IR;
    end
else
    popp = pop - IR;
end
% pop is a nloc x 1 vector
% IR/IU is a nloc x nens matrix
% Matlab will broadcast pop along columns of IU/IR


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


rates = pack_stats(U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10);
rates = max(rates, 0);
end


function samples = sample_poisson_noise(rates_per_stat)
[nloc, nens, nstat] = size(rates_per_stat);
samples = zeros(nloc, nens, nstat);
for i=1:nstat
    samples(:,:,i) = poissrnd(rates_per_stat(:,:,i));
end
%{
sz = size(stats);
stats = reshape(stats, [prod(sz), 1]);
samples = poissrnd(stats);
samples = reshape(samples, sz);
%}
end

function deltas = compute_component_deltas(stats)
% each delta is nloc*nens
[U3, U4, U7, U8, U11, U12, U1, U2, U5, U6, U9, U10] = unpack_stats(stats);
Sdelta = -U1-U2+U3-U4;
Edelta = U1+U2-U5-U6+U7-U8;
IRdelta = U5-U9;
IUdelta = U6-U10+U11-U12;
Odelta = U5;
deltas = pack_components(Sdelta, Edelta, IRdelta, IUdelta, Odelta);
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

