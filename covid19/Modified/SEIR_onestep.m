function [x_new, pop_new]=SEIR_onestep(x,M,pop,ts,pop0)

[states, params] = unpack_x(x);
Mt = M(:,:,ts);
[states_new, pop_new]=step(states, params, pop, Mt, pop0);
x_new = pack_x(states_new, params);

end

function [states_new, pop_new]=step(states, params, pop, Mt, pop0)

num_loc=size(pop,1);
[~,num_ens]=size(states);

[S, E, Is, Ia, ~] = unpack_states(states); % 375 x Nens
[beta, mu, theta, Z, alpha, D] = unpack_params(params); % 6 x Nens

dt1 = 1;

%first step
ESenter=dt1*(ones(num_loc,1)*theta).*(Mt*(S./(pop-Ia)));
ESleft=min(dt1*(ones(num_loc,1)*theta).*(S./(pop-Ia)).*(sum(Mt)'*ones(1,num_ens)),dt1*S);
EEenter=dt1*(ones(num_loc,1)*theta).*(Mt*(E./(pop-Ia)));
EEleft=min(dt1*(ones(num_loc,1)*theta).*(E./(pop-Ia)).*(sum(Mt)'*ones(1,num_ens)),dt1*E);
EIaenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Ia./(pop-Ia)));
EIaleft=min(dt1*(ones(num_loc,1)*theta).*(Ia./(pop-Ia)).*(sum(Mt)'*ones(1,num_ens)),dt1*Ia);

Eexps=dt1*(ones(num_loc,1)*beta).*S.*Is./pop;
Eexpa=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*S.*Ia./pop;
Einfs=dt1*(ones(num_loc,1)*alpha).*E./(ones(num_loc,1)*Z);
Einfa=dt1*(ones(num_loc,1)*(1-alpha)).*E./(ones(num_loc,1)*Z);
Erecs=dt1*Is./(ones(num_loc,1)*D);
Ereca=dt1*Ia./(ones(num_loc,1)*D);

ESenter=max(ESenter,0);ESleft=max(ESleft,0);
EEenter=max(EEenter,0);EEleft=max(EEleft,0);
EIaenter=max(EIaenter,0);EIaleft=max(EIaleft,0);
Eexps=max(Eexps,0);Eexpa=max(Eexpa,0);
Einfs=max(Einfs,0);Einfa=max(Einfa,0);
Erecs=max(Erecs,0);Ereca=max(Ereca,0);

%%%%%%%%%%stochastic version
ESenter=poissrnd(ESenter);ESleft=poissrnd(ESleft);
EEenter=poissrnd(EEenter);EEleft=poissrnd(EEleft);
EIaenter=poissrnd(EIaenter);EIaleft=poissrnd(EIaleft);
Eexps=poissrnd(Eexps);
Eexpa=poissrnd(Eexpa);
Einfs=poissrnd(Einfs);
Einfa=poissrnd(Einfa);
Erecs=poissrnd(Erecs);
Ereca=poissrnd(Ereca);

sk1=-Eexps-Eexpa+ESenter-ESleft;
ek1=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;
Isk1=Einfs-Erecs;
iak1=Einfa-Ereca+EIaenter-EIaleft;
ik1i=Einfs;

%second step
Ts1=S+sk1/2;
Te1=E+ek1/2;
TIs1=Is+Isk1/2;
Tia1=Ia+iak1/2;

ESenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Ts1./(pop-TIs1)));
ESleft=min(dt1*(ones(num_loc,1)*theta).*(Ts1./(pop-TIs1)).*(sum(Mt)'*ones(1,num_ens)),dt1*Ts1);
EEenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Te1./(pop-TIs1)));
EEleft=min(dt1*(ones(num_loc,1)*theta).*(Te1./(pop-TIs1)).*(sum(Mt)'*ones(1,num_ens)),dt1*Te1);
EIaenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Tia1./(pop-TIs1)));
EIaleft=min(dt1*(ones(num_loc,1)*theta).*(Tia1./(pop-TIs1)).*(sum(Mt)'*ones(1,num_ens)),dt1*Tia1);

Eexps=dt1*(ones(num_loc,1)*beta).*Ts1.*TIs1./pop;
Eexpa=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*Ts1.*Tia1./pop;
Einfs=dt1*(ones(num_loc,1)*alpha).*Te1./(ones(num_loc,1)*Z);
Einfa=dt1*(ones(num_loc,1)*(1-alpha)).*Te1./(ones(num_loc,1)*Z);
Erecs=dt1*TIs1./(ones(num_loc,1)*D);
Ereca=dt1*Tia1./(ones(num_loc,1)*D);

ESenter=max(ESenter,0);ESleft=max(ESleft,0);
EEenter=max(EEenter,0);EEleft=max(EEleft,0);
EIaenter=max(EIaenter,0);EIaleft=max(EIaleft,0);
Eexps=max(Eexps,0);Eexpa=max(Eexpa,0);
Einfs=max(Einfs,0);Einfa=max(Einfa,0);
Erecs=max(Erecs,0);Ereca=max(Ereca,0);

%%%%%%%%%%stochastic version
ESenter=poissrnd(ESenter);ESleft=poissrnd(ESleft);
EEenter=poissrnd(EEenter);EEleft=poissrnd(EEleft);
EIaenter=poissrnd(EIaenter);EIaleft=poissrnd(EIaleft);
Eexps=poissrnd(Eexps);
Eexpa=poissrnd(Eexpa);
Einfs=poissrnd(Einfs);
Einfa=poissrnd(Einfa);
Erecs=poissrnd(Erecs);
Ereca=poissrnd(Ereca);

sk2=-Eexps-Eexpa+ESenter-ESleft;
ek2=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;
Isk2=Einfs-Erecs;
iak2=Einfa-Ereca+EIaenter-EIaleft;
ik2i=Einfs;

%third step
Ts2=S+sk2/2;
Te2=E+ek2/2;
TIs2=Is+Isk2/2;
Tia2=Ia+iak2/2;

ESenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Ts2./(pop-TIs2)));
ESleft=min(dt1*(ones(num_loc,1)*theta).*(Ts2./(pop-TIs2)).*(sum(Mt)'*ones(1,num_ens)),dt1*Ts2);
EEenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Te2./(pop-TIs2)));
EEleft=min(dt1*(ones(num_loc,1)*theta).*(Te2./(pop-TIs2)).*(sum(Mt)'*ones(1,num_ens)),dt1*Te2);
EIaenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Tia2./(pop-TIs2)));
EIaleft=min(dt1*(ones(num_loc,1)*theta).*(Tia2./(pop-TIs2)).*(sum(Mt)'*ones(1,num_ens)),dt1*Tia2);

Eexps=dt1*(ones(num_loc,1)*beta).*Ts2.*TIs2./pop;
Eexpa=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*Ts2.*Tia2./pop;
Einfs=dt1*(ones(num_loc,1)*alpha).*Te2./(ones(num_loc,1)*Z);
Einfa=dt1*(ones(num_loc,1)*(1-alpha)).*Te2./(ones(num_loc,1)*Z);
Erecs=dt1*TIs2./(ones(num_loc,1)*D);
Ereca=dt1*Tia2./(ones(num_loc,1)*D);

ESenter=max(ESenter,0);ESleft=max(ESleft,0);
EEenter=max(EEenter,0);EEleft=max(EEleft,0);
EIaenter=max(EIaenter,0);EIaleft=max(EIaleft,0);
Eexps=max(Eexps,0);Eexpa=max(Eexpa,0);
Einfs=max(Einfs,0);Einfa=max(Einfa,0);
Erecs=max(Erecs,0);Ereca=max(Ereca,0);

%%%%%%%%%%stochastic version
ESenter=poissrnd(ESenter);ESleft=poissrnd(ESleft);
EEenter=poissrnd(EEenter);EEleft=poissrnd(EEleft);
EIaenter=poissrnd(EIaenter);EIaleft=poissrnd(EIaleft);
Eexps=poissrnd(Eexps);
Eexpa=poissrnd(Eexpa);
Einfs=poissrnd(Einfs);
Einfa=poissrnd(Einfa);
Erecs=poissrnd(Erecs);
Ereca=poissrnd(Ereca);

sk3=-Eexps-Eexpa+ESenter-ESleft;
ek3=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;
Isk3=Einfs-Erecs;
iak3=Einfa-Ereca+EIaenter-EIaleft;
ik3i=Einfs;

%fourth step
Ts3=S+sk3;
Te3=E+ek3;
TIs3=Is+Isk3;
Tia3=Ia+iak3;

ESenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Ts3./(pop-TIs3)));
ESleft=min(dt1*(ones(num_loc,1)*theta).*(Ts3./(pop-TIs3)).*(sum(Mt)'*ones(1,num_ens)),dt1*Ts3);
EEenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Te3./(pop-TIs3)));
EEleft=min(dt1*(ones(num_loc,1)*theta).*(Te3./(pop-TIs3)).*(sum(Mt)'*ones(1,num_ens)),dt1*Te3);
EIaenter=dt1*(ones(num_loc,1)*theta).*(Mt*(Tia3./(pop-TIs3)));
EIaleft=min(dt1*(ones(num_loc,1)*theta).*(Tia3./(pop-TIs3)).*(sum(Mt)'*ones(1,num_ens)),dt1*Tia3);

Eexps=dt1*(ones(num_loc,1)*beta).*Ts3.*TIs3./pop;
Eexpa=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1)*beta).*Ts3.*Tia3./pop;
Einfs=dt1*(ones(num_loc,1)*alpha).*Te3./(ones(num_loc,1)*Z);
Einfa=dt1*(ones(num_loc,1)*(1-alpha)).*Te3./(ones(num_loc,1)*Z);
Erecs=dt1*TIs3./(ones(num_loc,1)*D);
Ereca=dt1*Tia3./(ones(num_loc,1)*D);

ESenter=max(ESenter,0);ESleft=max(ESleft,0);
EEenter=max(EEenter,0);EEleft=max(EEleft,0);
EIaenter=max(EIaenter,0);EIaleft=max(EIaleft,0);
Eexps=max(Eexps,0);Eexpa=max(Eexpa,0);
Einfs=max(Einfs,0);Einfa=max(Einfa,0);
Erecs=max(Erecs,0);Ereca=max(Ereca,0);

%%%%%%%%%%stochastic version
ESenter=poissrnd(ESenter);ESleft=poissrnd(ESleft);
EEenter=poissrnd(EEenter);EEleft=poissrnd(EEleft);
EIaenter=poissrnd(EIaenter);EIaleft=poissrnd(EIaleft);
Eexps=poissrnd(Eexps);
Eexpa=poissrnd(Eexpa);
Einfs=poissrnd(Einfs);
Einfa=poissrnd(Einfa);
Erecs=poissrnd(Erecs);
Ereca=poissrnd(Ereca);

sk4=-Eexps-Eexpa+ESenter-ESleft;
ek4=Eexps+Eexpa-Einfs-Einfa+EEenter-EEleft;
Isk4=Einfs-Erecs;
iak4=Einfa-Ereca+EIaenter-EIaleft;
ik4i=Einfs;

%%%%%
S_new=S+round(sk1/6+sk2/3+sk3/3+sk4/6);
E_new=E+round(ek1/6+ek2/3+ek3/3+ek4/6);
Is_new=Is+round(Isk1/6+Isk2/3+Isk3/3+Isk4/6);
Ia_new=Ia+round(iak1/6+iak2/3+iak3/3+iak4/6);
Incidence_new=round(ik1i/6+ik2i/3+ik3i/3+ik4i/6);
obs_new=Incidence_new;

states_new = pack_states(S_new, E_new, Is_new, Ia_new, obs_new);

%%%update pop
pop_new = pop - sum(Mt,1)'*theta + sum(Mt,2)*theta;
minfrac=0.6;
ndx = find(pop_new < minfrac*pop0);
pop_new(ndx)=pop0(ndx)*minfrac;

end