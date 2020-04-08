function states = pack_states(S,E,Is,Ia,obs)
    %states: 5 vars (S,E,Is,Ia,obs) for each locn, by nensembles

    [num_loc, nens] = size(S);
states = zeros(num_loc*5, nens);

Sidx=(1:5:5*num_loc)';
Eidx=(2:5:5*num_loc)';
Isidx=(3:5:5*num_loc)';
Iaidx=(4:5:5*num_loc)';
obsidx=(5:5:5*num_loc)';

states(Sidx,:)=S;
states(Eidx,:)=E;
states(Isidx,:)=Is;
states(Iaidx,:)=Ia;
states(obsidx,:)=obs;
end