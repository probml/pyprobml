function pop_new = update_pop(pop_locs_ens_t, Mt, theta, pop_locs)

num_ens = size(pop_locs_ens_t,2);
pop_locs_ens_0 = pop_locs * ones(1,num_ens);
pop_new = pop_locs_ens_t + sum(Mt,2)*theta - sum(Mt,1)'*theta;  % eqn 5
minfrac=0.6;
ndx = find(pop_new < minfrac*pop_locs_ens_0);
pop_new(ndx)=pop_locs_ens_0(ndx)*minfrac;

end
    