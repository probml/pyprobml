function z = checkbound_states(z, pop_t)
% z_t has size (375*5, num_ens)
% The 5 refers to S,E,Is,Ia,obs for each locn
num_loc=size(pop_t,1);
for i=1:num_loc
    %S
    ndx = (z((i-1)*5+1,:)<0); % indices of samples where  below 0
    z((i-1)*5+1,  ndx)=0;
    ndx = (z((i-1)*5+1,:)>pop_t(i,:)); % indices of samples where exceed populatoion
    z((i-1)*5+1, ndx)=pop_t(i,ndx);
    %E
    z((i-1)*5+2,z((i-1)*5+2,:)<0)=0;
    %Ir
    z((i-1)*5+3,z((i-1)*5+3,:)<0)=0;
    %Iu
    z((i-1)*5+4,z((i-1)*5+4,:)<0)=0;
    %obs
    z((i-1)*5+5,z((i-1)*5+5,:)<0)=0;
end