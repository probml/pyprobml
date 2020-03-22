function x = checkbound_states(x,pop)
% x has size (375*5, num_ens)
%S,E,Is,Ia,obs for each locn
num_loc=size(pop,1);
for i=1:num_loc
    %S
    ndx = (x((i-1)*5+1,:)<0); % indices of samples where  below 0
    x((i-1)*5+1,  ndx)=0;
    ndx = (x((i-1)*5+1,:)>pop(i,:)); % indices of samples where exceed populatoion
    x((i-1)*5+1, ndx)=pop(i,ndx);
    %E
    x((i-1)*5+2,x((i-1)*5+2,:)<0)=0;
    %Ir
    x((i-1)*5+3,x((i-1)*5+3,:)<0)=0;
    %Iu
    x((i-1)*5+4,x((i-1)*5+4,:)<0)=0;
    %obs
    x((i-1)*5+5,x((i-1)*5+5,:)<0)=0;
end