function x = checkbound(x,pop)
%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D
betalow=0.8;betaup=1.5;%transmission rate
mulow=0.2;muup=1.0;%asymptomatic factor
thetalow=1;thetaup=1.75;%movement factor
Zlow=2;Zup=5;%incubation time
alphalow=0.02;alphaup=1.0;%symptomatic rate
Dlow=2;Dup=5;%infectious time
xmin=[betalow;mulow;thetalow;Zlow;alphalow;Dlow];
xmax=[betaup;muup;thetaup;Zup;alphaup;Dup];
num_loc=size(pop,1);
for i=1:num_loc
    %S
    x((i-1)*5+1,x((i-1)*5+1,:)<0)=0;
    x((i-1)*5+1,x((i-1)*5+1,:)>pop(i,:))=pop(i,x((i-1)*5+1,:)>pop(i,:));
    %E
    x((i-1)*5+2,x((i-1)*5+2,:)<0)=0;
    %Ir
    x((i-1)*5+3,x((i-1)*5+3,:)<0)=0;
    %Iu
    x((i-1)*5+4,x((i-1)*5+4,:)<0)=0;
    %obs
    x((i-1)*5+5,x((i-1)*5+5,:)<0)=0;
end
for i=1:6
    x(end-6+i,x(end-6+i,:)<xmin(i))=xmin(i)*(1+0.1*rand(sum(x(end-6+i,:)<xmin(i)),1));
    x(end-6+i,x(end-6+i,:)>xmax(i))=xmax(i)*(1-0.1*rand(sum(x(end-6+i,:)>xmax(i)),1));
end
end
