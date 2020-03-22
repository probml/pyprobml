function [paramin, paramax] = param_bounds()

%beta,mu,theta,Z,alpha,D
betalow=0.8;betaup=1.5;%transmission rate
mulow=0.2;muup=1.0;%asymptomatic factor
thetalow=1;thetaup=1.75;%movement factor
Zlow=2;Zup=5;%incubation time
alphalow=0.02;alphaup=1.0;%symptomatic rate
Dlow=2;Dup=5;%infectious time
paramin=[betalow;mulow;thetalow;Zlow;alphalow;Dlow];
paramax=[betaup;muup;thetaup;Zup;alphaup;Dup];
    
end
