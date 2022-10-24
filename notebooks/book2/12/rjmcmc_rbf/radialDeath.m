function [k,mu,M,aDeath,rDeath] = radialDeath(aDeath,rDeath,k,mu,M,delta,x,y,hyper,t,nabla);
% PURPOSE : Performs the death move of the reversible jump MCMC algorithm.
% INPUTS  : - aDeath: Number of times the death move has been accepted.
%           - rDeath: Number of times the death move has been rejected.
%           - k : Number of basis functions.
%           - mu : Basis functions centres.
%           - M : Regressors matrix.
%           - delta : Signal to noise ratio.
%           - x : Input data.
%           - y : Target data.
%           - hyper: hyperparameters.
%           - t : Current time step.
%           - nabla: Hyperparameter for k.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

if nargin < 11, error('Not enough input arguments.'); end
[N,d] = size(x);      % N = number of data, d = dimension of x.
[N,c] = size(y);      % c = dimension of y, i.e. number of outputs.
insideDeath=1;
uD=rand(1);

% INITIALISE H AND P MATRICES:
% ===========================
invH=zeros(k(t)+1+d,k(t)+1+d,c);
P=zeros(N,N,c);
invHproposal=zeros(k(t)+d,k(t)+d,c);
Pproposal=zeros(N,N,c);
for i=1:c,
  invH(:,:,i) = (M'*M + (1/delta(t,i))*eye(k(t)+1+d));
  P(:,:,i) = eye(N) - M*inv(invH(:,:,i))*M';
end;

% CHOOSE UNIFORMLY A BASIS FUNCTION TO BE DELETED:
% ===============================================
proposalPos= d+1+unidrnd(length(mu{t}(:,1)),1,1);
if (proposalPos==d+1+k(t)),
  Mproposal = [M(:,1:proposalPos-1)];     
else
  Mproposal = [M(:,1:proposalPos-1) M(:,proposalPos+1:k(t)+d+1)];      
end;
for i=1:c,
  invHproposal(:,:,i) = (Mproposal'*Mproposal + inv(delta(t,i))*eye(k(t)+d)); 
  Pproposal(:,:,i) = eye(N) - Mproposal*inv(invHproposal(:,:,i))*Mproposal'; 
end;

% PERFORM A DEATH MOVE:
% ====================
ratio= k(t) * sqrt(delta(t,1)) * sqrt((det(invH(:,:,1)))/(det(invHproposal(:,:,1)))) * ((hyper.gamma+y(:,1)'*P(:,:,1)*y(:,1))/(hyper.gamma + y(:,1)'*Pproposal(:,:,1)*y(:,1)))^((hyper.v+N)/2);  
for i=2:c,
  ratio= ratio * sqrt(delta(t,i)) * sqrt((det(invH(:,:,i)))/(det(invHproposal(:,:,i)))) * ((hyper.gamma+y(:,i)'*P(:,:,i)*y(:,i))/(hyper.gamma + y(:,i)'*Pproposal(:,:,i)*y(:,i)))^((hyper.v+N)/2);  
end;
acceptance = min(1,ratio);  
if (uD<acceptance),
  previousMu = mu{t};
  if (proposalPos==(1+d+1)),
    mu{t+1} = [previousMu(2:k(t),:)]; 
  elseif (proposalPos==(1+d+k(t))),
    mu{t+1} = [previousMu(1:k(t)-1,:)];
  else
    mu{t+1} = [previousMu(1:proposalPos-1-d-1,:); previousMu(proposalPos-d-1+1:k(t),:)];
  end;
  k(t+1) = k(t)-1;
  M=Mproposal;
  aDeath=aDeath+1;
else
  mu{t+1} = mu{t};
  k(t+1) = k(t);
  rDeath=rDeath+1;
  M=M;
end;










