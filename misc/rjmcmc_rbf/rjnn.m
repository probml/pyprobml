function [k,mu,alpha,sigma,nabla,delta,ypred,ypredv,post] = rjnn(x,y,chainLength,Ndata,bFunction,par,xv,yv);
% PURPOSE : Computes the parameters and number of parameters of a radial basis function (RBF)
%           network using the reversible jump MCMC algorithm. Please have a 
%           look at the paper first.
% INPUTS  : - x : Input data.
%           - y : Target data.
%           - chainLength: Number of iterations of the Markov chain.
%           - Ndata: Number of time steps in the training data set.
%           - bFunction: Type of basis function.
%           - par: Record of simulation parameters (see defaults).
%           - {xv,yv}: Validation data (optional).
% OUTPUTS : - k : Model order.
%           - mu: Basis centres.
%           - alpha: Coefficients + linear weights (see paper).
%           - sigma: Measurement noise variance.
%           - nabla: Hyperparameter for k.
%           - delta: Signal to noise ratio.
%           - ypred: Prediction on the train set.
%           - ypredv: Prediction on the test set.
%           - post: Log of the joint posterior density.

% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

% CHECK INPUTS AND SET DEFAULTS:
% =============================
if nargin < 5, error('Not enough input arguments.'); end;
if ((nargin==5) | (nargin==7)),
  if nargin == 5
    Validation = 0;
  else
    Validation = 1;
  end;
  hyper.a = 2;                      % Hyperparameter for delta.  
  hyper.b = 10;                     % Hyperparameter for delta.
  hyper.e1 = 0.0001;                % Hyperparameter for nabla.    
  hyper.e2 = 0.0001;                % Hyperparameter for nabla.   
  hyper.v = 0;                      % Hyperparameter for sigma    
  hyper.gamma = 0;                  % Hyperparameter for sigma. 
  kMax = 50;                        % Maximum number of basis.
  arbC = 0.5;                       % Constant for birth and death moves.
  doPlot = 1;                       % To plot or not to plot? Thats ...
  sigStar = .1;                     % Merge-split parameter.
  sWalk = .001;
  Lambda = .5;
  walkPer = 0.1;
elseif ((nargin==6) | (nargin==8))
  if nargin == 6
    Validation = 0;
  else
    Validation = 1;
  end;
  hyper.a = par.a;                 
  hyper.b = par.b;                
  hyper.e1 = par.e1;           
  hyper.e2 = par.e2;           
  hyper.v = par.v;                 
  hyper.gamma = par.gamma;             
  kMax = par.kMax;                   
  arbC = par.arbC;
  doPlot = par.doPlot;    
  sigStar = par.merge;
  sWalk = par.sRW;
  Lambda = par.Lambda;
  walkPer = par.walkPer;
else
 error('Wrong Number of input arguments.');
end;
if Validation,
  [Nv,dv] = size(xv);   % Nv = number of test data, dv = dimension of xv.
end;
[N,d] = size(x);      % N = number of train data, d = dimension of x.
[N,c] = size(y);      % c = dimension of y, i.e. number of outputs.
if Ndata ~= N, error('input must me N by d and output N by c.'); end;

% INITIALISATION:
% ==============
post = ones(chainLength,1);       % p(centres,k|y).
if Validation,
  ypredv = zeros(Nv,c,chainLength);  % Output fit (test set).
end;
ypred = zeros(N,c,chainLength);   % Output fit (train set).
nabla = zeros(chainLength,1);     % Poisson parameter.
delta = zeros(chainLength,c);     % Regularisation parameter.
k = ones(chainLength,1);          % Model order - number of basis.
sigma = ones(chainLength,c);      % Output noise variance.
mu = cell(chainLength,1);         % Radial basis centres.
alpha = cell(chainLength,c);      % Radial basis coefficients.

% DEFINE WALK INTERVAL FOR MU:
% ===========================
walk = walkPer*(max(x)-min(x));
walkInt=zeros(d,1);
for i=1:d,
  walkInt(i,1) = (max(x(:,i))-min(x(:,i))) + 2*walk(i);
end;

% SAMPLE INITIAL CONDITIONS FROM THEIR PRIORS:
% ===========================================
nabla(1) = gengamma(0.5 + hyper.e1,hyper.e2);
k(1) = poissrnd(nabla(1));
k(1) = 40;                              % TEMPORARY: for demo1 comparison.
k(1) = max(k(1),1);
k(1) = min(k(1),kMax);
for i=1:c
  delta(1,i) = inv(gengamma(hyper.a,hyper.b));
  sigma(1,i) = inv(gengamma(hyper.v/2,hyper.gamma/2));
  alpha{1,i} = mvnrnd(zeros(1,k(1)+d+1),sigma(1,i)*delta(1,i)*eye(k(1)+d+1),1)';
end;

% DRAW THE INITIAL RADIAL CENTRES:
% ===============================
mu{1}=zeros(k(1),d);
for i=1:d,
  mu{1}(:,i)= (min(x(:,i))-walk(i))*ones(k(1),1) + ((max(x(:,i))+walk(i))-(min(x(:,i))-walk(i)))*rand(k(1),1);
end;

% FILL THE REGRESSION MATRIX:
% ==========================
M=zeros(N,k(1)+d+1);
M(:,1) = ones(N,1);
M(:,2:d+1) = x;
for j=d+2:k(1)+d+1,
  M(:,j) = feval(bFunction,mu{1}(j-d-1,:),x);
end;
for i=1:c,
  ypred(:,i,1) = M*alpha{1,i};
end;
if Validation
  Mv=zeros(Nv,k(1)+d+1);
  Mv(:,1) = ones(Nv,1);
  Mv(:,2:d+1) = xv;
  for j=d+2:k(1)+d+1,
    Mv(:,j) = feval(bFunction,mu{1}(j-d-1,:),xv);
  end;
  for i=1:c,
    ypredv(:,i,1) = Mv*alpha{1,i};
  end;
end;

% INITIALISE COUNTERS:
% ===================
aUpdate=0;
rUpdate=0;
aBirth=0;
rBirth=0;
aDeath=0;
rDeath=0;
aMerge=0;
rMerge=0;
aSplit=0;
rSplit=0;
aRW=0;
rRW=0;
match=0;
if doPlot
  figure(3)
  clf;
end;

% ITERATE THE MARKOV CHAIN:
% ========================
for t=1:chainLength-1,
  iteration=t
  % COMPUTE THE CENTRES AND DIMENSION WITH METROPOLIS, BIRTH AND DEATH MOVES:
  % ========================================================================
  decision=rand(1);
  birth=arbC*min(1,(nabla(t)/(k(t)+1)));
  death=arbC*min(1,((k(t)+1)/nabla(t)));
  if ((decision <= birth) & (k(t)<kMax)),
    [k,mu,M,match,aBirth,rBirth] = radialBirth(match,aBirth,rBirth,k,mu,M,delta,x,y,hyper,t,bFunction,walkInt,walk);
  elseif ((decision <= birth+death) & (k(t)>0)),
    [k,mu,M,aDeath,rDeath] = radialDeath(aDeath,rDeath,k,mu,M,delta,x,y,hyper,t,nabla);
  elseif ((decision <= 2*birth+death) & (k(t)<kMax) & (k(t)>1)),
    [k,mu,M,aSplit,rSplit] = radialSplit(aSplit,rSplit,k,mu,M,delta,x,y,hyper,t,bFunction,sigStar,walkInt,walk);
  elseif ((decision <= 2*birth+2*death) & (k(t)>1)),
    [k,mu,M,aMerge,rMerge] = radialMerge(aMerge,rMerge,k,mu,M,delta,x,y,hyper,t,bFunction,sigStar,walkInt);
  else
    uLambda = rand(1);
    if ((uLambda>Lambda) & (k(t)>0))
      [k,mu,M,match,aRW,rRW] = radialRW(match,aRW,rRW,k,mu,M,delta,x,y,hyper,t,bFunction,sWalk,walk);
    else  
      [k,mu,M,match,aUpdate,rUpdate] = radialUpdate(match,aUpdate,rUpdate,k,mu,M,delta,x,y,hyper,t,bFunction,walkInt,walk);
    end;
  end;

  % UPDATE OTHER PARAMETERS WITH GIBBS:
  % ==================================
  H=zeros(k(t+1)+1+d,k(t+1)+1+d,c);
  F=zeros(k(t+1)+1+d,c);
  P=zeros(N,N,c);
  for i=1:c,
    H(:,:,i) = inv(M'*M + (1/delta(t,i))*eye(k(t+1)+1+d));
    F(:,i) = H(:,:,i)*M'*y(:,i);
    P(:,:,i) = eye(N) - M*H(:,:,i)*M';
    sigma(t+1,i) = inv(gengamma((hyper.v+N)/2,(hyper.gamma+y(:,i)'*P(:,:,i)*y(:,i))/2));
    alpha{t+1,i} = mvnrnd(F(:,i),sigma(t+1,i)*H(:,:,i),1)';
    delta(t+1,i) = inv(gengamma(hyper.a+(k(t+1)+d+1)/2,hyper.b+inv(2*sigma(t+1,i))*alpha{t+1,i}'*alpha{t+1,i}));
  end;
  nabla(t+1) = gengamma(0.5+hyper.e1+k(t+1),1+hyper.e2); 

  % COMPUTE THE POSTERIOR FOR MONITORING:
  % ==================================== 
  posterior  =exp(-nabla(t+1)) * delta(t+1,1)^(-(d+k(t+1)+1)/2) * inv(prod(1:k(t+1)) * prod(walkInt)^(k(t+1))) * nabla(t+1)^(k(t+1)) * sqrt(det(H(:,:,1))) * (hyper.gamma+y(:,1)'*P(:,:,1)*y(:,1))^(-(hyper.v+N)/2);
  for i=2:c,
    newpost = delta(t+1,i)^(-(d+k(t+1)+1)/2) * sqrt(det(H(:,:,i))) * (hyper.gamma+y(:,i)'*P(:,:,i)*y(:,i))^(-(hyper.v+N)/2);  
    posterior  = posterior * newpost;
  end;
  post(t+1) = log(posterior);

  % PLOT FOR FUN AND MONITORING:
  % ============================ 
  for i=1:c,
    ypred(:,i,t+1) = M*alpha{t+1,i};
  end;
  msError = inv(N) * trace((y-ypred(:,:,t+1))'*(y-ypred(:,:,t+1)));
%  NRMSE = sqrt((y-ypred(:,:,t+1))'*(y-ypred(:,:,t+1))*inv((y-mean(y)*ones(size(y)))'*(y-mean(y)*ones(size(y)))))

  if Validation,
    % FILL THE VALIDATION REGRESSION MATRIX: 
    % ======================================
    Mv=zeros(Nv,k(t+1)+d+1);
    Mv(:,1) = ones(Nv,1);
    Mv(:,2:d+1) = xv;
    for j=d+2:k(t+1)+d+1,
      Mv(:,j) = feval(bFunction,mu{t+1}(j-d-1,:),xv);
    end;
    for i=1:c,
      ypredv(:,i,t+1) = Mv*alpha{t+1,i};
    end;
    msErrorv = inv(Nv) * trace((yv-ypredv(:,:,t+1))'*(yv-ypredv(:,:,t+1)));
  end;

  if doPlot,
    figure(1)  
    clf
    if (c==2),
      plot(x(:,1),y(:,1),'b+',x(:,2),y(:,2),'r+',x(:,1),ypred(:,1,t+1),'bo',x(:,2),ypred(:,2,t+1),'ro');
    elseif c==1,
     plot(x,y,'b+',x,ypred(:,:,t+1),'ro');
    end;
    errorv = sum(abs(yv-ypredv(:,:,t+1)))*100*inv(Nv);
    ylabel('Output','fontsize',15)
    xlabel('Input','fontsize',15)
    figure(3)
    subplot(511);
    hold on;
    plot(t,k(t),'*');
    ylabel('k','fontsize',15);
    subplot(512);
    hold on;
    plot(t,post(t+1),'*');
    ylabel('p(k,mu|y)','fontsize',15);  
    subplot(513);
    hold on;
    plot(t,msError,'r*');
    ylabel('Train error','fontsize',15);
    subplot(514);
    hold on;
    plot(t,msErrorv,'r*');
    ylabel('Test error','fontsize',15);
    subplot(515);
    hold on;
    bar([1 2 3 4 5 6 7 8 9 10 11 12 13],[match aUpdate rUpdate aBirth rBirth aDeath rDeath aMerge rMerge aSplit rSplit aRW rRW]);
    ylabel('Acceptance','fontsize',15);
    xlabel('match aU rU aB rB aD rD aM rM aS rS aRW rRW','fontsize',15)
  end;
end;









