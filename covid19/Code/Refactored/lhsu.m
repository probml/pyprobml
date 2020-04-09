function s=lhsu(xmin,xmax,nsample, random)
% latin hypercube sampling utility
% xmin(i) min value for site i=1:nvar
% xmax(i)
% s(nsamples, nvar)

if nargin < 4, random = true; end

if random
    % original code
    nvar=length(xmin);
    ran=rand(nsample,nvar);
    s=zeros(nsample,nvar);
    for j=1:nvar
       idx=randperm(nsample);
       P =(idx'-ran(:,j))/nsample;
       s(:,j) = xmin(j) + P.* (xmax(j)-xmin(j));
    end
else
    % determinsitic version for debugging
    nvar=length(xmin);
    s=zeros(nsample,nvar);
    for j=1:nvar
       P = linspace(0, 1, nsample); 
       s(:,j) = xmin(j) + P.* (xmax(j)-xmin(j));
    end
end

end