function s=lhsu(xmin,xmax,nsample)
% latin hypercube sampling utility
% xmin(i) min value for site i=1:nvar
% xmax(i)
% s(nsamples, nvar)

nvar=length(xmin);
%ran=rand(nsample,nvar);
s=zeros(nsample,nvar);
for j=1:nvar
   %idx=randperm(nsample);
   %P =(idx'-ran(:,j))/nsample;
   P = linspace(0, 1, nsample); % HACK KPM to make deterministic
   s(:,j) = xmin(j) + P.* (xmax(j)-xmin(j));
   %s(:,j) = xmin(j) + 0.5.* (xmax(j)-xmin(j)); 
end

end