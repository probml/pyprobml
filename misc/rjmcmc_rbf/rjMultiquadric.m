function [y] = rjMultiquadric(mu,x);
% PURPOSE : Multiquadric basis function.
% INPUTS  : - mu: The basis centre.
%           - x:  The evaluation point in the domain.
% OUTPUTS : - y: The value of the Multiquadric at x.
% AUTHOR  : Nando de Freitas - Thanks for the acknowledgement :-)
% DATE    : 21-01-99

if nargin < 2, error('Not enough input arguments.'); end

[N,d] = size(x);      % N = number of data, d = dimension of x.
y=zeros(N,1);
for j=1:N,
  z=norm(x(j,:)-mu(1,:));             % Euclidean distance.
  y(j,1)= sqrt(z.^(2) -(0.5)^(2));    % Multiquadric>
end;








