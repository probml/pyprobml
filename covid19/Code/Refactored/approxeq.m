function p = approxeq(a, b, tol, rel)
% Are a and b approximately equal (to within a specified tolerance)?
% p = approxeq(a, b, thresh)
% 'tol' defaults to 1e-3.
% p(i) = 1 iff abs(a(i) - b(i)) < thresh
%
% p = approxeq(a, b, thresh, 1)
% p(i) = 1 iff abs(a(i)-b(i))/abs(a(i)) < thresh

% This file is from pmtk3.googlecode.com


if nargin < 3, tol = 1e-2; end
if nargin < 4, rel = 0; end

a = a(:);
b = b(:);
if length(a) ~= length(b)
  p = false;
  return;
end
d = abs(a-b);
if rel
  p = ~any( (d ./ (abs(a)+eps)) > tol);
else
  p = ~any(d > tol);
end

end