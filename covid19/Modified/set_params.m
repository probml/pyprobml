function params=set_params(ndx)

if nargin < 1, ndx = 1; end

if ndx == 1
    % values from Table 1 of Science paper
    beta = 1.12;
    mu = 0.55;
    Z = 3.69;
    D = 3.48;
    alpha = 0.14;
    theta = 1.36;
end

if ndx==2
    beta=1.151;
    mu=0.564;
    theta=1.268;
    Z=3.66;
    alpha=0.257;
    D=3.407;
end

params = [beta,mu,theta,Z,alpha,D];
params = params(:);

end
