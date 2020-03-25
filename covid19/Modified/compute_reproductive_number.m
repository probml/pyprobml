function R = compute_reproductive_number(params)
% Re = basic reproductive number at beginning of epiddemic
% param_samples(p, 1:nsamples) for p=1:6
% reuturns R(1:nsamples)
 
[beta, mu, theta, Z, alpha, D] = unpack_params(params);
nsamples = length(beta);
R = zeros(1, nsamples);
R = alpha .* beta .* D + (1-alpha) .* mu .* beta .* D;

end
