function plot_params()

results_folder = '~/covid19/Results/';
    
%{
    % results from executing the following: 
    % rng(42); inference(); 
    results = load(sprintf('%s/inference.mat', results_folder));
    ppost = results.para_post;
    theta = results.theta;
    ttl = 'original-E300-I10-S42';
%}

% results from exeucting inference_run.m 
num_ens = 300;
num_iter = 10;
seed = 42;
fname = sprintf('leg-results-E%d-I%d-S%d', num_ens, num_iter, seed);
results = load(sprintf('%s/%s.mat', results_folder, fname)); 
ppost = results.ppost;
theta = results.theta;
%zpost = results.zpost; % nloc*5 x nens x ntimes x niter
ttl = safeStr(fname);


fig_folder = '~/covid19/Figures';

param_names = {'\beta', '\mu', '\theta', 'Z', '\alpha', 'D'};

[nparams, nsamples, ntimes, niter] = size(ppost);
%samples = ppost(:,:,1,end);
samples = reshape(ppost(:,:,:,:), [nparams, nsamples*ntimes*niter]);
plot_R_hist(samples);
suptitle(sprintf('R, %s', ttl));
fname = sprintf('%s/%s-R-hist', fig_folder, ttl);
print(fname, '-dpng');

params = ppost(:,:,1,end); % first time step of final iteration
plot_param_hist(params, param_names);
suptitle(sprintf('param-hist at start of final iter, %s', ttl));
fname = sprintf('%s/%s-params-hist-final', fig_folder, ttl);
print(fname, '-dpng');

params = ppost(:,:,1,end); % first time step of final iteration
plot_param_boxplot(params, param_names);
suptitle(sprintf('param-box at start of final iter, %s', ttl));
fname = sprintf('%s/%s-params-box-final', fig_folder, ttl);
print(fname, '-dpng');

params = ppost; % all the samples
plot_param_boxplot_over_iter(params, param_names);
suptitle(sprintf('param-box vs iter, %s', ttl));
fname = sprintf('%s/%s-params-box-iter', fig_folder, ttl);
print(fname, '-dpng');

params = theta;
plot_param_mean_over_iter(params, param_names);
suptitle(sprintf('param-mean vs iter, %s', ttl));
fname = sprintf('%s/%s-params-mean-iter', fig_folder, ttl);
print(fname, '-dpng');


end

function plot_R_hist(samples)
% samples(p,s)
R = compute_reproductive_number(samples);
figure;
nbins = 10;
histogram(R, nbins, 'Normalization','probability');
m = mean(R);
h=xline(m, '-r');
set(h, 'linewidth', 3);
title(sprintf('mean %5.3f', m))
xlabel('R_e') 
end

function plot_param_hist(params, param_names)
% params(p,:)
[paramin, paramax] = param_bounds();
nrows = 2; ncols = 3;
nparams = length(param_names);
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    nbins = 10;
    samples = params(p,:);
    h = histogram(samples, nbins, 'Normalization','probability');
    m = mean(samples);
    h=xline(m, '-r');
    set(h, 'linewidth', 3);
    title(sprintf('mean %5.3f', m))
    xlim([paramin(p), paramax(p)])
    ylabel(param_names{p}) 
    xlabel('prior range')
end
end

function plot_param_boxplot(params, param_names)
nrows = 2; ncols = 3;
nparams = length(param_names);
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    boxplot(params(p,:)) 
    ylabel(param_names{p})
end
end

function plot_param_boxplot_over_iter(params, param_names)
nrows = 6; ncols = 1;
nparams = length(param_names);
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    samples = squeeze(params(p,:,1,:));
    boxplot(samples) 
    ylabel(param_names{p})
end
end

function plot_param_mean_over_iter(theta, param_names)
nrows = 2; ncols = 3;
nparams = length(param_names);
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    plot(theta(p,:), 'o-');
    ylabel(param_names{p})
    final_val = theta(p, end);
    title(sprintf('%5.3f', final_val))
end
end

