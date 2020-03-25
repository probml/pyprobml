

if false
    % results from executing the following: 
    % rng(42); inference(); 
    data_dir = '~/covid19/Results/';
    results = load('%s/inference.mat', data_dir);
    ppost = results.para_post;
    theta = results.theta;
    ttl = 'original-E300-I10-S42';
else
    % results from inference_run.m f
    num_ens = 300;
    num_iter = 10;
    seed = 42;
    legacy = false;
    data_dir = '~/covid19/Results/';
    if legacy
         fname = sprintf('leg-results-E%d-I%d-S%d', num_ens, num_iter, seed);
    else
        fname = sprintf('leg-results-E%d-I%d-S%d', num_ens, num_iter, seed);
    end
    results = load(sprintf('%s/%s.mat', data_dir, fname)); % zpost, ppost
    ppost = results.ppost;
    ttl = safeStr(fname);
end
figfolder = '~/covid19/Figures/';

param_names = {'\beta', '\mu', '\theta', 'Z', '\alpha', 'D'};
nparams = length(param_names);
nbins  = 10;

% fig S8: Histogram of R
figure;
[nparams, nsamples, ntimes, niter] = size(ppost);
%samples = ppost(:,:,1,end);
samples = reshape(ppost(:,:,:,:), [nparams, nsamples*ntimes*niter]);
R = compute_reproductive_number(samples);
histogram(R, nbins, 'Normalization','probability');
m = mean(R);
h=xline(m, '-r');
set(h, 'linewidth', 3);
title(sprintf('mean %5.3f', m))
xlabel('R_e') 
fname = sprintf('%s%s-R-hist', figfolder, ttl);
print(fname, '-dpng')

% Figure S4: histogram of params
[paramin, paramax] = param_bounds();
nrows = 2; ncols = 3;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    nbins = 10;
    samples = squeeze(ppost(p,:,:,:));
    samples = samples(:);
    h = histogram(samples, nbins, 'Normalization','probability');
    m = mean(samples);
    h=xline(m, '-r');
    set(h, 'linewidth', 3);
    title(sprintf('mean %5.3f', m))
    xlim([paramin(p), paramax(p)])
    ylabel(param_names{p}) 
    xlabel('prior range')
end
suptitle(sprintf('param-hist over all samples, %s', ttl));
fname = sprintf('%s%s-params-hist-overall', figfolder, ttl);
print(fname, '-dpng');

% Figure S4: histogram of params
[paramin, paramax] = param_bounds();
nrows = 2; ncols = 3;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    nbins = 10;
    histogram(ppost(p,:,1,end), nbins, 'Normalization','probability')
    xlim([paramin(p), paramax(p)])
    ylabel(param_names{p}) 
    xlabel('prior range')
end
suptitle(sprintf('param-hist at start of final iter, %s', ttl));
fname = sprintf('%s%s-params-hist-time1-iterI', figfolder, ttl);
print(fname, '-dpng');

% Figure S4: histogram of params
[paramin, paramax] = param_bounds();
nrows = 2; ncols = 3;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    nbins = 10;
    histogram(ppost(p,:,end,end), nbins, 'Normalization','probability')
    xlim([paramin(p), paramax(p)])
    ylabel(param_names{p}) 
    xlabel('prior range')
end
suptitle(sprintf('param-hist at end of final iter, %s', ttl));
fname = sprintf('%s%s-params-hist-timeT-iterI',  figfolder, ttl);
print(fname, '-dpng');

% Figure S18: boxplot of params
nrows = 2; ncols = 3;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    boxplot(ppost(p,:,1,end)) % final set of samples
    ylabel(param_names{p})
end
suptitle(sprintf('param-box at start of final iter, %s', ttl));
fname = sprintf('%s%s-params-box-time1-iterI',  figfolder, ttl);
print(fname, '-dpng');

% Figure S18: boxplot of params
nrows = 2; ncols = 3;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    boxplot(ppost(p,:,end,end)) % final set of samples
    ylabel(param_names{p})
end
suptitle(sprintf('param-box at end of final iter, %s', ttl));
fname = sprintf('%s%s-params-box-timeT-iterI',  figfolder, ttl);
print(fname, '-dpng');


% Figure S6: boxplot of params over iterations
nrows = 6; ncols = 1;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    samples = squeeze(ppost(p,:,1,:));
    boxplot(samples) 
    ylabel(param_names{p})
end
suptitle(sprintf('param-box at start of each iter, %s', ttl));
fname = sprintf('%s%s-params-box-time1-iters',  figfolder, ttl);
print(fname, '-dpng');

% Figure S6: boxplot of params over iterations
nrows = 6; ncols = 1;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    samples = squeeze(ppost(p,:,end,:));
    boxplot(samples) 
    ylabel(param_names{p})
end
suptitle(sprintf('param-box at end of each iter, %s', ttl));
fname = sprintf('%s%s-params-box-timeT-iters',  figfolder, ttl);
print(fname, '-dpng');

% Figure S6: boxplot of param-avg over iterations
nrows = 2; ncols = 3;
figure;
for p=1:nparams
    subplot(nrows, ncols, p);
    plot(theta(p,:), 'o-');
    ylabel(param_names{p})
    final_val = theta(p, end);
    title(sprintf('%5.3f', final_val))
end
suptitle(sprintf('param-mean at end of each iter, %s', ttl));
fname = sprintf('%s%s-param-mean-iters',  figfolder, ttl);
print(fname, '-dpng');



