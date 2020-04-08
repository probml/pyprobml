results_folder = '~/covid19/Results/';
    
load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)


% results from exeucting inference_run.m 
num_ens = 300;
num_iter = 10;
seed = 42;
fname = sprintf('leg-results-E%d-I%d-S%d', num_ens, num_iter, seed);
results = load(sprintf('%s/%s.mat', results_folder, fname)); 
ppost = results.ppost;
theta = results.theta;
zpost = results.zpost; % nloc*5 x nens x ntimes x niter
ttl = safeStr(fname);


fig_folder = '~/covid19/Figures';

[nstates, nens, ntimes, niter] = size(zpost);
nloc = nstates/5;
S = zeros(nloc, nens, ntimes);
E = zeros(nloc, nens, ntimes);
IR = zeros(nloc, nens, ntimes);
IU = zeros(nloc, nens, ntimes);
O = zeros(nloc, nens, ntimes);
for t=1:ntimes
    [S(:,:,t), E(:,:,t), IR(:,:,t), IU(:,:,t), O(:,:,t)]=unpack_states(zpost(:,:,t,end));
end

Smean = squeeze(mean(S,2));
figure; imagesc(Smean./pop); title('susceptible/pop'); colorbar
fname = sprintf('%s/Snormalized', fig_folder); print(fname, '-dpng');

Emean = squeeze(mean(E,2));
figure; imagesc(Emean./pop); title('exposed/pop'); colorbar
fname = sprintf('%s/Enormalized', fig_folder); print(fname, '-dpng');

IRmean = squeeze(mean(IR,2));
figure; imagesc(IRmean./pop); title('InfectedReported/pop'); colorbar
fname = sprintf('%s/IRnormalized', fig_folder); print(fname, '-dpng');

IRmean = squeeze(mean(IR,2));
figure; imagesc(IRmean); title('InfectedReported'); colorbar
fname = sprintf('%s/IR', fig_folder); print(fname, '-dpng');

IUmean = squeeze(mean(IU,2));
figure; imagesc(IUmean./pop); title('InfectedUnreported/pop'); colorbar
fname = sprintf('%s/IUnormalized', fig_folder); print(fname, '-dpng');

Omean = squeeze(mean(O,2));
figure; imagesc(Omean); title('predicted new exposures'); colorbar
fname = sprintf('%s/O', fig_folder); print(fname, '-dpng');

% Trace plots for chosen cities
beijing = 1; wuhan=170;
cities = [beijing, wuhan];
city_names = {'beijing', 'wuhan'};
for i=1:length(city_names)
    id = cities(i);
    name = city_names{i};
    nrows = 3; ncols = 2;
    figure;
    subplot(nrows, ncols, 1)
    plot(Smean(id,:)); ylabel('S')
    subplot(nrows, ncols, 2)
    plot(Emean(id,:)); ylabel('E')
    subplot(nrows, ncols, 3)
    plot(IRmean(id,:)); ylabel('IR')
    subplot(nrows, ncols, 4)
    plot(IUmean(id,:)); ylabel('IU')
    
    subplot(nrows, ncols, 5)
    plot(Omean(id,:)); ylabel('Opred')
    subplot(nrows, ncols, 6)
    plot(obs_truth(id,:)); ylabel('Otrue')
   
    
    suptitle(sprintf('%s', name));
    fname = sprintf('%s/%s-traj', fig_folder, name); print(fname, '-dpng');
end




