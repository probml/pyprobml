load('../Data/M.mat') %M(l,l,t)
load('../Data/pop.mat') % pop(l)
load('../Data/incidence.mat') % O(t,l)

G = sum(M,3); % connectivity structure of graph

fig_folder = '~/covid19/Figures';
%https://www.mathworks.com/help/matlab/ref/colormap.html

[num_times, num_loc] =size(incidence);
obs_truth=incidence'; % obs(l,t)
wuhan=170;

figure;
bar(pop)
title('population size')
fname = sprintf('%s/pop', fig_folder);
print(fname, '-dpng');

figure;
imagesc(obs_truth)
colorbar;
xlabel('time (1/10/20-1/23/20)')
ylabel('locations')
title('reported new infections')
fname = sprintf('%s/incidence', fig_folder);
print(fname, '-dpng');


%{
nrows = 7; ncols = 2;
figure;
for t=1:num_times
    subplot(nrows, ncols, t)
    imagesc(M(:,:,t))
    title(sprintf('mobility day %d', t));
end
%}
figure;
%cmap=copper(256);
cmap=gray(256);
montage(M,cmap);
fname = sprintf('%s/mobility', fig_folder);
print(fname, '-dpng');

figure;
imagesc(G)
colorbar;
title('connectivity')
fname = sprintf('%s/connectivity', fig_folder);
print(fname, '-dpng');

figure;
imagesc(G>0);
colormap(gray)
title('connectivity (binarized)')
fname = sprintf('%s/connectivity_binary', fig_folder);
print(fname, '-dpng');

