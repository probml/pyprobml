function plot_time_series(obs_truth)

wuhan = 170;
beijing = 1;
chongqing = 255;

obs_truth_all = sum(obs_truth,1); % sum over all locations
truth_list = {obs_truth_all, obs_truth(wuhan,:), ...
    obs_truth(beijing,:), obs_truth(chongqing,:)};
city_name_list = {'all', 'wuhan', 'beijing', 'chonginq'};

figure;
for i=1:length(city_name_list)
    subplot(2,2,i)
    truth = truth_list{i};
    city_name = city_name_list{i};
    plot(truth, 'o-')
    title(sprintf('loc=%s', city_name))
end


end