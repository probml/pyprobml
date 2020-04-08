
function plot_samples(obs_samples, obs_truth)


[num_loc, num_ens, num_times] = size(obs_samples);
wuhan = 170;
beijing = 1;
chongqing = 255;

obs_truth_all = sum(obs_truth,1); % sum over all locations
max_count = max(obs_truth_all(:)); % max over time to set scale
obs_samples_all =  squeeze(sum(obs_samples,1));

truth_list = {obs_truth_all, obs_truth(wuhan,:), ...
    obs_truth(beijing,:), obs_truth(chongqing,:)};
samples_list = {obs_samples_all, ...
    squeeze(obs_samples(wuhan,:,:)), ...
    squeeze(obs_samples(beijing,:,:)), ...
    squeeze(obs_samples(chongqing,:,:))};
city_name_list = {'all', 'wuhan', 'beijing', 'chonginq'};

figure;
for i=1:length(city_name_list)
    subplot(2,2,i)
    truth = truth_list{i};
    samples = samples_list{i};
    city_name = city_name_list{i};
    plot(truth, 'kx', 'markersize', 10)
    hold on
    boxplot(samples)
    ylabel('reported cases')
    xlim([0 num_times+1]);
    ylim([-10 max_count+10])
    [mse, mae, nll] =  evaluate_preds(truth, samples);
    title(sprintf('loc=%s, mse=%5.3f, mae=%5.3f, nll=%5.3f',...
        city_name, mse, mae, nll))
end


end

function [mse, mae, nll] =  evaluate_preds(truth, samples)
ntimes = length(truth);
nll = 0;
mse = 0;
mae = 0;
for t=1:ntimes
    dist = fitdist(samples(:,t), 'kernel');
    nll = nll + -1*log(pdf(dist, truth(t)));
    mae = mae + abs(median(dist) - truth(t));
    mse = mse + (mean(dist) - truth(t))^2;
end
mse = mse/ntimes;
mae = mae/ntimes;
end


