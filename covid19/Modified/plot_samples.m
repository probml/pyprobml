
function plot_samples(model, data, num_ens, fig_folder)

obs_truth = data.obs_truth;
[num_loc, num_times] = size(obs_truth);
wuhan = 170;
obs_truth_wuhan = obs_truth(wuhan,:);
obs_truth_all = sum(obs_truth); % sum over all locations
max_count = max(obs_truth_all(:)); % max over time to set scale
  
[obs_samples] = sample_data(model, data, num_ens);
    
obs_samples_all =  squeeze(sum(obs_samples,1));
obs_samples_wuhan = squeeze(obs_samples(wuhan,:,:));

truth_list = {obs_truth_all, obs_truth_wuhan};
samples_list = {obs_samples_all, obs_samples_wuhan};
city_name_list = {'all', 'wuhan'};

%truth_list = {obs_truth_wuhan};
%samples_list = {obs_samples_wuhan};
%city_name_list = {'wuhan'};

for i=1:length(city_name_list)
    truth = truth_list{i};
    samples = samples_list{i};
    city_name = city_name_list{i};
    figure;
    plot(truth, 'kx', 'markersize', 10)
    hold on
    boxplot(samples)
    ylabel('reported cases')
    xlim([0 num_times+1]);
    ylim([-10 max_count+10])
    
    [mse, mae, nll] =  evaluate_preds(truth, samples);
    title(sprintf('%s, loc=%s, mse=%5.3f, mae=%5.3f, nll=%5.3f',...
        model.name, city_name, mse, mae, nll))
    fname = sprintf('%s/predictions-%s-%s', fig_folder, city_name, model.name);
    print(fname, '-dpng');
    
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


