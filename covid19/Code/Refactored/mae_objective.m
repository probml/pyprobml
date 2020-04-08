function [loss_total, loss_per_example]  = mae_objective(obs_truth,  obs_pred_samples, loc_ndx)
 %truth(l,t)
 %samples(l,e,t)
 
[num_loc, num_times] = size(obs_truth);
if nargin < 3, loc_ndx = 1:num_loc; end

obs_pred_median = squeeze(median(obs_pred_samples, 2)); 
loss_per_example = zeros(num_loc, num_times);
loss_per_example(loc_ndx,:) = abs(obs_truth(loc_ndx,:) - obs_pred_median(loc_ndx,:));
loss_total = mean(loss_per_example(:));

end