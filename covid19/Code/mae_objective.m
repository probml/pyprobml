function [loss_total, loss_per_example]  = mae_objective(obs_truth,  obs_pred_samples)
 %truth(l,t)
 %samples(l,e,t)

obs_pred_median = squeeze(median(obs_pred_samples, 2)); 
loss_per_example = abs(obs_truth - obs_pred_median);
loss_per_example = loss_per_example(:);
loss_total = mean(loss_per_example);

end