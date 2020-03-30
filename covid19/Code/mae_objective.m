function loss = mae_objective(obs_truth,  obs_pred_samples)
 %truth(l,t)
 %samples(l,e,t)

obs_pred_median = squeeze(median(obs_pred_samples, 2)); 
err = abs(obs_truth - obs_pred_median);
loss = mean(err(:));


end