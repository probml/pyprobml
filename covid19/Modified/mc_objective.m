function loss = mc_objective(model, data,  num_ens)

[paramin, paramax] = param_bounds();     
violations = sum(model.params < paramin) + sum(model.params > paramax);
penalty = violations * 1e5;
if penalty > 0
    loss = penalty;
else
    [obs_pred_samples] = sample_data(model, data, num_ens); %X(l,e,t)
    obs_pred_median = squeeze(median(obs_pred_samples, 2)); %X(l,t)
    err = abs(data.obs_truth - obs_pred_median);
    loss = mean(err(:));
end


end