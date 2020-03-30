function loss = mc_objective(model, data,  num_ens)

[paramin, paramax] = param_bounds();     
violations = sum(model.params < paramin) + sum(model.params > paramax);
penalty = violations * 1e5;
if penalty > 0
    loss = penalty;
else
    [obs_pred_samples] = sample_data(model, data, num_ens); %X(l,e,t)
    loss = mae_objective(data.obs_truth,  obs_pred_samples);
end


end