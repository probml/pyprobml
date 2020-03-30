function [loss, loss_per_ex] = mc_objective(model, input_data, output_data, num_ens)

[paramin, paramax] = param_bounds();     
violations = sum(model.params < paramin) + sum(model.params > paramax);
penalty = violations * 1e5;
nexamples = sum(size(output_data));
if penalty > 0
    loss = penalty;
    loss_per_ex = penalty*ones(nexamples,1);
else
    [obs_pred_samples] = sample_data(model, input_data, num_ens); %X(l,e,t)
    [loss, loss_per_ex] = mae_objective(output_data,  obs_pred_samples);
end


end