function [loss, loss_per_ex] = mc_objective(model, input_data, obs_truth, loc_ndx)
[num_loc, num_times] = size(obs_truth);
if nargin < 3, loc_ndx = 1:num_loc; end

[paramin, paramax] = param_bounds();     
violations = sum(model.params < paramin) + sum(model.params > paramax);
penalty = violations * 1e5;
nexamples = sum(size(obs_truth));
if penalty > 0
    loss = penalty;
    loss_per_ex = penalty*ones(nexamples,1);
else
    [obs_pred_samples] = sample_data(model, input_data); %X(l,e,t)
    [loss, loss_per_ex] = mae_objective(obs_truth,  obs_pred_samples, loc_ndx);
end


end