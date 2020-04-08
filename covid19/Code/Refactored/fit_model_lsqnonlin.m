function [model, loss] = fit_model_lsqnonlin(...
    model, input_data, output_data, num_ens, max_iter)


function loss_per_example = objective(params)
    model.params = params;
  [loss_total, loss_per_example] = mc_objective(model, input_data, output_data, num_ens);
end

options = optimset('Display','iter','MaxIter',max_iter);
[paramin, paramax] = param_bounds();
[params, resnorm, residual] = lsqnonlin(@objective, model.params, paramin, paramax, options);
model.params = params;
loss = mean(residual);

% resnorm is sum of L2 errors.
% Here we compute mean of L1 errors, to be compatible with other solvers 
%[loss, loss_per_example] = mc_objective(model, input_data, output_data, num_ens);

end