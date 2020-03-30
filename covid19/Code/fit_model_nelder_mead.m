function [model, loss] = fit_model_nelder_mead(...
    model, input_data, output_data, num_ens, max_iter)


function loss = objective(params)
    model.params = params;
  loss = mc_objective(model, input_data, output_data, num_ens);
end

options = optimset('Display','iter','MaxIter',max_iter);
[params, loss] = fminsearch(@objective, model.params, options);
model.params = params;
 


end

