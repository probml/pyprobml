function [model, loss] = optimize_model(model, data, num_ens, max_iter)


function loss = objective(params)
    model.params = params;
  loss = mc_objective(model, data, num_ens);
end

options = optimset('Display','iter','MaxIter',max_iter);
[params0]=model.params;
[params, loss] = fminsearch(@objective, params0, options);
model.params = params;
 
%{
for param_ndx = 1:2
    params = set_params(param_ndx);
    loss = objective(params);
    fprintf('param ndx %d loss %5.3f\n', param_ndx, loss);
end
%}

end

