function [obs, pred] = exclude_top_k(obs_truth, pred_samples, k)
% remove the top k cities with highest counts
% obs_truth(l,t)
% pred_samples(l,e,t) 

[Osort, ndx] = sort(obs_truth, 'descend');
% ndx(1:k) are the top k cities with highest counts
% for china data, the top 3 are 
% id=170 name=wuhan count=454
% id=255 nane=chongqing count=27
% id=1 name=beijing count=26
% There are 116 with >=1 counts
keep = ndx(k+1:end);
obs = obs_truth(keep,:);
pred = pred_samples(keep,:,:);
        
end