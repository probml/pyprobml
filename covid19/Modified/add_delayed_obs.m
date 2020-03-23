
function obs_seq = add_delayed_obs(obs_seq, t, obs_now)
%pred_obs_seq(l,e,t) 
% pred_obs_now(l,e)
[num_loc, num_ens, num_times] = size(obs_seq);

 
Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
%gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generage gamma random numbers

for k=1:num_ens
    for l=1:num_loc
        N = obs_now(l,k);
        if N>0
            gam_rnds = ceil(gamrnd(a,b,1e4,1));
            rnd=datasample(gam_rnds,N);
            % sample N random delays, and insert current observations later
            for h=1:length(rnd)
                if (t+rnd(h)<=num_times)
                    obs_seq(l,k,t+rnd(h))=obs_seq(l,k,t+rnd(h))+1;
                end
            end
        end
    end
end

end