
function obs_seq = add_delayed_obs(obs_seq, t, obs_now, gam_rnds, rnd_delay)
%pred_obs_seq(l,e,t) 
% pred_obs_now(l,e)
[num_loc, num_ens, num_times] = size(obs_seq);

 
Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
%gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generage gamma random numbers

if nargin < 5
    rnd_delay = true;
else
    % deterministically recyrcles delats 0:10
    ndx = 0:10;
    delays = repmat(ndx, 1, 100);
end
for k=1:num_ens
    for l=1:num_loc
        N = obs_now(l,k);
        if N>0
            %gam_rnds = ceil(gamrnd(a,b,1e4,1));
            if rnd_delay
              rnd = datasample(gam_rnds,N); 
            else
              rnd = delays(1:N);
            end
            % sample N random delays, and insert current observations later
            for h=1:N
                if (t+rnd(h)<=num_times)
                    obs_seq(l,k,t+rnd(h))=obs_seq(l,k,t+rnd(h))+1;
                end
            end
        end
    end
end

end