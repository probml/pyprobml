function x_post = process_trajectory(x, M, pop, obs_truth, OEV, lambda)


[num_var, num_ens] = size(x);
[num_loc, num_times] = size(obs_truth);

pop0=pop*ones(1,num_ens);

%observation operator: obs=Hx
H=zeros(num_loc,5*num_loc+6);
for i=1:num_loc
    H(i,(i-1)*5+5)=1;
end

pop=pop0;
obs_temp=zeros(num_loc,num_ens,num_times);%records of reported cases
x_post=zeros(num_var,num_ens,num_times);%posterior
for t=1:num_times
     fprintf('timestep %d\n', t)  
    %inflation
    x=mean(x,2)*ones(1,num_ens)+lambda*(x-mean(x,2)*ones(1,num_ens));
    x=checkbound(x,pop);
    %integrate forward
    [x,pop]=SEIR_refactored(x,M,pop,t,pop0);
    obs_cnt=H*x;%new infection

    obs_temp = add_delayed_obs(obs_temp, t, obs_cnt);
    obs_ens=obs_temp(:,:,t);%observation at t
    
    %obs_ens = obs_cnt; % predicted observed counts

    %loop through local observations
    for l=1:num_loc
        %Get the variance of the ensemble
        obs_var = OEV(l,t);
        prior_var = var(obs_ens(l,:));
        post_var = prior_var*obs_var/(prior_var+obs_var);
        if prior_var==0%if degenerate
            post_var=1e-3;
            prior_var=1e-3;
        end
        prior_mean = mean(obs_ens(l,:));
        post_mean = post_var*(prior_mean/prior_var + obs_truth(l,t)/obs_var);
        %%%% Compute alpha and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);
        %Loop over each state variable (connected to location l)
        rr=zeros(1,num_var);
        neighbors=union(find(sum(M(:,l,:),3)>0),find(sum(M(l,:,:),3)>0));
        neighbors=[neighbors;l];%add location l
        for i=1:length(neighbors)
            idx=neighbors(i);
            for j=1:5
                A=cov(x((idx-1)*5+j,:),obs_ens(l,:));
                rr((idx-1)*5+j)=A(2,1)/prior_var;
            end
        end
        for i=num_loc*5+1:num_loc*5+6
            A=cov(x(i,:),obs_ens(l,:));
            rr(i)=A(2,1)/prior_var;
        end
        %Get the adjusted variable
        dx=rr'*dy;
        x=x+dx;
        %Corrections to DA produced aphysicalities
        x = checkbound(x,pop);
    end
    x_post(:,:,t)=x;
end

end


