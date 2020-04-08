function [z0]=initialize_state_deterministic(pop0, M)

num_loc = 375;
S0 = pop0;
E0 = zeros(num_loc,1);
IR0 = zeros(num_loc,1);
IU0 = zeros(num_loc,1);
O0 = zeros(num_loc,1);

seed_city = 170; % wuhan
num_init_infected = 10000;
E0(seed_city) = num_init_infected;
IU0(seed_city) = num_init_infected;

G = M(:,seed_city,1); 
for i=1:num_loc
    if i ~= seed_city
        E0(i) = G(i) * E0(seed_city)/pop0(seed_city);
        IU0(i) = G(i) * IU0(seed_city)/pop0(seed_city);
    end
end
z0 = pack_states(S0,E0,IR0,IU0,O0);
z0 = round(z0);

end