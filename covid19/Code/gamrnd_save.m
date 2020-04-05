rng(42);
Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers
save('../Data/gamrnd_seed42', 'gam_rnds');
csvwrite('../Data/gamrnd_seed42.csv', gam_rnds);