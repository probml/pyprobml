function plot_nonzero_states(states, t, thresh)
if nargin < 3, thresh = 0; end

[S,E,IR,IU,O] = unpack_states(states);
figure;

subplot(2,2,1);  ndx = find(S>thresh);
plot(S(ndx)); title(sprintf('S t=%d nnz=%d', t, length(ndx)));

subplot(2,2,2);  ndx = find(E>thresh);
plot(E(ndx)); title(sprintf('E t=%d nnz=%d', t, length(ndx)));

subplot(2,2,3);  ndx = find(IR>thresh);
plot(IR(ndx)); title(sprintf('IR t=%d nnz=%d', t, length(ndx)));

subplot(2,2,4);  ndx = find(IU>thresh);
plot(IU(ndx)); title(sprintf('IU t=%d nnz=%d', t, length(ndx)));
end
