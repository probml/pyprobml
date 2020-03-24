function subplot2(nrows, ncols, i, j)
% function subplot2(nrows, ncols, i, j)

% This file is from pmtk3.googlecode.com


sz = [nrows ncols];
k = sub2ind(sz(end:-1:1), j, i);
subplot(nrows, ncols, k);

end