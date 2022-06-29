function landscape()
% Script for drawing probability landscapes

%fig_dir = '/home/gpapan/Documents/myPubs/pmChapter/papandreou/figures/approx';
fig_dir = '/Users/kpmurphy/github/bookv2/figures';
%print_fun = @(fname) print('-depsc2',fullfile(fig_dir,sprintf('%s.eps',fname)));
print_fun = @(fname) print('-dpdf',fullfile(fig_dir,sprintf('%s.pdf',fname)));
lw = 4;

prior = [0.5 0.5];
[x,y] = meshgrid(0:0.01:6,0:0.01:6);
gauss_fun = @(mu,sigma) gauss2d(x,y,mu,sigma);

mu1 = [2 2]'; sigma1 = rotgauss(1,0.5,45*pi/180);   info1 = inv(sigma1);
mu2 = [4 4]'; sigma2 = rotgauss(1,0.5,135*pi/180);  info2 = inv(sigma1);
mu3 = [3 3]'; sigma3 = rotgauss(1.5,0.75,45*pi/180);

z1 = gauss_fun(mu1,sigma1);
z2 = gauss_fun(mu2,sigma2);
z3 = gauss_fun(mu3,sigma3);
z = prior(1)*z1+prior(2)*z2;
v = [0.02 0.045 0.08 0.14];

%% original
figure(1), contour(x,y,z,v,'b--','LineWidth',lw)
axis off, axis equal, axis tight
set(gcf, 'Renderer', 'opengl')
print_fun('posterior2d_orig');

%% MAP
figure(2), clf
contour(x,y,z,v,'b--','LineWidth',lw)
hold on
plot(mu1(1),mu1(2),'ro','MarkerSize',20,'MarkerFaceColor','Red')
hold off
axis off, axis equal, axis tight
set(gcf, 'Renderer', 'opengl')
print_fun('posterior2d_map');

%% Variational
figure(3), clf
contour(x,y,z,v,'b--','LineWidth',lw)
hold on
contour(x,y,z3,v,'r-','LineWidth',lw)
hold off
axis off, axis equal, axis tight
set(gcf, 'Renderer', 'opengl')
print_fun('posterior2d_var');

%% MCMC
randn('seed',1)
N = 12;
xn = zeros(N,1); yn = zeros(N,1);
% Gibbs sampling on 1st distro
xn(1) = 1.5; yn(1) = 3;
gain = 0.7;
for i=2:N
  xn(i) = (mu1(1)-info1(1,2)/info1(1,1)*(yn(i-1)-mu1(2))) + gain*sqrt(1/info1(1,1))*randn(1);
  yn(i) = (mu1(2)-info1(2,1)/info1(2,2)*(xn(i)  -mu1(1))) + gain*sqrt(1/info1(2,2))*randn(1);
end
figure(4), clf
contour(x,y,z,v,'b--','LineWidth',lw)
hold on
line(xn,yn,'Color','Red','LineWidth',lw,'Marker','*','MarkerSize',12)
hold off
axis off, axis equal, axis tight
set(gcf, 'Renderer', 'opengl')
print_fun('posterior2d_mcmc');

%% Perturb-and-MAP
randn('seed',10)
N1 = 5;
N2 = 4;
gain = 0.7;
xyn = horzcat(gauss2d_sample(mu1,gain*sigma1,N1), gauss2d_sample(mu2,gain*sigma2,N2));
% Gibbs sampling on 1st distro
figure(5), clf
contour(x,y,z,v,'b--','LineWidth',lw)
hold on
plot(xyn(1,:),xyn(2,:),'ro','MarkerSize',15,'MarkerFaceColor','Red')
hold off
axis off, axis equal, axis tight
set(gcf, 'Renderer', 'opengl')
print_fun('posterior2d_samples');


function z = gauss2d(x,y,mu,sigma)
xy = bsxfun(@minus,vertcat(x(:)',y(:)'),mu); % [2 N]
d2 = sum(xy.*(sigma\xy),1);
z = 1/sqrt(det(2*pi*sigma))*exp(-0.5*d2);
z = reshape(z,size(x));

function x = gauss2d_sample(mu,sigma,N)
L = chol(sigma,'lower');
x = bsxfun(@plus,mu,L*randn(2,N));

function sigma = rotgauss(r1, r2, theta)
e1 = complex2vec(exp(1i*theta));
e2 = complex2vec(exp(1i*(theta+pi/2)));
sigma = r1^2*(e1*e1') + r2^2*(e2*e2');

function v = complex2vec(c)
v = [real(c) imag(c)]';
