function x = gengamma(alpha, beta)
% USED TO SAMPLE FROM A GAMMA DISTRIBUTION.

% AUTHOR: ARNAUD DOUCET.

% si alpha=1, on a une exponentielle beta
if (alpha==1)
   x = -log(1-rand(1,1))/beta;
   return
end
flag=0;       % teste si alpha<1 ou alpha>1
if (alpha<1)
   flag=1;
   alpha=alpha+1;
end
gamma=alpha-1;
eta=sqrt(2.0*alpha-1.0);
c=.5-atan(gamma/eta)/pi;
aux=-.5;
while(aux<0)
   y=-.5;
   while(y<=0)
      u=rand(1,1);  
      y = gamma + eta * tan(pi*(u-c)+c-.5);
   end
   v=-log(rand(1,1));
   aux=v+log(1.0+((y-gamma)/eta)^2)+gamma*log(y/gamma)-y+gamma;
end;

if (flag==1)
   x = y/beta*(rand(1))^(1.0/(alpha-1));
else
   x = y/beta;
end
