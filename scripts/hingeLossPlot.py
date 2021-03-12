import numpy as np
import matplotlib.pyplot as plt 

def mMax(z):
    a = []
    for i in z:
        a.append(max(0, 1-i))
    return np.array(a)
    
z = np.linspace(-2.0, 2.0, 400)
L01 = np.array(np.sign(z) < 0)
Lhinge = mMax(z)
Lnll = np.log2(1+np.exp(np.negative(z)));
Lbinom = np.log2(1+np.exp(-2*z));
Lexp = np.exp(np.negative(z));

# plt.plot(z, L01, 'k-', 3);
# plt.plot(z, Lnll, 'r--', 3);
# plt.xlabel('z') 
# plt.ylabel('loss') 
# plt.title('nllLoss') 
# plt.legend(['0-1', 'logloss']) 
# plt.show() 

# plt.plot(z, L01, 'k-', 3);
# plt.plot(z, np.array(Lhinge)+0.02, 'b:', 3);
# plt.plot(z, Lnll, 'r--', 3);
# plt.xlabel('z') 
# plt.ylabel('loss') 
# plt.title('hingeLoss')
# plt.legend(['0-1','hinge','logloss']);
# plt.show()

plt.plot(z, L01, 'k-', 3);
plt.plot(z, Lbinom, 'b:', 3);
plt.plot(z, Lexp, 'r--', 3);
plt.xlabel('z') 
plt.ylabel('loss') 
plt.title('expLoss')
plt.legend(['0-1','logloss','exp'])
plt.show()