import numpy as np
import matplotlib.pyplot as plt 

err = np.linspace(-3.0, 3.0, 60)
L1 = abs(err);
L2 = err**2;
delta = 1.5;
ind = abs(err) <= delta;
huber = np.multiply(0.5*ind, (err**2)) + np.multiply((1-ind) , (delta*(abs(err)-delta/2)))
vapnik = np.multiply(ind, 0) + np.multiply((1-ind), (abs(err) - delta))

# plt.plot(err, L2, 'r-')
# plt.plot(err, L1, 'b:')
# plt.plot(err, huber, 'g-.')
# plt.legend(['L2','L1','huber', 'location', 'north'])
# plt.ylim((-0.5, 5))   
# plt.title('huberLoss')
# plt.show()

plt.plot(err, L2, 'r-')
plt.plot(err, vapnik, 'b:')
plt.plot(err, huber, 'g-.')
plt.legend(['L2','$ϵ$-insensitive', 'huber', 'location', 'north'])
plt.ylim((-0.5, 5))   
plt.show()