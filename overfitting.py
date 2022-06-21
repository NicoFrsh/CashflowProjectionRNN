# Over-/Underfitting example
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_oneway

# Plot overfitting loss
# x = np.arange(start=0,stop=5, step=1)
# y = np.array([1.0, 0.6, 0.4, 0.35, 0.3])
# f_overfit = np.poly1d(np.polyfit(x,y,2))
x_f = np.linspace(0,30, 50)

# Underfitting loss
f_train_underfit = (1/(x_f+1)) + 0.2
f_test_underfit = f_train_underfit + 0.2

plt.figure(2)
# plt.plot(x_f, f_overfit(x_f))
plt.plot(x_f, f_train_underfit, label='Training')
plt.plot(x_f, f_test_underfit, label='Test')
plt.xlabel('Epoche')
plt.xticks([])
plt.yticks([0.0])
plt.ylabel('Kosten')
plt.legend()
plt.ylim(0.0, 1.0)
# plt.savefig('./plots/underfitting.pdf')


# Overfitting loss
f_train_overfit = (1/(x_f+1))
f_test_overfit = f_train_overfit + (x_f*x_f/5500) + 0.1# + x_f/10000 + 0.1

plt.figure(3)
# plt.plot(x_f, f_overfit(x_f))
plt.plot(x_f, f_train_overfit, label='Training')
plt.plot(x_f, f_test_overfit, label='Test')
plt.xlabel('Epoche')
plt.xticks([])
plt.yticks([0.0])
plt.ylabel('Kosten')
plt.legend()
plt.ylim(0.0, 1.0)
# plt.savefig('./plots/overfitting.pdf')

# xp = np.linspace(0,6,50)
# x_test = np.linspace(0,6,5)
# y = xp*xp - xp
# y = y + np.random.normal(0,1,len(xp))
# y_test = x_test * x_test - x_test + np.random.normal(0,1,len(x_test))

# p_2 = np.poly1d(np.polyfit(xp, y, 2))
# p_30 = np.poly1d(np.polyfit(xp, y, len(xp)))

# plt.figure(0)
# plt.plot(xp,y, '.')
# plt.plot(x_test, y_test, '.')
# plt.plot(np.linspace(0,6,100), p_2(np.linspace(0,6,100)), '--')
# plt.plot(np.linspace(0,6,100), p_30(np.linspace(0,6,100)), '--')

# x_train = np.arange(0,7,1)
# y_train = np.array([0.0, 0.7, 0.9, 1.0, 0.7, 0.6, 0.4])

# z = np.polyfit(x_train,y_train,2)
# p = np.poly1d(z)
# p10 = np.poly1d(np.polyfit(x_train,y_train,6))
# xp = np.linspace(0,6,100)

# x_test = x_train - 0.5
# x_test = x_test[1:]
# y_test = np.array([0.35, 0.89, 0.95, 0.8, 0.7, 0.5])

# plt.figure(1)
# plt.plot(xp,p(xp),'-')
# plt.plot(xp,p10(xp), '--')
# plt.plot(x_train,y_train,'k.')
# plt.plot(x_test, y_test, 'r.')
plt.show()