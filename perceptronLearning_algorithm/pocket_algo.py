import matplotlib.pyplot as plt
import numpy as np

h_actual = np.array([1, 1, 1])
h_initial = np.array([-20, 12, 22])
data_size = 100

x_train = np.vstack([np.ones(data_size), np.random.uniform(-10, 10, (2, data_size)), ])
x_trainT = np.transpose(x_train)
# print(x_trainT.shape)

n = 0
h = h_initial.copy()

mc = 1
mmc = 100
d = []
"""
y1 = np.ones(data_size // 2)
y2 = np.ones(data_size // 2) * -1
y_f = np.concatenate((y1, y2))
# print(y_f.shape)
"""
y = np.sign(h @ x_train)
# print(y)
y_noise1 = (y[-100:]) * -1
y_noise2 = (y[:-100])
y_f = np.concatenate((y_noise2, y_noise1))
# print(y_f)

while mc != 0 and (n < 200):
    n = n + 1
    mc = 0
    for i in range(0, len(x_trainT)):
        x_point = (x_trainT[i])
        y_point = y_f[i]
        yt = np.dot(x_point, h)
        # print(x_point)
        # print(y_f)
        if y_point == 1 and yt < 0:
            mc = mc + 1
            h = h + 0.01 * np.transpose(x_point)
        elif y_point == -1 and yt > 0:
            mc = mc + 1
            h = h - 0.01 * np.transpose(x_point)
    d.append(mc)
    if mc < mmc:
        mmc = mc
    print("Iteration %d......" % n)
    print("Number of Misclassified points are: ", mc)

print("Minimum no. of Misclassified points are: ", mmc)
print("Pocket Algorithm weights are: ", h.transpose())
h_final = h.transpose()


def hypo_plot(x_cord, h, *plot_args, **plot_kargs):
    if h[2] != 0:
        slope = -h[1] / h[2]
    else:
        return 0
    if h[2] != 0:
        intercept = -h[0] / h[2]
    else:
        return 0
    y_cord = slope * x_cord + intercept
    plt.plot(x_cord, y_cord, *plot_args, **plot_kargs)


x_cord = x_train[1, :]
y_cord = x_train[2, :]
colors = ['b' if y > 0 else 'r' for y in y_f]

plt.scatter(x_cord, y_cord, c=colors)
x_cord_hypo = np.array([-10, 10])

hypo_plot(x_cord_hypo, h_final, 'k', label='final')
hypo_plot(x_cord_hypo, h_actual, 'y', label='actual')

plt.legend(fontsize=6, loc=1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()