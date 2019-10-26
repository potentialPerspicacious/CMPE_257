import matplotlib.pyplot as plt
import numpy as np


def evaluate_hypo(h, x):
    if len(h.shape) != 1:
        raise Exception('Hypothesis vector has more than 1 Dimension')
    if len(x.shape) != 2:
        raise Exception('Data has more than 2 dimensions')
    if h.shape[0] != x.shape[0]:
        raise Exception('h[0] and X[0] not matching and not equal to 1')

    return np.sign(h @ x)


def percep_algo(h_init, x_train, y_train, iteration=None):
    n = 0
    h = h_init.copy()

    while True:
        y = evaluate_hypo(h, x_train)

        if iteration:
            iteration(n, h)

        corr = y == y_train

        if np.all(corr):
            return h
        else:
            i = np.argmax(~corr)
            h = h + y_train[i] * x_train[:, i]
            # print("False location", i)
            # print(corr)
        n = n + 1
        mc = 0
        for i in range(0, len(corr)):
            if not corr[i]:
                mc = mc + 1
        print("%d Iteration......" % n, "Number of Misclassified points are: ", mc)


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


def plot_iteration(n, h):
    if n % 10 == 0:
        label = 'iteration {}'.format(n)
        hypo_plot(x_cord_hypo, h, 'k:', label=label)


h_actual = np.array([1, 1, 1])
h_initial = np.array([-20, 12, 22])
data_size = 50

x_train = np.vstack([np.ones(data_size), np.random.uniform(-10, 10, (2, data_size)), ])
y_train = evaluate_hypo(h_actual, x_train)
print(x_train[:, ])
print(y_train)

x_cord = x_train[1, :]
y_cord = x_train[2, :]
colors = ['b' if y > 0 else 'r' for y in y_train]

plt.scatter(x_cord, y_cord, c=colors)
x_cord_hypo = np.array([-10, 10])

h_final = percep_algo(h_initial, x_train, y_train, plot_iteration)
print("Final Hypothesis Vector h(x) is: ", h_final)

hypo_plot(x_cord_hypo, h_final, 'k', label='final')
hypo_plot(x_cord_hypo, h_actual, 'y', label='actual')

plt.legend(fontsize=6, loc=1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
