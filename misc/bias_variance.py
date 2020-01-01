
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

X_train = np.c_[.5, 1].T
y_train = [.5, 1]
X_test = np.c_[0, 2].T

#

classifiers = {'$\lambda=0$': linear_model.LinearRegression(),
               '$\lambda=0.5$': linear_model.Ridge(alpha=.5)}

for name, clf in classifiers.items():
    np.random.seed(2)
    fig, ax = plt.subplots(figsize=(4, 3))

    intercepts = []
    slopes = []
    for _ in range(10):
        y = .05 * np.random.normal(size=(2, 1)) + X_train
        clf.fit(X_train, y)
        intercepts.append(clf.intercept_)
        slopes.append(clf.coef_[0])

        ax.plot(X_test, clf.predict(X_test), color='gray')
        ax.scatter(X_train, y, s=3, c='gray', marker='o', zorder=10)

    clf.intercept_ = np.mean(intercepts)
    clf.coef_ = np.mean(slopes)
    ax.plot(X_test, clf.predict(X_test), linewidth=2, color='blue')
    ax.scatter(X_train, y_train, s=30, c='red', marker='o', zorder=10)

    ax.set_title(name)
    ax.set_xlim(0, 2)
    ax.set_ylim((0, 1.6))
    ax.set_xlabel('X')
    ax.set_ylabel('y')

    fig.tight_layout()

plt.show()
