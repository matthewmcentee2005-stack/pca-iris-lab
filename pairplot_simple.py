import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
names = ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]
species = ["setosa", "versicolor", "virginica"]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        for k in range(3):
            if i == j:
                ax.hist(X[y == k, j], bins=10, alpha=0.5, label=species[k])
            else:
                ax.scatter(X[y == k, j], X[y == k, i], s=10, label=species[k])
        if i == 3:
            ax.set_xlabel(names[j])
        if j == 0:
            ax.set_ylabel(names[i])

axes[0, 0].set_ylabel("Count")
axes[0, 1].set_ylabel("Sepal length (cm)")
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=3)
fig.suptitle("Iris data: each variable plotted against the others", x=0.35)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig("pairplot.png", dpi=150)
plt.show()
