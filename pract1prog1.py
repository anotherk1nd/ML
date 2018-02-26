from sklearn.datasets import load_iris
irisData = load_iris()
from matplotlib import pyplot as plt

colors = ["red", "green", "blue"]
for i in range(3):
plt.scatter(X[Y==i][:, x], X[Y==i][:,y], c=colors[i], label=irisData.target_names[i])
plt.legend()
plt.xlabel(irisData.feature_names[x])
plt.ylabel(irisData.feature_names[y])
plt.title("Iris Data - size of the sepals only")
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.show()