import matplotlib.pyplot as plt

def plot_residual(x, y, labels, title, x_axis='Iteration', y_axis='Residual'):
    markers=['s', 'o', '*', '.']
    for ii in range(len(labels)):
        plt.semilogy(x, y[ii], label=labels[ii], marker=markers[ii])
    plt.legend()
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()
    plt.show()

