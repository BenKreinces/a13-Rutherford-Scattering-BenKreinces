from model import modelU
import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    U = modelU(X, Y)
    fig1, ax1 = plt.subplots()
    heatmap = ax1.contourf(X, Y, U, levels=100, cmap='viridis')
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linewidth=2, label='R = 1')
    ax1.add_patch(circle)
    ax1.set_aspect('equal')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    fig1.colorbar(heatmap, ax=ax1, label="Potential U")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Simplified Potential")
    ax1.legend()
    plt.savefig("plots/potential.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

