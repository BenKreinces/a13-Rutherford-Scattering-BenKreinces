
import numpy as np
import matplotlib.pyplot as plt
from scattering import runTrajectory, diagnostics
from model import force
import os

energies = [0.75, 1.25, 5.00]


def modelChecks():
    print("Force Checks")
    print("Origin:", force(0, 0))
    print("Inside boundary:", force(0.999999, 0))
    print("At boundary:", force(1, 0))
    print("Outside boundary:", force(1.000001, 0))

def headOnTest(h):
    print("Head-On Tests")
    for E in energies:
        states = runTrajectory(0, E, h)
        theta, rmin, error = diagnostics(states)
        print(f"E = {E}, theta = {theta}, rmin = {rmin}, error = {error}")

def symmetryTest(h):
    print("Symmetry Tests")
    for E in energies:
        for x0 in [-0.5, 0.5]:
            states = runTrajectory(x0, E, h)
            theta, rmin, error = diagnostics(states)
            print(f"E = {E}, x0 = {x0}, theta = {theta}, rmin = {rmin}")

def timestepStudy():
    timesteps = [0.1, 0.05, 0.01]
    positions = [0.1, 0.5, 3]
    print("Timestep Study")
    for E in energies:
        for x0 in positions:
            for h in timesteps:
                states = runTrajectory(x0, E, h)
                theta, rmin, error = diagnostics(states)
                print(f"Energy = {E} \n Position = {x0} \n Timestep = {h} \n Theta = {theta} \n RMin = {rmin} \n Error = {error}")


def plotTrajectories(h):
    positions = [-3, -1, -0.2, -0.05, 0.05, 0.2, 1, 3]
    colors = ['blue', 'orange', 'green']
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, E in enumerate(energies):
        for j, x0 in enumerate(positions):
            states = runTrajectory(x0, E, h)
            x = states[:, 0]
            y = states[:, 1]
            ax.plot(x, y, color=colors[i], linewidth=1.2, label=f"E = {E}" if j == 0 else None)        
            
            # Arrow showing direction of travel
            indices = np.where((np.abs(x) < 4.5) & (np.abs(y) < 4.5))[0]
            if len(indices) > 1:
                index = indices[len(indices)//2]
                vx = states[index, 2]
                vy = states[index, 3]
                speed = np.hypot(vx, vy)
                if speed > 0:
                    ax.quiver(
                        x[index], y[index],
                        vx/speed, vy/speed,
                        color=colors[i],
                        angles='xy',
                        scale_units='xy',
                        scale=3,
                        width=0.004)

    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2, label='Boundary')
    ax.add_patch(circle)
    ax.annotate(
        "Incoming beam",
        xy=(-4.3, -3),
        xytext=(-4.3, -4.5),
        arrowprops=dict(arrowstyle="->"),
        ha='center',
        fontsize=9
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Representative Rutherford Scattering Trajectories")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/trajectories.png", dpi=300)
    plt.show()


def main():
    h = 0.01
    modelChecks()
    headOnTest(h)
    symmetryTest(h)
    timestepStudy()
    plotTrajectories(h)


if __name__ == "__main__":
    main()