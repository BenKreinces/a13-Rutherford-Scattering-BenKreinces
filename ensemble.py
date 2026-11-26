import numpy as np
from model import modelU, force
rng = np.random.default_rng(seed = 22)
from scattering import VerletEnsemble
import matplotlib.pyplot as plt
def runEnsemble(incomingPositions, E, h, m=1, L = -100, rdet = 500):
    N = len(incomingPositions)
    x = incomingPositions.copy()
    y = np.full(N, L, dtype=float)
    vx = np.zeros(N, dtype= float)
    vy = np.sqrt((E - modelU(x, y)) * 2 / m)  
    Fx, Fy = force(x, y)
    ax, ay = Fx / m, Fy / m
    rmin = np.hypot(x, y)
    active = np.ones(N, dtype=bool)
    while np.any(active):
        state = (x[active], y[active], vx[active], vy[active], ax[active], ay[active])
        x[active], y[active], vx[active], vy[active], ax[active], ay[active] = VerletEnsemble(state, h, m)
        r = np.hypot(x[active], y[active])
        rmin[active] = np.minimum(rmin[active], r)
        active[active] = r < rdet
    theta = np.degrees(np.arctan2(x, y))
    return x, y, vx, vy, theta, rmin

def plotEnsemble():
    energies = [0.75, 1.25, 5.00]
    N = 100000
    m = 1
    h = 0.05
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots(1, 3, figsize=(15, 5))
    fig4, ax4 = plt.subplots(1, 3, figsize=(15, 5))
    fig5, ax5 = plt.subplots(1, 3, figsize=(15, 5))
    fig6, ax6 = plt.subplots(1, 3, figsize=(15, 5))
    rminPlots = []
    thetaPlots = []
    rminZoomPlots = []
    thetaZoomPlots = []
    for i, E in enumerate(energies):
        initPos = rng.uniform(-50, 50, N)
        x, y, vx, vy, theta, rmin = runEnsemble(initPos, E, h, m)
        ax1.hist(initPos, bins=100, range=(-50, 50), histtype='step', label=f"E = {E}")
        ax2.hist(theta, bins=100, range=(-180, 180), histtype='step', label=f"E = {E}")
        rminPlots.append(ax3[i].hist2d(initPos, rmin, bins=[100, 100],range=[[-50, 50], [0, 50]], cmin=1))
        thetaPlots.append(ax4[i].hist2d(initPos, theta, bins=[100, 100], range=[[-50, 50], [-180, 180]], cmin=1))
        rminZoomPlots.append(ax5[i].hist2d(initPos, rmin, bins=[100, 100], range=[[-5, 5], [0, 6]], cmin=1))
        thetaZoomPlots.append(ax6[i].hist2d(initPos, theta, bins=[100, 100], range=[[-5, 5], [-180, 180]], cmin=1))
        ax3[i].axhline(1, color='red', linestyle='--')
        ax5[i].axhline(1, color='red', linestyle='--')
        for ax in [ax3[i], ax4[i], ax5[i], ax6[i]]:
            ax.set_title(f"E = {E}")
            ax.set_xlabel("Initial Position")
        ax3[i].set_ylabel("Minimum Radius")
        ax4[i].set_ylabel("Scattering Angle (degrees)")
        ax5[i].set_ylabel("Minimum Radius Zoomed")
        ax6[i].set_ylabel("Scattering Angle (degrees) Zoomed")
        #Stats
        backscatter = np.count_nonzero(np.abs(theta) > 90)
        penetration = np.count_nonzero(rmin < 1)
        print(f"E = {E}")
        print(f"Backscatter count: {backscatter}")
        print(f"Backscatter fraction: {backscatter/N}")
        print(f"Penetration count: {penetration}")
        print(f"Penetration fraction: {penetration/N}")

    
    for fig, axes, plots in [
        (fig3, ax3, rminPlots),
        (fig4, ax4, thetaPlots),
        (fig5, ax5, rminZoomPlots),
        (fig6, ax6, thetaZoomPlots)
        ]:
        vmax = max(np.nanmax(plot[0]) for plot in plots)
        for plot in plots:
            plot[3].set_clim(0, vmax)

        fig.colorbar(plots[0][3], ax=axes, label="Counts")
    ax1.set_xlabel("Initial Position")
    ax1.set_ylabel("Counts")
    ax1.set_title("Incoming Particle Distribution")
    ax1.legend()
    ax2.set_xlabel("Scattering Angle (degrees)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Scattering Angle Distribution")
    ax2.legend()


    fig1.savefig("plots/incomingPositions.png", dpi=300, bbox_inches="tight")
    fig2.savefig("plots/scatteringAngles.png", dpi=300, bbox_inches="tight")
    fig3.savefig("plots/rMinFull.png", dpi=300, bbox_inches="tight")
    fig4.savefig("plots/thetaFull.png", dpi=300, bbox_inches="tight")
    fig5.savefig("plots/rMinZoom.png", dpi=300, bbox_inches="tight")
    fig6.savefig("plots/thetaZoom.png", dpi=300, bbox_inches="tight")

    plt.show()
        



def main():
   plotEnsemble()
   



if __name__ == "__main__":
    main()

