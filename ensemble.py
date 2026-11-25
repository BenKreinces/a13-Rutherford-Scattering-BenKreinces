import numpy as np
from model import modelU, force
rng = np.random.default_rng(seed = 22)
from scattering import VerletEnsemble

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

def main():
    energies = [0.75, 1.25, 5.00]
    N = 100000
    m = 1
    for E in energies:
        initPos = rng.uniform(-50, 50, N)
        x, y, vx, vy, theta, rmin = runEnsemble(initPos, E, h = 0.05)

if __name__ == "__main__":
    main()

