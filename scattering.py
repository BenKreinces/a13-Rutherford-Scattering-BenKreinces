from model import force
import numpy as np
from model import modelU

def Verlet(state, h, m=1, rdet = 500):
    if h <= 0 or m <= 0 or rdet <= 0:
        raise ValueError("h, m, and rdet must be positive")
    states = [state]
    while np.hypot(state[0], state[1]) < rdet:
        rNextX = state[0] + state[2] * h + 0.5 * h**2 * state[4]
        rNextY = state[1] + state[3] * h + 0.5 * h**2 * state[5]
        FX, FY = force(rNextX, rNextY)
        aNextX, aNextY = FX / m, FY / m
        vNextX = state[2] + 0.5 * h * (state[4] + aNextX)
        vNextY = state[3] + 0.5 * h * (state[5] + aNextY)
        state = (rNextX, rNextY,vNextX, vNextY, aNextX, aNextY)
        states.append(state)
    return np.array(states)

def initSpeed(x0, E, L = -100, m = 1):
    U = modelU(x0, L)
    if E <= U:
        raise ValueError("Energy must exceed initial potential")

    v0x, v0y = 0, np.sqrt((E - U) * 2 / m)  
    return v0x, v0y

def runTrajectory(x0, E, h, L = -100, m = 1, rdet = 500):
    v0x, v0y = initSpeed(x0, E, L, m)
    Fx, Fy = force(x0, L)
    state = (x0, L, v0x, v0y, Fx / m, Fy / m)
    states = Verlet(state, h, m ,rdet)
    return states

def diagnostics(states, m=1):
    x = states[:, 0]
    y = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    r = np.sqrt(x**2 + y**2)
    rmin = np.min(r)
    theta = np.degrees(np.arctan2(x[-1], y[-1]))
    energy = 0.5*m*(vx**2 + vy**2) + modelU(x, y)
    error = np.max(np.abs((energy - energy[0])/energy[0]))
    return theta, rmin, error
