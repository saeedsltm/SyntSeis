import os
import sys
from glob import glob
from pathlib import Path

import pykonal
from numpy import clip
from numpy.random import RandomState
from pandas import read_csv
from scipy.ndimage import gaussian_filter


def readVelocityFile(config, z):
    vmPath = os.path.join(config["RSS"]["Inputs"]["velocityFile"])
    df = read_csv(vmPath)
    depths = df.depth
    for i, (z_low, z_high) in enumerate(zip(depths[:-1], depths[1:])):
        if z >= z_low and z < z_high:
            return df.vp[i], df.vpvs[i]
    else:
        idx = len(df) - 1
        return df.vp[idx], df.vpvs[idx]


def createVelocityModel(config):
    """
    Create Velocity Model

    Parameters
    ----------
    config : dictionary
        a dictionary contains user-defined parameters.

    Returns
    -------
    vp : pykonal object
        P velocity model.
    vs : pykonal object
        S velocity model.

    """
    Path("model").mkdir(parents=True, exist_ok=True)
    # Path for which the velocity models will be saved in.
    vpPath = os.path.join("model", "vp.mod")
    vsPath = os.path.join("model", "vs.mod")
    if config["Model"]["reset"]:
        print("+++ Generating new P and S velocity models...")
        rng = RandomState(config["Model"]["rndID"])
        # Node spacing of the velocity model
        dx, dy, dz = (config["Model"]["dx"],
                      config["Model"]["dy"],
                      config["Model"]["dz"])
        # Number of nodes in x,y and z direction
        nx, ny, nz = (config["Model"]["nx"],
                      config["Model"]["ny"],
                      config["Model"]["nz"])
        # Create P and S empty velocity models
        vp = pykonal.fields.ScalarField3D(coord_sys="cartesian")
        vs = pykonal.fields.ScalarField3D(coord_sys="cartesian")
        # Initialize P and S coordinates
        vp.min_coords = -nx * dx * 0.5, -ny * dy * 0.5, 0
        vs.min_coords = -nx * dx * 0.5, -ny * dy * 0.5, 0
        # Define P and S velocity models structure
        vp.node_intervals = dx, dy, dz
        vs.node_intervals = dx, dy, dz
        vp.npts = nx, ny, nz
        vs.npts = nx, ny, nz
        # P velocity in shallow crust
        vp0 = config["Model"]["vp0"]
        # Increasing velocity ratio between velocity and Depth
        vDepthCC = config["Model"]["vDepthCC"]
        # Background noise will added to velocity
        vPertCoef = config["Model"]["vPertCoef"]
        # Ratio for blurring velocity
        vGamma = config["Model"]["vGamma"]
        # Ratio for blurring anomalies
        vAnomalyGamma = config["Model"]["vAnomalyGamma"]
        # Position of user-defined anomalies
        anomaliesPos = config["Model"]["anomaliesPos"]
        # Perturbation of user-defined anomalies
        anomaliesPert = config["Model"]["anomaliesPert"]
        # Bounds for P velocity
        vpLimits = config["Model"]["vpLimits"]
        vsLimits = config["Model"]["vsLimits"]
        # Vp/Vs
        vpvs = config["Model"]["vpvs"]
        if config["FSS"]["flag"]:
            # Generate 2D velocity model increasing with depth, then smooth it
            for iz in range(vp.npts[2]):
                depth = vp.nodes[0, 0, iz, 2]
                vp.values[:, :, iz] = vp0 + vDepthCC * depth
            vp.values += vPertCoef * rng.randn(*vp.npts)
            vp.values = gaussian_filter(vp.values, vGamma)
        elif config["RSS"]["flag"]:
            for iz in range(vp.npts[2]):
                depth = vp.nodes[0, 0, iz, 2]
                Vp, VpVs = readVelocityFile(config, depth)
                vp.values[:, :, iz] = Vp
                vpvs = VpVs.mean()
        # Put anomalies in P and S 2D velocity models
        for anPos, anPer in zip(anomaliesPos, anomaliesPert):
            x1, x2 = anPos[0], anPos[1]
            y1, y2 = anPos[2], anPos[3]
            z1, z2 = anPos[4], anPos[5]
            vp.values[x1:x2, y1:y2, z1:z2] += anPer
        vp.values = gaussian_filter(vp.values, vAnomalyGamma)
        vs.values = vp.values/vpvs
        # Clip for extreme P and S velocities
        vp.values = clip(vp.values, vpLimits[0], vpLimits[1])
        vs.values = clip(vs.values, vsLimits[0], vsLimits[1])
        # Save
        for f in glob(os.path.join("model", "*.mod")):
            os.remove(f)
        vp.to_hdf(vpPath)
        vs.to_hdf(vsPath)
        # Return P and S velocity models
    elif os.path.exists(vpPath) and os.path.exists(vsPath):
        print("+++ Reading P and S velocity models ...")
        vp = pykonal.fields.read_hdf(vpPath)
        vs = pykonal.fields.read_hdf(vsPath)
    else:
        print("+++ No P and S velocity modes found!")
        sys.exit()

    return vp, vs


def loadVelocityModel():
    # Path for P and S velocity model files
    vpPath = os.path.join("model", "vp.mod")
    vsPath = os.path.join("model", "vs.mod")
    vp = pykonal.fields.read_hdf(vpPath)
    vs = pykonal.fields.read_hdf(vsPath)
    return vp, vs
