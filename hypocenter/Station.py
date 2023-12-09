from pandas import Series, read_csv
from obspy.geodetics.base import degrees2kilometers as d2k
from numpy.random import RandomState
from numpy import mean, sqrt, array, linspace, gradient
import os
import latlon as ll
from core.VelocityModel import loadVelocityModel
from core.Extra import roundTo


def toSTATION0HYP(config):
    print("+++ Generating STATION0.HYP file ...")
    rng = RandomState(config["FPS"]["VelocityModel"]["rndID"])
    nLayers = config["FPS"]["VelocityModel"]["numberOfLayers"]
    stationPath = os.path.join("inputs", "stations.csv")
    resetsPath = os.path.join("files", "resets.dat")
    station0hypPath = os.path.join("results", "STATION0.HYP")
    station_db = read_csv(stationPath)
    station_db.code = station_db.code.str.strip()
    vp, vs = loadVelocityModel()
    velocities = vp.values.mean(axis=1).mean(axis=0)
    depths = vp.nodes.mean(axis=1).mean(axis=0)[:, -1]
    _, _, nz = vp.npts
    if config["FPS"]["VelocityModel"]["choseVelocityLayers"] == "r":
        idz = sorted(rng.choice(range(0, nz, 2), nLayers, replace=False))
    elif config["FPS"]["VelocityModel"]["choseVelocityLayers"] == "e":
        idz = linspace(0, nz/2, nLayers, dtype=int)
    elif config["FPS"]["VelocityModel"]["choseVelocityLayers"] == "g":
        g = Series(gradient(velocities))
        idz = array(depths[~g.duplicated().values] + 1, dtype=int)
        print(g, idz)
    idz[0] = 0
    velocities = velocities[idz]
    depths = depths[idz]
    depths = array(depths, dtype=int)
    trialDepth = 10
    xNear = mean(station_db.apply(lambda x: mean(
        Series(sqrt((x.lon-station_db.lon)**2 + (x.lat-station_db.lat)**2))),
        axis=1))
    xNear = roundTo(d2k(xNear), base=5)
    xFar = 2.5*xNear
    VpVs = (vp.values.mean(axis=1).mean(axis=0) /
            vs.values.mean(axis=1).mean(axis=0)).mean()
    stationLine = "  {code:4s}{latDeg:2.0f}{latMin:05.2f}N {lonDeg:2.0f}\
{lonMin:05.2f}E{elv:4.0f}\n"
    modelLine = " {v:5.2f}  {z:6.3f}             \n"
    controlLine = "{trialDepth:4.0f}.{xNear:4.0f}.{xFar:4.0f}. {VpVs:4.2f}"
    with open(resetsPath) as f, open(station0hypPath, "w") as g:
        for line in f:
            g.write(line)
        g.write("\n\n")
        for r, row in station_db.iterrows():
            code = row.code
            lat = ll.Latitude(row.lat)
            lon = ll.Longitude(row.lon)
            elv = row.elv
            g.write(stationLine.format(
                code=code,
                latDeg=lat.degree, latMin=lat.decimal_minute,
                lonDeg=lon.degree, lonMin=lon.decimal_minute,
                elv=elv
            ))
        g.write("\n")
        for v, z in zip(velocities, depths):
            g.write(modelLine.format(
                v=v, z=z
            ))
        g.write("\n")
        g.write(controlLine.format(
            trialDepth=trialDepth, xNear=xNear, xFar=xFar, VpVs=VpVs
        ))
        g.write("\nNew")
