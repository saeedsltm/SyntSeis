from obspy import read_events
import latlon as ll
import os
from numpy.random import RandomState
from numpy import array, mean, sqrt, array, linspace, gradient
from pandas import Series, read_csv
from obspy.geodetics.base import degrees2kilometers as d2k
from obspy import UTCDateTime as utc
from core.Extra import roundTo, getPick
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def addResets(phaseFile, resetsPath):
    with open(phaseFile, "w") as f, open(resetsPath) as g:
        f.write("HEAD                     GENERATED USING 'PyHypo71' CODE\n")
        for line in g:
            f.write(line)
        f.write("\n")
        f.write("\n")


def addStation(phaseFile, station_df):
    stationLineFmt = "  {code:4s}{latDeg:2.0f}{latMin:05.2f}N {lonDeg:2.0f}\
{lonMin:05.2f}E{elv:4.0f}\n"
    with open(phaseFile, "a") as f:
        for r, row in station_df.iterrows():
            code = row.code
            lat = ll.Latitude(row.lat)
            lon = ll.Longitude(row.lon)
            elv = row.elv
            f.write(stationLineFmt.format(
                code=code,
                latDeg=lat.degree, latMin=lat.decimal_minute,
                lonDeg=lon.degree, lonMin=lon.decimal_minute,
                elv=elv
            ))
        f.write("\n")


def addVelocityModel(config, vp, vs, phaseFile):
    rng = RandomState(config["FPS"]["VelocityModel"]["rndID"])
    nLayers = config["FPS"]["VelocityModel"]["numberOfLayers"]
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
    idz[0] = 0
    velocities = velocities[idz]
    depths = depths[idz]
    depths = array(depths, dtype=int)
    modelLineFmt = " {v:5.2f}  {z:6.3f}\n"
    with open(phaseFile, "a") as f:
        for v, z in zip(velocities, depths):
            f.write(modelLineFmt.format(
                v=v, z=z
            ))
        f.write("\n")


def addControlLine(phaseFile, station_df, vp, vs):
    trialDepth = 10
    xNear = mean(station_df.apply(lambda x: mean(
        Series(sqrt((x.lon-station_df.lon)**2 + (x.lat-station_df.lat)**2))),
        axis=1))
    xNear = roundTo(d2k(xNear), base=5)
    xFar = 2.5*xNear
    VpVs = (vp.values.mean(axis=1).mean(axis=0) /
            vs.values.mean(axis=1).mean(axis=0)).mean()
    with open(phaseFile, "a") as f:
        f.write(
            f"{trialDepth:4.0f}.{xNear:4.0f}.{xFar:4.0f}. {VpVs:4.2f}    4    0    0    1    1    0    0 0111\n")


def addArrivals(catalogFile, phaseFile):
    catalog = read_events(catalogFile)
    phaseLinePSFmt = "{code:4s} P {wP:1.0f} {Part:15s}      {Stt:6s} S {wS:1.0f}          \n"
    phaseLinePFmt = "{code:4s} P {wP:1.0f} {Part:15s}                          \n"
    with open(phaseFile, "a") as f:
        for event in catalog:
            data = {}
            po = event.preferred_origin()
            arrivals = sorted(po.arrivals, key=lambda p: (p.phase, p.distance))
            picks = event.picks
            for arrival in arrivals:
                pick = getPick(picks, arrival.pick_id)
                code = pick.waveform_id.station_code
                pha = pick.phase_hint.upper()
                try:
                    w = int(pick.extra["nordic_pick_weight"]["value"])
                except (KeyError, ValueError):
                    w = 0
                pht = pick.time.strftime("%y%m%d%H%M%S.%f")[:15]
                if "S" in pha and code in data:
                    ptime = utc.strptime(data[code]["P"]["Part"], "%y%m%d%H%M%S.%f")
                    stime = ptime.second + ptime.microsecond*1e-6 + (pick.time - ptime)
                    pht = "{0:6.2f}".format(stime)
                if code not in data and "P" in pha:
                    data[code] = {"P": {"Part": pht, "wP": w},
                                  "S": None}
                elif "S" in pha and code in data:
                    data[code].update({"S": {"Stt": pht, "wS": w}})
            for code, v in data.items():
                if data[code]["S"] is not None:
                    f.write(phaseLinePSFmt.format(
                        code=code,
                        wP=v["P"]["wP"],
                        Part=v["P"]["Part"],
                        Stt=v["S"]["Stt"],
                        wS=v["S"]["wS"]
                    ))
                else:
                    f.write(phaseLinePFmt.format(
                        code=code,
                        wP=v["P"]["wP"],
                        Part=v["P"]["Part"]
                    ))
            f.write("                 10\n")


def preparPhaseFile(config, resetsPath, stationPath, vp, vs, catalogPath):
    outName = catalogPath.split("_")[1].split(".")[0]
    phasePath = f"phase_{outName}.dat"
    station_df = read_csv(stationPath)
    station_df.code = station_df.code.str.strip()
    addResets(phasePath, resetsPath)
    addStation(phasePath, station_df)
    addVelocityModel(config, vp, vs, phasePath)
    addControlLine(phasePath, station_df, vp, vs)
    addArrivals(catalogPath, phasePath)


def prepareInputFile(config, resetsPath, stationPath, vp, vs, catalogPath):
    preparPhaseFile(config, resetsPath, stationPath, vp, vs, catalogPath)
    inputPath = "input.dat"
    outName = catalogPath.split("_")[1].split(".")[0]
    with open(inputPath, "w") as f:
        f.write(f"phase_{outName}.dat\n")
        f.write(f"print_{outName}.out\n")
        f.write(f"hyp71_{outName}.dat\n")
        f.write("\n\n\n")
