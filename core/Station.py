import os
import sys
from pathlib import Path
from string import ascii_letters as as_le

from numpy import arange, meshgrid
from numpy.random import RandomState
from obspy.geodetics.base import kilometers2degrees as k2d
from pandas import DataFrame, Series, read_csv
from pyproj import Proj
from yaml import FullLoader, dump, load


def generateStations(config):
    print("+++ Generating new stations ...")
    Path("inputs").mkdir(parents=True, exist_ok=True)
    stationPath = os.path.join("inputs", "stations.csv")
    clat = config["StudyArea"]["lat"]
    clon = config["StudyArea"]["lon"]
    station_db = {}
    sLon = sLat = 0
    # Projection to convert degrees to km relative to the center of the area
    proj = Proj(f"+proj=sterea\
            +lon_0={clon}\
            +lat_0={clat}\
            +units=km")
    if config["FSS"]["flag"]:
        rng = RandomState(config["FSS"]["Stations"]["rndID"])
        radii = config["FSS"]["Stations"]["radius"]
        nsta = config["FSS"]["Stations"]["noOfStations"]
        minDis = k2d(config["FSS"]["Stations"]["minSpacing"])
        shiftedCLon = clon + k2d(config["FSS"]["Stations"]["dx"])
        shiftedCLat = clat + k2d(config["FSS"]["Stations"]["dy"])
        sCod = ["".join(rng.choice(list(as_le), 4)).upper()
                for _ in range(nsta)]
        sElv = 0 + rng.random_sample(nsta) * 0
        if config["FSS"]["Stations"]["shapeOfDist"] == "c":
            sIgn = rng.choice((-1, 1), nsta)
            sLon = shiftedCLon + rng.random_sample(nsta) * k2d(radii) * sIgn
            sLat = shiftedCLat + rng.random_sample(nsta) * k2d(radii) * sIgn
        elif config["FSS"]["Stations"]["shapeOfDist"] == "r":
            sLon = shiftedCLon + rng.uniform(-k2d(radii), k2d(radii), nsta)
            sLat = shiftedCLat + rng.uniform(-k2d(radii), k2d(radii), nsta)
        elif config["FSS"]["Stations"]["shapeOfDist"] == "g":
            x = arange(shiftedCLon-k2d(radii), shiftedCLon+k2d(radii), minDis)
            y = arange(shiftedCLat-k2d(radii), shiftedCLat+k2d(radii), minDis)
            sLon, sLat = meshgrid(x, y)
            sLon, sLat = sLon.flatten(), sLat.flatten()
            try:
                ids = rng.choice(range(sLon.size), nsta, replace=False)
            except ValueError:
                print("! > Number of requested stations is larger than gridpoints")
                print("- decrease min station spacing 'minSpacing' parameter ...")
                sys.exit()
            sLon, sLat = sLon[ids], sLat[ids]
            sLon += rng.random_sample(nsta) * minDis * 0.1
            sLat += rng.random_sample(nsta) * minDis * 0.1
            sElv += 0.01
        station_db = {
            "code": sCod,
            "lat": sLat,
            "lon": sLon,
            "elv": sElv}
    elif config["RSS"]["flag"]:
        stationsPath = os.path.join(config["RSS"]["Inputs"]["stationFile"])
        station_db = read_csv(stationsPath)
    station_db = DataFrame(station_db)
    station_db["elv"] *= 1e-3
    station_db[["x", "y"]] = station_db.apply(
        lambda x: Series(
            proj(longitude=x.lon, latitude=x.lat)), axis=1)
    station_db["z"] = station_db["elv"]
    station_db.sort_values("code", inplace=True)
    station_db.to_csv(stationPath, index=False, float_format="%8.3f")


def generateStationNoiseModel(config):
    stationPath = os.path.join("inputs", "stations.csv")
    stationNoiseModelPath = os.path.join("inputs", "stationsNoiseModel.yml")
    station_db = read_csv(stationPath)
    station_db.code = station_db.code.str.strip()
    resetFlag = config["FSS"]["Stations"]["resetNoiseModel"]
    if not os.path.exists(stationNoiseModelPath) or resetFlag:
        print("+++ Generating station noise model ...")
        data = {}
        for r, row in station_db.iterrows():
            data[row.code] = {"siteNoiseLevel": [0.0, 0.0],
                              "probabilityOfOccurrence": 1.0,
                              "S_P_Error": 1.0
                              }
        with open(stationNoiseModelPath, "w") as f:
            dump(data, f)


def loadStationNoiseModel():
    stationNoiseModelPath = os.path.join("inputs", "stationsNoiseModel.yml")
    with open(stationNoiseModelPath) as f:
        return load(f, Loader=FullLoader)
