import os
import sys
import warnings

from numpy import mean, nan, round_
from numpy.random import RandomState
from obspy import read_events
from obspy.geodetics.base import kilometers2degrees as k2d
from pandas import DataFrame, Series, concat, date_range, merge, read_csv
from pyproj import Proj
from tqdm import tqdm

from core.Events2Catalog import feedCatalog
from core.Extra import clearRays, getHer, getRMS, getZer, handleNone
from core.Fault import distributeEqOnFault
from core.Forward import trace
from core.Station import loadStationNoiseModel
from core.VelocityModel import loadVelocityModel

warnings.filterwarnings("ignore")


def generateTravelTimes(config, station_P_df, station_S_df, event, eid):
    vp, vs = loadVelocityModel()
    travelTime_P_df = trace(vp, event, eid, station_P_df, "P")
    travelTime_S_df = trace(vs, event, eid, station_S_df, "S")
    travelTime_df = concat([travelTime_P_df, travelTime_S_df],
                           ignore_index=True, sort=False)  # type: ignore
    return travelTime_df


def checkEventInsideVelocityGrid(hypocenter_df):
    vp, _ = loadVelocityModel()
    xmin, ymin, zmin = vp.min_coords * vp.node_intervals
    xmax, ymax, zmax = vp.max_coords * vp.node_intervals
    X_out = hypocenter_df[(hypocenter_df.x < xmin) |
                          (hypocenter_df.x > xmax)].x.count()
    Y_out = hypocenter_df[(hypocenter_df.y < ymin) |
                          (hypocenter_df.y > ymax)].y.count()
    Z_out = hypocenter_df[(hypocenter_df.z < zmin) |
                          (hypocenter_df.z > zmax)].z.count()
    if X_out != 0 or Y_out != 0:
        print("! > events found outside XY velocity grid:")
        print(" - Increase XY dimension ...")
        print(" - Change fault dimension ...")
        sys.exit()
    if Z_out != 0:
        print("! > events found outside Z velocity grid:")
        print(" - Change fault dimension, you may increase 'depth' parameter ...")
        sys.exit()


def generateCandidateHypocenters(config, fault):
    org = config["FSS"]["Catalog"]["date"]
    dt = config["FSS"]["Catalog"]["dt"]
    clat = config["StudyArea"]["lat"]
    clon = config["StudyArea"]["lon"]
    proj = Proj(f"+proj=sterea\
            +lon_0={clon}\
            +lat_0={clat}\
            +units=km")
    rndID = fault["rndID"]
    rng = RandomState(rndID)
    noEvents = fault["noEvents"]
    minEventSpacing = k2d(fault["minEventSpacing"])
    dx = fault["dx"]
    dy = fault["dy"]
    width = fault["width"]
    length = fault["length"]
    depth = fault["depth"]
    strike = fault["strike"]
    dip = fault["dip"]
    cLon = config["StudyArea"]["lon"] + k2d(dx)
    cLat = config["StudyArea"]["lat"] + k2d(dy)
    lonMin, lonMax = cLon - k2d(width), cLon + k2d(width)
    latMin, latMax = cLat - k2d(length), cLat + k2d(length)
    bound = [lonMin, lonMax, latMin, latMax]
    hypocenter_df = distributeEqOnFault(
        bound, noEvents, minEventSpacing,
        depth, strike, dip, rndID)
    hypocenter_df["OriginTime"] = date_range(
        org, periods=noEvents, freq=f"{dt}S")
    hypocenter_df[["x", "y"]] = hypocenter_df.apply(
        lambda x: Series(
            proj(longitude=x.Longitude, latitude=x.Latitude)), axis=1)
    hypocenter_df["z"] = hypocenter_df["Depth"]
    magnitudes = rng.gamma(shape=4, scale=0.5, size=noEvents)
    hypocenter_df["Mag"] = magnitudes
    checkEventInsideVelocityGrid(hypocenter_df)

    return hypocenter_df


def catalog2hypocenters(config):
    catalogPath = os.path.join(config["RSS"]["Inputs"]["catalogFile"])
    catalog = read_events(catalogPath)
    data = []
    clat = config["StudyArea"]["lat"]
    clon = config["StudyArea"]["lon"]
    proj = Proj(f"+proj=sterea\
            +lon_0={clon}\
            +lat_0={clat}\
            +units=km")
    for event in catalog:
        d = {
            "Longitude": event.preferred_origin().longitude,
            "Latitude": event.preferred_origin().latitude,
            "Depth": event.preferred_origin().depth*1e-3,
            "OriginTime": event.preferred_origin().time
        }
        data.append(d)
    hypocenter_df = DataFrame(data)
    hypocenter_df[["x", "y"]] = hypocenter_df.apply(
        lambda x: Series(
            proj(longitude=x.Longitude, latitude=x.Latitude)), axis=1)
    hypocenter_df["z"] = hypocenter_df["Depth"]
    return hypocenter_df


def catalog2stations(event, stationNoiseModel):
    stationPath = os.path.join("inputs", "stations.csv")
    stations = read_csv(stationPath)
    stationNoiseModel = DataFrame(stationNoiseModel).T
    stationNoiseModel["code"] = stationNoiseModel.index
    stations = merge(stations, stationNoiseModel, on="code")
    picks = event.picks
    arrivals = event.preferred_origin().arrivals
    stations = merge(stations, stationNoiseModel, on="code")
    stations_P = [
        arrival.pick_id for arrival in arrivals if "P" in arrival.phase.upper()]
    stations_S = [
        arrival.pick_id for arrival in arrivals if "S" in arrival.phase.upper()]
    stations_P = [
        pick.waveform_id.station_code for pick in picks if pick.resource_id in stations_P]
    stations_S = [
        pick.waveform_id.station_code for pick in picks if pick.resource_id in stations_S]
    mask_P = stations.code.isin(stations_P)
    mask_S = stations.code.isin(stations_S)
    stations_P = stations[mask_P]
    stations_S = stations[mask_S]
    return stations_P, stations_S


def generateCandidateStations(config, stationNoiseModel, eid):
    stationPath = os.path.join("inputs", "stations.csv")
    stations = read_csv(stationPath)
    stationNoiseModel = DataFrame(stationNoiseModel).T
    stationNoiseModel["code"] = stationNoiseModel.index
    stations = merge(stations, stationNoiseModel, on="code")
    rng = RandomState(eid)
    minPphasePercentage = config["FSS"]["Catalog"]["minPphasePercentage"]
    maxPphasePercentage = config["FSS"]["Catalog"]["maxPphasePercentage"]
    minSphasePercentage = config["FSS"]["Catalog"]["minSphasePercentage"]
    maxSphasePercentage = config["FSS"]["Catalog"]["maxSphasePercentage"]
    minPphase = int(minPphasePercentage * len(stations) * 1e-2)
    maxPphase = int(maxPphasePercentage * len(stations) * 1e-2)
    minSphase = int(minSphasePercentage * len(stations) * 1e-2)
    maxSphase = int(maxSphasePercentage * len(stations) * 1e-2)
    k_P = rng.randint(max(4, minPphase), maxPphase+1)
    k_S = rng.randint(minSphase, min(k_P, maxSphase)+1)
    candidateStations_P = stations.sample(
        k_P, replace=False, random_state=rng, weights="probabilityOfOccurrence")  # type: ignore
    candidateStations_S = candidateStations_P.sample(
        k_S, replace=False, random_state=rng, weights="probabilityOfOccurrence")  # type: ignore
    return candidateStations_P, candidateStations_S


def generateHypocenters(config):
    clearRays()
    hypocenterPath = os.path.join("results", "hypocenters.csv")
    hypocenter_df = DataFrame()
    if config["FSS"]["flag"]:
        for fault in config["FSS"]["Catalog"]["Faults"]:
            new_hypocenters = generateCandidateHypocenters(config, fault)
            hypocenter_df = concat([hypocenter_df, new_hypocenters],
                                   ignore_index=True, sort=False)
    elif config["RSS"]["flag"]:
        hypocenter_df = catalog2hypocenters(config)
    hypocenter_df.to_csv(hypocenterPath, index=False,
                         float_format="%8.4f", na_rep="nan")


def generateBulletin(config, stationNoiseModel):
    hypocentersPath = os.path.join("results", "hypocenters.csv")
    hypocenter_df = read_csv(hypocentersPath)
    bulletin = DataFrame()
    catalog = {}
    station_P_df = DataFrame()
    station_S_df = DataFrame()
    if config["RSS"]["flag"]:
        catalogPath = os.path.join(config["RSS"]["Inputs"]["catalogFile"])
        catalog = read_events(catalogPath)
    desc = "+++ Generating bulletin, travel times ..."
    for eid, event in tqdm(hypocenter_df.iterrows(), desc=desc, unit=" event"):
        if config["FSS"]["flag"]:
            station_P_df, station_S_df = generateCandidateStations(
                config, stationNoiseModel, eid)
        elif config["RSS"]["flag"]:
            evt = catalog[eid]
            station_P_df, station_S_df = catalog2stations(
                evt, stationNoiseModel)
        travelTime_df = generateTravelTimes(
            config, station_P_df, station_S_df, event, eid)
        bulletin = concat([bulletin, travelTime_df],
                          ignore_index=True, sort=False)
    bulletinPath = os.path.join("results", "bulletin.csv")
    bulletin.to_csv(bulletinPath, index=False,
                    float_format="%8.4f", na_rep="nan")


def createCatalog(config, stationNoiseModel):
    hypocenterPath = os.path.join("results", "hypocenters.csv")
    bulletinPath = os.path.join("results", "bulletin.csv")
    hypocenter_df = read_csv(hypocenterPath, parse_dates=["OriginTime"])
    bulletin_df = read_csv(bulletinPath)
    catalogPool = feedCatalog()
    for weighting, w in zip([False, True], ["_unw", "_w"]):
        catalogPath = os.path.join("results", f"catalog{w}.out")
        catalog = catalogPool.setCatalog(
            config, hypocenter_df, bulletin_df, stationNoiseModel, weighting)
        catalog.write(catalogPath, format="NORDIC", high_accuracy=False)


def generateCatalog(config):
    stationNoiseModel = loadStationNoiseModel()
    generateHypocenters(config)
    generateBulletin(config, stationNoiseModel)
    createCatalog(config, stationNoiseModel)


def catalog2xyzm(hypInp, outName):
    """Convert catalog to xyzm file format

    Args:
        hypInp (str): file name of NORDIC file
        catalogFileName (str): file name of xyzm.dat file
    """
    cat = read_events(hypInp)
    outputFile = f"xyzm_{outName:s}.dat"
    catDict = {}
    for i, event in enumerate(cat):
        preferred_origin = event.preferred_origin()
        preferred_magnitude = event.preferred_magnitude()
        arrivals = preferred_origin.arrivals
        ort = preferred_origin.time
        lat = preferred_origin.latitude
        lon = preferred_origin.longitude
        mag = preferred_magnitude.mag if preferred_magnitude else nan
        try:
            dep = preferred_origin.depth*0.001
        except TypeError:
            dep = nan
        try:
            nus = handleNone(
                preferred_origin.quality.used_station_count, dtype="int")
        except AttributeError:
            nus = nan
        nuP = len(
            [arrival.phase for arrival in arrivals if "P" in arrival.phase.upper()])
        nuS = len(
            [arrival.phase for arrival in arrivals if "S" in arrival.phase.upper()])
        mds = handleNone(
            min([handleNone(arrival.distance) for arrival in preferred_origin.arrivals]), degree=True)
        ads = round_(handleNone(
            mean([handleNone(arrival.distance) for arrival in preferred_origin.arrivals]), degree=True), 2)
        try:
            gap = handleNone(
                preferred_origin.quality.azimuthal_gap, dtype="int")
        except AttributeError:
            gap = nan
        rms = getRMS(preferred_origin.arrivals)
        erh = getHer(event)
        erz = getZer(event)
        catDict[i] = {
            "ORT": ort,
            "Lon": lon,
            "Lat": lat,
            "Dep": dep,
            "Mag": mag,
            "Nus": nus,
            "NuP": nuP,
            "NuS": nuS,
            "ADS": ads,
            "MDS": mds,
            "GAP": gap,
            "RMS": rms,
            "ERH": erh,
            "ERZ": erz,
        }
    df = DataFrame(catDict).T
    df = df.replace({"None": nan})
    with open(outputFile, "w") as f:
        df.to_string(f, index=False, float_format="%7.3f")
