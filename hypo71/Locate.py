import os
import latlon as ll
from numpy import nan
from pandas import to_datetime, Series, read_fwf
from core.VelocityModel import loadVelocityModel
from core.Catalog import catalog2xyzm
from hypo71.Input import prepareInputFile
from hypo71.Extra import hypo71Nordic
from pathlib import Path
from glob import glob
from tqdm import tqdm
from shutil import copy


def locateHypo71(config):
    vp, vs = loadVelocityModel()
    stationPath = os.path.join("inputs", "stations.csv")
    stationPath = os.path.abspath(stationPath)
    resetsPath = os.path.join("files", "resets.dat")
    resetsPath = os.path.abspath(resetsPath)
    locationPath = os.path.join("results", "location", "hypo71")
    Path(locationPath).mkdir(parents=True, exist_ok=True)
    catalogs = glob(os.path.join("results", "catalog_*.out"))
    for catalogFile in catalogs:
        copy(catalogFile, locationPath)
    root = os.getcwd()
    os.chdir(locationPath)
    desc = "+++ Locate catalog using 'Hypo71' ..."
    for catalogPath in tqdm(glob("catalog_*.out"), desc=desc):
        outName = catalogPath.split("_")[1].split(".")[0]
        prepareInputFile(config, resetsPath, stationPath, vp, vs, catalogPath)
        cmd = "Hypo71PC < input.dat >/dev/null 2>/dev/null"
        os.system(cmd)
        writexyzm(outName)
        hypo71Nordic(outName)
    catalog2xyzm("catalog_unw.out", "initial")
    os.chdir(root)


def writexyzm(outName):
    catalog_db = loadhypo71Out(outName)
    catalog_db.replace("*****", nan, inplace=True)
    outputFile = f"xyzm_{outName}.dat"
    catalog_db["year"] = 2000 + catalog_db.yy
    catalog_db["month"] = catalog_db.mo
    catalog_db["day"] = catalog_db.dd
    catalog_db["hour"] = catalog_db.hh
    catalog_db["minute"] = catalog_db.mm
    catalog_db["second"] = catalog_db.sssss
    catalog_db["ORT"] = to_datetime(catalog_db[["year",
                                                "month",
                                                "day",
                                                "hour",
                                                "minute",
                                                "second"]])
    catalog_db["ORT"] = catalog_db["ORT"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    catalog_db["Lon"] = catalog_db.apply(lambda x: Series(
        ll.Longitude(degree=x.xdd, minute=x.xmmmm).decimal_degree), axis=1)
    catalog_db["Lat"] = catalog_db.apply(lambda x: Series(
        ll.Latitude(degree=x.yd, minute=x.ymmmm).decimal_degree), axis=1)
    catalog_db["Dep"] = catalog_db.depth_
    catalog_db["Mag"] = catalog_db.mag
    catalog_db["Nus"] = catalog_db.ns
    catalog_db["NuP"] = nan
    catalog_db["NuS"] = nan
    catalog_db["ADS"] = nan
    catalog_db["MDS"] = catalog_db.dmin
    catalog_db["GAP"] = catalog_db.gap
    catalog_db["RMS"] = catalog_db.rms_
    catalog_db["ERH"] = catalog_db.erh__
    catalog_db["ERZ"] = catalog_db.erz__
    columns = ["ORT", "Lon", "Lat", "Dep", "Mag",
               "Nus", "NuP", "NuS", "ADS", "MDS", "GAP", "RMS", "ERH", "ERZ"]
    with open(outputFile, "w") as f:
        catalog_db.to_string(f, columns=columns, index=False, float_format="%7.3f")


def loadhypo71Out(outName):
    names = ["yy", "mo", "dd", "A", "hh", "mm", "B", "sssss", "C",
             "yd", "D", "ymmmm", "E", "xdd", "F", "xmmmm", "G",
             "depth_", "HHHH", "mag", "L", "ns", "M", "gap", "N", "dmin", "O",
             "rms_", "erh__", "erz__", "P", "qm"]
    widths = [len(name) for name in names]
    bulletin_df = read_fwf(f"hyp71_{outName}.dat", names=names,
                           widths=widths, header=0)
    return bulletin_df
