from hypodd.Input import prepareHypoddInputs
from hypodd.Extra import writexyzm, hypoDD2nordic
from core.Catalog import catalog2xyzm
import os
from pathlib import Path
from glob import glob
from shutil import copy
from tqdm import tqdm
from core.VelocityModel import loadVelocityModel


def locateHypoDD(config):
    stationPath = os.path.join("inputs", "stations.csv")
    stationFile = os.path.abspath(stationPath)
    locationPath = os.path.join("results", "location", "hypoDD")
    Path(locationPath).mkdir(parents=True, exist_ok=True)
    catalogs = glob(os.path.join("results", "catalog_*.out"))
    vp, vs = loadVelocityModel()
    for catalogFile in catalogs:
        copy(catalogFile, locationPath)
    root = os.getcwd()
    os.chdir(locationPath)
    desc = "+++ Locate catalog using 'HypoDD' ..."
    for catalogFile in tqdm(glob("catalog_*.out"), desc=desc, unit=" catalog"):
        outName = catalogFile.split("_")[1].split(".")[0]
        prepareHypoddInputs(config,
                            catalogFile,
                            stationFile,
                            vp, vs,
                            locationPath)
        cmd = "ph2dt ph2dt.inp >/dev/null 2>/dev/null"
        os.system(cmd)
        cmd = "hypoDD hypoDD.inp >/dev/null 2>/dev/null"
        os.system(cmd)
        writexyzm(outName)
        hypoDD2nordic(outName)
        for f in glob("hypoDD.reloc*"):
            os.remove(f)
    catalog2xyzm("catalog_unw.out", "initial")
    os.chdir(root)
