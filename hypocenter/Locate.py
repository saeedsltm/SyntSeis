from pathlib import Path
from glob import glob
import os
from shutil import copy
from tqdm import tqdm
from hypocenter.Catalog import catalog2xyzm
from hypocenter.Station import toSTATION0HYP


def locateHypocenter(config):
    locationPath = os.path.join("results", "location", "hypocenter")
    Path(locationPath).mkdir(parents=True, exist_ok=True)
    toSTATION0HYP(config)
    catalogs = glob(os.path.join("results", "catalog_*.out"))
    desc = "+++ Locate catalog using 'Hypocenter' ..."
    for catalogFile in tqdm(catalogs, desc=desc):
        copy(catalogFile, locationPath)
    copy(os.path.join("files", "report.inp"), locationPath)
    copy(os.path.join("results", "STATION0.HYP"), locationPath)
    root = os.getcwd()
    os.chdir(locationPath)
    for inpFile in glob("catalog_*.out"):
        with open("hyp.inp", "w") as f:
            f.write("{inpFile:s}\nn\n".format(inpFile=inpFile))
        cmd = "hyp < hyp.inp >/dev/null 2>/dev/null"
        os.system(cmd)
        outName = inpFile.split("_")[1].split(".")[0]
        catalog2xyzm("hyp.out", outName)
    initial = "catalog_unw.out"
    catalog2xyzm(initial, "initial.out")
    os.chdir(root)
