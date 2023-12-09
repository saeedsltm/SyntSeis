from obspy import read_events
import os
from numpy.random import RandomState
from numpy import linspace, array, nan, gradient
from pandas import read_csv, Series


def preparePhaseFile(catalogFile):
    phaseFile = os.path.join("phase.dat")
    catalog = read_events(catalogFile)
    ws = {"0": 1.00,
          "1": 0.75,
          "2": 0.50,
          "3": 0.25,
          "4": 0.00, }
    with open(phaseFile, "w") as f:
        for e, event in enumerate(catalog):
            po = event.preferred_origin()
            pm = event.preferred_magnitude()
            ORT = po.time.strftime("%Y %m %d %H %M %S.%f")
            LAT = po.latitude
            LON = po.longitude
            DEP = po.depth*1e-3
            MAG = pm.mag if pm else nan
            picks = event.picks
            header = f"# {ORT} {LAT:6.3f} {LON:6.3f} {DEP:5.1f} {MAG:4.1f} 0.0 0.0 0.0 {e+1:9.0f}\n"
            f.write(header)
            for arrival in po.arrivals:
                pick = [p for p in picks if p.resource_id == arrival.pick_id][0]
                sta = pick.waveform_id.station_code
                tt = pick.time - po.time
                w = ws[pick.extra["nordic_pick_weight"]["value"]]
                pha = pick.phase_hint
                phase = f"{sta:4s} {tt:6.3f} {w:4.2f} {pha:1s}\n"
                f.write(phase)


def prepareStationFile(stationFile):
    station_df = read_csv(stationFile)
    station_df.code = station_df.code.str.strip()
    station_df.to_csv("station.dat",
                      header=None,
                      index=None,
                      sep=" ",
                      columns=["code", "lat", "lon", "elv"])


def prepareVelocity(config, vp, vs):
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
    VpVs = (vp.values.mean(axis=1).mean(axis=0) /
            vs.values.mean(axis=1).mean(axis=0)).mean()
    if config["FPS"]["VelocityModel"]["choseVelocityLayers"] == "g":
        nLayers = velocities.size
    velocities = " ".join([f"{v:5.2f}" for v in velocities])
    depths = " ".join([f"{d:5.1f}" for d in depths])
    VpVs = f"{VpVs:5.2f}"

    return velocities, depths, VpVs, nLayers


def preparePH2DT(config):
    ph2dtFile = os.path.join("ph2dt.inp")
    MINWGHT = 0
    MAXDIST = config["StudyArea"]["radius"]
    MAXSEP = 15
    MAXNGH = 10
    MINLNKS = 6
    MINOBS = 4
    MAXOBS = 99
    with open(ph2dtFile, "w") as f:
        f.write("* ph2dt.inp - input control file for program ph2dt\n")
        f.write("* Input station file:\n")
        f.write("station.dat\n")
        f.write("* Input phase file:\n")
        f.write("phase.dat\n")
        f.write("*IFORMAT: input format (0 = ph2dt-format, 1 = NCSN format)\n")
        f.write("*IPHASE: phase (1 = P; 2 = S).\n")
        f.write("*MINWGHT: min. pick weight allowed [0]\n")
        f.write(
            "*MAXDIST: max. distance in km between event pair and stations [200]\n")
        f.write("*MAXSEP: max. hypocentral separation in km [10]\n")
        f.write("*MAXNGH: max. number of neighbors per event [10]\n")
        f.write("*MINLNKS: min. number of links required to define a neighbor [8]\n")
        f.write("*MINOBS: min. number of links per pair saved [8]\n")
        f.write("*MAXOBS: max. number of links per pair saved [20]\n")
        f.write("*MINWGHT MAXDIST MAXSEP MAXNGH MINLNKS MINOBS MAXOBS\n")
        f.write(f"{MINWGHT:0.0f}      {MAXDIST:0.0f}       {MAXSEP:0.0f}      {MAXNGH:0.0f}       {MINLNKS:0.0f}      {MINOBS:0.0f}      {MAXOBS:0.0f}\n")


def prepareHypoDD(config, vp, vs):
    DIST = config["StudyArea"]["radius"]
    hypoddFile = os.path.join("hypoDD.inp")
    velocities, depths, VpVs, nLayers = prepareVelocity(config, vp, vs)
    with open(hypoddFile, "w") as f:
        f.write("* Make hypoDD.INP.\n")
        f.write("*--- input file selection\n")
        f.write("* cross correlation diff times: (not used)\n")
        f.write("\n")
        f.write("*\n")
        f.write("*catalog P & S diff times:\n")
        f.write("dt.ct\n")
        f.write("*\n")
        f.write("* event file:\n")
        f.write("event.dat\n")
        f.write("*\n")
        f.write("* station file:\n")
        f.write("station.dat\n")
        f.write("*\n")
        f.write("*--- output file selection\n")
        f.write("* original locations:\n")
        f.write("hypoDD.loc\n")
        f.write("* relocations:\n")
        f.write("hypoDD.reloc\n")
        f.write("* station information:\n")
        f.write("hypoDD.sta\n")
        f.write("* residual information:\n")
        f.write("*hypoDD.res\n")
        f.write("\n")
        f.write("* source paramater information:\n")
        f.write("*hypoDD.src\n")
        f.write("\n")
        f.write("*\n")
        f.write("*--- data type selection: \n")
        f.write("* IDAT:  0 = synthetics; 1= cross corr; 2= catalog; 3= cross & cat \n")
        f.write("* IPHA: 1= P; 2= S; 3= P&S\n")
        f.write("* DIST:max dist (km) between cluster centroid and station \n")
        f.write("* IDAT   IPHA   DIST\n")
        f.write(f"    2     3     {DIST:0.0f}\n")
        f.write("*\n")
        f.write("*--- event clustering:\n")
        f.write("* OBSCC:    min # of obs/pair for crosstime data (0= no clustering)\n")
        f.write("* OBSCT:    min # of obs/pair for network data (0= no clustering)\n")
        f.write("* OBSCC  OBSCT    \n")
        f.write("     0     6      \n")
        f.write("*\n")
        f.write("*--- solution control:\n")
        f.write("* ISTART:       1 = from single source; 2 = from network sources\n")
        f.write("* ISOLV:        1 = SVD, 2=lsqr\n")
        f.write("* NSET:         number of sets of iteration with specifications following\n")
        f.write("*  ISTART  ISOLV  NSET\n")
        f.write("    2        2      4 \n")
        f.write("*\n")
        f.write("*--- data weighting and re-weighting: \n")
        f.write("* NITER:                last iteration to use the following weights\n")
        f.write("* WTCCP, WTCCS:         weight cross P, S \n")
        f.write("* WTCTP, WTCTS:         weight catalog P, S \n")
        f.write("* WRCC, WRCT:           residual threshold in sec for cross, catalog data \n")
        f.write("* WDCC, WDCT:           max dist (km) between cross, catalog linked pairs\n")
        f.write("* DAMP:                 damping (for lsqr only) \n")
        f.write("*       ---  CROSS DATA ----- ----CATALOG DATA ----\n")
        f.write("* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP\n")
        f.write("  5      -9     -9   -9   -9   1.0   1.0  -9    -9   55\n")
        f.write("  5      -9     -9   -9   -9   1.0   0.8   10   20   50\n")
        f.write("  5      -9     -9   -9   -9   1.0   0.8   9    15   45\n")
        f.write("  5      -9     -9   -9   -9   1.0   0.8   8    10   40\n")
        f.write("*\n")
        f.write("*--- 1D model:\n")
        f.write("* NLAY:         number of model layers  \n")
        f.write("* RATIO:        vp/vs ratio \n")
        f.write("* TOP:          depths of top of layer (km) \n")
        f.write("* VEL:          layer velocities (km/s)\n")
        f.write("* NLAY  RATIO \n")
        f.write(f"  {nLayers:0.0f}     {VpVs}\n")
        f.write("*Loma Prieta model 2 (North America). Depth to top, velocity\n")
        f.write("* TOP \n")
        f.write(f"{depths}\n")
        f.write("* VEL\n")
        f.write(f"{velocities}\n")
        f.write("*\n")
        f.write("*--- event selection:\n")
        f.write("* CID:  cluster to be relocated (0 = all)\n")
        f.write("* ID:   ids of event to be relocated (8 per line)\n")
        f.write("* CID    :\n")
        f.write("    0      \n")
        f.write("* ID\n")


def prepareHypoddInputs(config,
                        catalogFile,
                        stationFile,
                        vp, vs,
                        locationPath):
    preparePhaseFile(catalogFile)
    prepareStationFile(stationFile)
    preparePH2DT(config)
    prepareHypoDD(config, vp, vs)
