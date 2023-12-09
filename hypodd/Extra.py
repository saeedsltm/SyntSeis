from pandas import read_csv, to_datetime
from numpy import sqrt
from obspy import read_events
from obspy import UTCDateTime as utc
from obspy.geodetics.base import kilometers2degrees as k2d
from obspy.core.event import Catalog
from tqdm import tqdm


def loadHypoDDRelocFile():
    names = ["ID",  "LAT",  "LON",  "DEPTH",
             "X",  "Y",  "Z",
             "EX",  "EY",  "EZ",
             "YR",  "MO",  "DY",  "HR",  "MI",  "SC",
             "MAG",
             "NCCP",  "NCCS",
             "NCTP",  "NCTS",
             "RCC",  "RCT",
             "CID "]
    hypodd_df = read_csv("hypoDD.reloc", delim_whitespace=True, names=names)
    return hypodd_df


def writexyzm(outName):
    hypodd_df = loadHypoDDRelocFile()
    outputFile = f"xyzm_{outName}.dat"
    hypodd_df["year"] = hypodd_df.YR
    hypodd_df["month"] = hypodd_df.MO
    hypodd_df["day"] = hypodd_df.DY
    hypodd_df["hour"] = hypodd_df.HR
    hypodd_df["minute"] = hypodd_df.MI
    hypodd_df["second"] = hypodd_df.SC
    hypodd_df["ORT"] = to_datetime(hypodd_df[["year",
                                              "month",
                                              "day",
                                              "hour",
                                              "minute",
                                              "second"]])
    hypodd_df["ORT"] = hypodd_df["ORT"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    hypodd_df["Lon"] = hypodd_df.LON
    hypodd_df["Lat"] = hypodd_df.LAT
    hypodd_df["Dep"] = hypodd_df.DEPTH
    hypodd_df["Mag"] = hypodd_df.MAG
    hypodd_df["Nus"] = hypodd_df.NCTP
    hypodd_df["NuP"] = hypodd_df.NCTP
    hypodd_df["NuS"] = hypodd_df.NCTS
    hypodd_df["ADS"] = 0
    hypodd_df["MDS"] = 0
    hypodd_df["GAP"] = 0
    hypodd_df["RMS"] = hypodd_df.RCT
    hypodd_df["ERH"] = sqrt(hypodd_df.EX**2 + hypodd_df.EY**2)*1e-3
    hypodd_df["ERZ"] = hypodd_df.EZ*1e-3
    columns = ["ORT", "Lon", "Lat", "Dep", "Mag",
               "Nus", "NuP", "NuS", "ADS", "MDS", "GAP", "RMS", "ERH", "ERZ"]
    with open(outputFile, "w") as f:
        hypodd_df.to_string(f, columns=columns, index=False, float_format="%7.3f")


def hypoDD2nordic(outName):
    hypoddInp = read_events("phase.dat")
    hypoddCat = loadHypoDDRelocFile()
    outCatalog = Catalog()
    mapW = {
        0.00: 4,
        0.25: 3,
        0.50: 2,
        0.75: 1,
        1.00: 0
    }
    desc = f"+++ Converting hypoDD to NORDIC for {outName} ..."
    for event in tqdm(hypoddInp, desc=desc, unit=" event"):
        preferred_origin = event.preferred_origin()
        eId = int(preferred_origin.resource_id.id.split("/")[-1])
        if eId in hypoddCat.ID.values:
            indx = hypoddCat.ID[hypoddCat.ID == eId].index.values[0]
            row = hypoddCat.iloc[indx].copy()
            if row.SC == 60:
                row.SC = 59.99
            eOrt = utc(int(row.YR), int(row.MO), int(row.DY),
                       int(row.HR), int(row.MI), row.SC)
            eLat = row.LAT
            erLat = row.EY
            eLon = row.LON
            erLon = row.EX
            eDep = row.DEPTH
            erDep = row.EZ
            preferred_origin.time = eOrt
            preferred_origin.latitude = eLat
            preferred_origin.longitude = eLon
            preferred_origin.depth = eDep*1e3
            preferred_origin.latitude_errors.uncertainty = k2d(erLat)
            preferred_origin.longitude_errors.uncertainty = k2d(erLon)
            preferred_origin.depth_errors.uncertainty = erDep
            arrivals_id = {
                arrival.pick_id: arrival.time_weight for arrival in preferred_origin.arrivals}
            for pick in event.picks:
                pick_id = pick.resource_id
                if pick_id in arrivals_id:
                    w = arrivals_id[pick_id]
                    pick.update({"extra": {"nordic_pick_weight": {"value": 0}}})
                    pick.extra.nordic_pick_weight.value = mapW[w]
            outCatalog.append(event)
    outCatalog.write(f"hypodd_{outName}.out", format="nordic", high_accuracy=False)
