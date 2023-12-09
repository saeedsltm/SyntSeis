import os
from statistics import mean

from numpy import sqrt
from obspy.geodetics.base import degrees2kilometers as d2k
from pandas import Series, read_csv

from core.Extra import loadxyzm


def countNP(catalog, catalog_metrics, key, column):
    n = catalog[catalog_metrics[key]][column].size
    p = n/catalog[column].size*100.0
    return n, p


def computeClass(config, report_unw, report_w, c):
    """Compute class statistics

    Args:
        report_unw (data frame): unweighted data frame of events
        report_w (data frame): weighted data frame of events
        c (str): defined class
        config (dict): a dictionary contains main configuration

    Returns:
        tuple: number and percentage of classes in unweighted and
        weighted catalog.
    """
    Neq_unw = report_unw["ORT"].size
    Neq_w = report_w["ORT"].size
    ERH = config["RPS"]["Classes"][c]["ERH"]
    ERZ = config["RPS"]["Classes"][c]["ERZ"]
    GAP = config["RPS"]["Classes"][c]["GAP"]
    RMS = config["RPS"]["Classes"][c]["RMS"]
    MDS = config["RPS"]["Classes"][c]["MDS"]
    NuP = config["RPS"]["Classes"][c]["NuP"]
    NuS = config["RPS"]["Classes"][c]["NuS"]
    c_unw_h, c_w_h = report_unw["ERH"] <= ERH, report_w["ERH"] <= ERH
    c_unw_z, c_w_z = report_unw["ERZ"] <= ERZ, report_w["ERZ"] <= ERZ
    c_unw_g, c_w_g = report_unw["GAP"] <= GAP, report_w["GAP"] <= GAP
    c_unw_r, c_w_r = report_unw["RMS"] <= RMS, report_w["RMS"] <= RMS
    c_unw_m, c_w_m = report_unw["MDS"] <= MDS, report_w["MDS"] <= MDS
    c_unw_p, c_w_p = report_unw["NuP"] >= NuP, report_w["NuP"] >= NuP
    c_unw_s, c_w_s = report_unw["NuS"] >= NuS, report_w["NuS"] >= NuS
    report_unw = report_unw[(c_unw_h) & (c_unw_z) & (c_unw_g) & (
        c_unw_r) & (c_unw_m) & (c_unw_p) & (c_unw_s)]
    report_w = report_w[(c_w_h) & (c_w_z) & (c_w_g) & (
        c_w_r) & (c_w_m) & (c_w_p) & (c_w_s)]
    return (report_unw["ORT"].size,
            report_unw["ORT"].size/Neq_unw*100.0,
            report_w["ORT"].size,
            report_w["ORT"].size/Neq_w*100.0)


def summarize(config, locator):
    print(f"+++ Making reports for '{locator}' ...")
    stationPath = os.path.join("inputs", "stations.csv")
    station_df = read_csv(stationPath)
    catalog_ini_path = os.path.join(
        "results", "location", locator, "xyzm_initial.dat")
    catalog_unw_path = os.path.join(
        "results", "location", locator, "xyzm_unw.dat")
    catalog_w_path = os.path.join(
        "results", "location", locator, "xyzm_w.dat")
    report_ini, report_unw, report_w = loadxyzm(catalog_ini_path,
                                                catalog_unw_path,
                                                catalog_w_path)
    metrics_report_unw = {
        "ERH_2km": 'report_unw["ERH"]<=2.0',
        "ERH_5km": 'report_unw["ERH"]<=5.0',
        "ERZ_2km": 'report_unw["ERZ"]<=2.0',
        "ERZ_5km": 'report_unw["ERZ"]<=5.0',
        "RMS_0_1s": 'report_unw["RMS"]<=0.1',
        "RMS_0_3s": 'report_unw["RMS"]<=0.3',
        "GAP_150d": 'report_unw["GAP"]<=150',
        "GAP_200d": 'report_unw["GAP"]<=200',
        "GAP_250d": 'report_unw["GAP"]<=250',
        "MDS_5km": 'report_unw["MDS"]<=5',
        "MDS_10km": 'report_unw["MDS"]<=10',
        "MDS_15km": 'report_unw["MDS"]<=15',
        "Nus_n": 'report_unw["Nus"]>=5',
        "NuP_n": 'report_unw["NuP"]>=5',
        "NuS_n": 'report_unw["NuS"]>=5'
    }
    metrics_report_w = {
        "ERH_2km": 'report_w["ERH"]<=2.0',
        "ERH_5km": 'report_w["ERH"]<=5.0',
        "ERZ_2km": 'report_w["ERZ"]<=2.0',
        "ERZ_5km": 'report_w["ERZ"]<=5.0',
        "RMS_0_1s": 'report_w["RMS"]<=0.1',
        "RMS_0_3s": 'report_w["RMS"]<=0.3',
        "GAP_150d": 'report_w["GAP"]<=150',
        "GAP_200d": 'report_w["GAP"]<=200',
        "GAP_250d": 'report_w["GAP"]<=250',
        "MDS_5km": 'report_w["MDS"]<=5',
        "MDS_10km": 'report_w["MDS"]<=10',
        "MDS_15km": 'report_w["MDS"]<=15',
        "Nus_n": 'report_w["Nus"]>=5',
        "NuP_n": 'report_w["NuP"]>=5',
        "NuS_n": 'report_w["NuS"]>=5'
    }
    for metric in metrics_report_unw.keys():
        metrics_report_unw[metric] = eval(metrics_report_unw[metric])
    for metric in metrics_report_w.keys():
        metrics_report_w[metric] = eval(metrics_report_w[metric])

    nSta = station_df["lat"].size
    nEvt = report_unw["Lat"].size
    mDistEvt = mean(report_unw.apply(lambda x: mean(
        Series(sqrt((x.Lon-report_unw.Lon)**2 + (x.Lat-report_unw.Lat)**2))), axis=1))  # type: ignore
    mDistSta = mean(station_df.apply(lambda x: mean(
        Series(sqrt((x.lon-station_df.lon)**2 + (x.lat-station_df.lat)**2))), axis=1))  # type: ignore
    mDistEvt, mDistSta = d2k(mDistEvt), d2k(mDistSta)
    mDistEvtSta = mean(report_unw["ADS"])

    with open(os.path.join("results", f"summary_{locator}.dat"), "w") as f:
        f.write("----Problem status:\n")
        f.write("Number of events: {nEvt:.0f}\n".format(nEvt=nEvt))
        f.write("Number of stations: {nSta:.0f}\n".format(nSta=nSta))
        f.write("Average distance between events (km): {mDistEvt:.2f}\n".format(
            mDistEvt=mDistEvt))
        f.write("Average distance between stations (km): {mDistSta:.2f}\n".format(
            mDistSta=mDistSta))
        f.write(
            "Average distance between events-stations pair (km): {mDistEvtSta:.1f}\n".format(mDistEvtSta=mDistEvtSta))

        f.write("----Statistics:\n")
        f.write("".center(50, "="))
        f.write("| Events relocated without weighting scheme |".center(40, "="))
        f.write("| Events relocated using weighting scheme |".center(40, "="))
        f.write("\n")

        f.write("+++ Number(%) of event with ERH<2.0km:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "ERH_2km", "ERH")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "ERH_2km", "ERH")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with ERH<5.0km:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "ERH_5km", "ERH")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "ERH_5km", "ERH")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with ERZ<2.0km:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "ERZ_2km", "ERZ")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "ERZ_2km", "ERZ")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with ERZ<5.0km:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "ERZ_5km", "ERZ")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "ERZ_5km", "ERZ")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with RMS<0.1s:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "RMS_0_1s", "RMS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "RMS_0_1s", "RMS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with RMS<0.3s:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "RMS_0_3s", "RMS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "RMS_0_3s", "RMS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with GAP<150:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "GAP_150d", "GAP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "GAP_150d", "GAP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with GAP<200:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "GAP_200d", "GAP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "GAP_200d", "GAP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with GAP<250:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "GAP_250d", "GAP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "GAP_250d", "GAP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with MDS<5:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "MDS_5km", "MDS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "MDS_5km", "MDS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with MDS<10:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "MDS_10km", "MDS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "MDS_10km", "MDS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with MDS<15:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "MDS_15km", "MDS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "MDS_15km", "MDS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with Nus>5:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "Nus_n", "Nus")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "Nus_n", "Nus")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with NuP>5:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "NuP_n", "NuP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "NuP_n", "NuP")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        f.write("+++ Number(%) of event with NuS>5:".ljust(50, " "))
        n, p = countNP(report_unw, metrics_report_unw, "NuS_n", "NuS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        n, p = countNP(report_w, metrics_report_w, "NuS_n", "NuS")
        f.write("{n:.0f}({p:.1f})".format(n=n, p=p).rjust(40, " "))
        f.write("\n")

        # classes
        for c in config["RPS"]["Classes"].keys():
            f.write(" Class {c:s} ".format(c=c).center(50, "="))
            f.write("| Events relocated without weighting scheme |".center(40, "="))
            f.write("| Events relocated using weighting scheme |".center(40, "="))
            f.write("\n")
            (NoEqInClass_unw,
             PerEqInClass_unw,
             NoEqInClass_w,
             PerEqInClass_w) = computeClass(config, report_unw, report_w, c)
            f.write("+++ Number(%) of event:".ljust(50, " "))
            f.write("{n:.0f}({p:.1f})".format(
                n=NoEqInClass_unw, p=PerEqInClass_unw).rjust(40, " "))
            f.write("{n:.0f}({p:.1f})".format(
                n=NoEqInClass_w, p=PerEqInClass_w).rjust(40, " "))
            f.write("\n")
