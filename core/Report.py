import os

from numpy import abs, mean, sqrt
from obspy.geodetics.base import degrees2kilometers as d2k
from obspy.geodetics.base import gps2dist_azimuth as gps
from pandas import Series, read_csv

from core.Extra import computeClass, extractCommons, loadxyzm


def prepareReport(config, locator):
    print(f"+++ Preparing reports for {locator} ...")

    reportFile = os.path.join("results", f"{locator}_report.dat")

    catalog_ini_path = os.path.join(
        "results", "location", locator, "xyzm_initial.dat")
    catalog_unw_path = os.path.join(
        "results", "location", locator, "xyzm_unw.dat")
    catalog_w_path = os.path.join(
        "results", "location", locator, "xyzm_w.dat")
    report_ini, report_unw, report_w = loadxyzm(catalog_ini_path,
                                                catalog_unw_path,
                                                catalog_w_path)

    neq_ini = len(report_ini) - report_ini.Lon.isna().sum()
    neq_unw = len(report_unw) - report_unw.Lon.isna().sum()
    neq_w = len(report_w) - report_w.Lon.isna().sum()

    df_ini_unw = extractCommons(config, report_ini, report_unw)
    df_ini_w = extractCommons(config, report_ini, report_w)
    df_unw_w = extractCommons(config, report_unw, report_w)
    df_station = read_csv(os.path.join("inputs", "stations.csv"))

    classes = {}
    for Class, _ in config["RPS"]["Classes"].items():
        classes[Class] = computeClass(config, report_unw, report_w, Class)

    for df in [df_ini_unw, df_ini_w, df_unw_w]:
        df["dh"] = df.apply(lambda x: Series(
            gps(x.Lat_ini, x.Lon_ini, x.Lat_tar, x.Lon_tar)[0]*1e-3), axis=1)
        df["dz"] = df.apply(lambda x: Series(
            abs(x.Dep_ini - x.Dep_tar)), axis=1)

    df_ini_unw["dh"] = df_ini_unw.apply(lambda x: Series(
        gps(x.Lat_ini, x.Lon_ini, x.Lat_tar, x.Lon_tar)[0]*1e-3), axis=1)
    df_ini_unw["dz"] = df_ini_unw.apply(
        lambda x: Series(abs(x.Dep_ini - x.Dep_tar)), axis=1)

    df_ini_unw["dh"] = df_ini_unw.apply(lambda x: Series(
        gps(x.Lat_ini, x.Lon_ini, x.Lat_tar, x.Lon_tar)[0]*1e-3), axis=1)
    df_ini_unw["dz"] = df_ini_unw.apply(
        lambda x: Series(abs(x.Dep_ini - x.Dep_tar)), axis=1)

    # --- unw
    # erh < 2.0km, 5.0km
    erh_2km_unw_n = (df_unw_w.ERH_ini < 2.0).sum()
    erh_2km_unw_p = (erh_2km_unw_n / len(df_unw_w)) * 1e2
    erh_5km_unw_n = (df_unw_w.ERH_ini < 5.0).sum()
    erh_5km_unw_p = (erh_5km_unw_n / len(df_unw_w)) * 1e2
    # --- real
    erh_2km_unw_real_n = (df_ini_unw.dh < 2.0).sum()
    erh_2km_unw_real_p = (erh_2km_unw_real_n / len(df_ini_unw)) * 1e2
    erh_5km_unw_real_n = (df_ini_unw.dh < 5.0).sum()
    erh_5km_unw_real_p = (erh_5km_unw_real_n / len(df_ini_unw)) * 1e2
    # erz < 2.0km, 5.0km
    erz_2km_unw_n = (df_unw_w.ERZ_ini < 2.0).sum()
    erz_2km_unw_p = (erz_2km_unw_n / len(df_unw_w)) * 1e2
    erz_5km_unw_n = (df_unw_w.ERZ_ini < 5.0).sum()
    erz_5km_unw_p = (erz_5km_unw_n / len(df_unw_w)) * 1e2
    # --- real
    erz_2km_unw_real_n = (df_ini_unw.dz < 2.0).sum()
    erz_2km_unw_real_p = (erz_2km_unw_real_n / len(df_ini_unw)) * 1e2
    erz_5km_unw_real_n = (df_ini_unw.dz < 5.0).sum()
    erz_5km_unw_real_p = (erz_5km_unw_real_n / len(df_ini_unw)) * 1e2
    # rms < 0.2s, 0.5s
    rms_2s_unw_n = (df_unw_w.RMS_ini < 0.2).sum()
    rms_2s_unw_p = (rms_2s_unw_n / len(df_unw_w)) * 1e2
    rms_5s_unw_n = (df_unw_w.RMS_ini < 0.5).sum()
    rms_5s_unw_p = (rms_5s_unw_n / len(df_unw_w)) * 1e2
    # gap < 150°, 250°
    gap_150d_unw_n = (df_unw_w.GAP_ini < 150).sum()
    gap_150d_unw_p = (gap_150d_unw_n / len(df_unw_w)) * 1e2
    gap_250d_unw_n = (df_unw_w.GAP_ini < 250).sum()
    gap_250d_unw_p = (gap_250d_unw_n / len(df_unw_w)) * 1e2
    # mds < 10km, 30km
    mds_10km_unw_n = (df_unw_w.MDS_ini < 10).sum()
    mds_10km_unw_p = (mds_10km_unw_n / len(df_unw_w)) * 1e2
    mds_30km_unw_n = (df_unw_w.MDS_ini < 30).sum()
    mds_30km_unw_p = (mds_30km_unw_n / len(df_unw_w)) * 1e2

    # --- w
    # erh < 2.0km, 5.0km
    erh_2km_w_n = (df_unw_w.ERH_tar < 2.0).sum()
    erh_2km_w_p = (erh_2km_w_n / len(df_unw_w)) * 1e2
    erh_5km_w_n = (df_unw_w.ERH_tar < 5.0).sum()
    erh_5km_w_p = (erh_5km_w_n / len(df_unw_w)) * 1e2
    # --- real
    erh_2km_w_real_n = (df_ini_w.dh < 2.0).sum()
    erh_2km_w_real_p = (erh_2km_w_real_n / len(df_ini_w)) * 1e2
    erh_5km_w_real_n = (df_ini_w.dh < 5.0).sum()
    erh_5km_w_real_p = (erh_5km_w_real_n / len(df_ini_w)) * 1e2
    # erz < 2.0km, 5.0km
    erz_2km_w_n = (df_unw_w.ERZ_tar < 2.0).sum()
    erz_2km_w_p = (erz_2km_w_n / len(df_unw_w)) * 1e2
    erz_5km_w_n = (df_unw_w.ERZ_tar < 5.0).sum()
    erz_5km_w_p = (erz_5km_w_n / len(df_unw_w)) * 1e2
    # --- real
    erz_2km_w_real_n = (df_ini_w.dz < 2.0).sum()
    erz_2km_w_real_p = (erz_2km_w_real_n / len(df_ini_w)) * 1e2
    erz_5km_w_real_n = (df_ini_w.dz < 5.0).sum()
    erz_5km_w_real_p = (erz_5km_w_real_n / len(df_ini_w)) * 1e2
    # rms < 0.2s, 0.5s
    rms_2s_w_n = (df_unw_w.RMS_tar < 0.2).sum()
    rms_2s_w_p = (rms_2s_w_n / len(df_unw_w)) * 1e2
    rms_5s_w_n = (df_unw_w.RMS_tar < 0.5).sum()
    rms_5s_w_p = (rms_5s_w_n / len(df_unw_w)) * 1e2
    # gap < 150°, 250°
    gap_150d_w_n = (df_unw_w.GAP_tar < 150).sum()
    gap_150d_w_p = (gap_150d_w_n / len(df_unw_w)) * 1e2
    gap_250d_w_n = (df_unw_w.GAP_tar < 250).sum()
    gap_250d_w_p = (gap_250d_w_n / len(df_unw_w)) * 1e2
    # mds < 10km, 30km
    mds_10km_w_n = (df_unw_w.MDS_tar < 10).sum()
    mds_10km_w_p = (mds_10km_w_n / len(df_unw_w)) * 1e2
    mds_30km_w_n = (df_unw_w.MDS_tar < 30).sum()
    mds_30km_w_p = (mds_30km_w_n / len(df_unw_w)) * 1e2

    mDisStations = mean(df_station.apply(lambda x: Series(
        d2k(sqrt((x.lon-df_station.lon)**2 +
                 (x.lat-df_station.lat)**2))), axis=1).mean())
    mDisEvents_ini = mean(report_ini.apply(lambda x: Series(
        d2k(sqrt((x.Lon-report_ini.Lon)**2 +
                 (x.Lat-report_ini.Lat)**2))), axis=1).mean())
    mDisEvents_unw = mean(report_unw.apply(lambda x: Series(
        d2k(sqrt((x.Lon-report_unw.Lon)**2 +
                 (x.Lat-report_unw.Lat)**2))), axis=1).mean())
    mDisEvents_w = mean(report_w.apply(lambda x: Series(
        d2k(sqrt((x.Lon-report_w.Lon)**2 +
                 (x.Lat-report_w.Lat)**2))), axis=1).mean())

    with open(reportFile, "w") as f:
        f.write("Summary:\n")
        f.write(f"Number of events in initial catalog    : {neq_ini:0.0f}\n")
        f.write(f"Number of events in Unweighted catalog : {neq_unw:0.0f}\n")
        f.write(f"Number of events in weighted catalog   : {neq_w:0.0f}\n")
        f.write(
            f"Number of common events between initial and Unweighted catalog : {len(df_ini_unw):0.0f}\n")
        f.write(
            f"Number of common events between initial and weighted catalog   : {len(df_ini_w):0.0f}\n")
        f.write(
            f"Mean distance between stations         : {mDisStations:5.1f}km\n")
        f.write(
            f"Mean distance between events in initial catalog    : {mDisEvents_ini:5.1f}km\n")
        f.write(
            f"Mean distance between events in Unweighted catalog : {mDisEvents_unw:5.1f}km\n")
        f.write(
            f"Mean distance between events in weighted catalog   : {mDisEvents_w:5.1f}km\n")
        f.write("#-----No. events with-----#---------------Unweighted---------------#----------------Weighted---------------#\n")
        # erh < 2.0km
        p1 = f"{erh_2km_unw_n:4.0f}({erh_2km_unw_p:5.1f}%)"
        p2 = f"{erh_2km_unw_real_n:4.0f}({erh_2km_unw_real_p:5.1f}%)"
        p3 = f"{erh_2km_w_n:4.0f}({erh_2km_w_p:5.1f}%)"
        p4 = f"{erh_2km_w_real_n:4.0f}({erh_2km_w_real_p:5.1f}%)\n"
        f.write(
            f"          ERH<2.0km       : {p1}, but real is: {p2}, {p3}, but real is: {p4}")
        # erh < 5.0km
        p1 = f"{erh_5km_unw_n:4.0f}({erh_5km_unw_p:5.1f}%)"
        p2 = f"{erh_5km_unw_real_n:4.0f}({erh_5km_unw_real_p:5.1f}%)"
        p3 = f"{erh_5km_w_n:4.0f}({erh_5km_w_p:5.1f}%)"
        p4 = f"{erh_5km_w_real_n:4.0f}({erh_5km_w_real_p:5.1f}%)\n"
        f.write(
            f"          ERH<5.0km       : {p1}, but real is: {p2}, {p3}, but real is: {p4}")
        # erz < 2.0km
        p1 = f"{erz_2km_unw_n:4.0f}({erz_2km_unw_p:5.1f}%)"
        p2 = f"{erz_2km_unw_real_n:4.0f}({erz_2km_unw_real_p:5.1f}%)"
        p3 = f"{erz_2km_w_n:4.0f}({erz_2km_w_p:5.1f}%)"
        p4 = f"{erz_2km_w_real_n:4.0f}({erz_2km_w_real_p:5.1f}%)\n"
        f.write(
            f"          ERZ<2.0km       : {p1}, but real is: {p2}, {p3}, but real is: {p4}")
        # erz < 5.0km
        p1 = f"{erz_5km_unw_n:4.0f}({erz_5km_unw_p:5.1f}%)"
        p2 = f"{erz_5km_unw_real_n:4.0f}({erz_5km_unw_real_p:5.1f}%)"
        p3 = f"{erz_5km_w_n:4.0f}({erz_5km_w_p:5.1f}%)"
        p4 = f"{erz_5km_w_real_n:4.0f}({erz_5km_w_real_p:5.1f}%)\n"
        f.write(
            f"          ERZ<5.0km       : {p1}, but real is: {p2}, {p3}, but real is: {p4}")
        # rms < 0.2s
        p1 = f"{rms_2s_unw_n:4.0f}({rms_2s_unw_p:5.1f}%)"
        p3 = f"{rms_2s_w_n:4.0f}({rms_2s_w_p:5.1f}%)\n"
        f.write(
            f"          RMS<0.2s        : {p1},                            {p3}")
        # rms < 0.5s
        p1 = f"{rms_5s_unw_n:4.0f}({rms_5s_unw_p:5.1f}%)"
        p3 = f"{rms_5s_w_n:4.0f}({rms_5s_w_p:5.1f}%)\n"
        f.write(
            f"          RMS<0.5s        : {p1},                            {p3}")
        # gap < 150°
        p1 = f"{gap_150d_unw_n:4.0f}({gap_150d_unw_p:5.1f}%)"
        p3 = f"{gap_150d_w_n:4.0f}({gap_150d_w_p:5.1f}%)\n"
        f.write(
            f"          GAP<150°        : {p1},                            {p3}")
        # gap < 250°
        p1 = f"{gap_250d_unw_n:4.0f}({gap_250d_unw_p:5.1f}%)"
        p3 = f"{gap_250d_w_n:4.0f}({gap_250d_w_p:5.1f}%)\n"
        f.write(
            f"          GAP<250°        : {p1},                            {p3}")
        # mds < 10km
        p1 = f"{mds_10km_unw_n:4.0f}({mds_10km_unw_p:5.1f}%)"
        p3 = f"{mds_10km_w_n:4.0f}({mds_10km_w_p:5.1f}%)\n"
        f.write(
            f"          MDS<10km        : {p1},                            {p3}")
        # mds < 30km
        p1 = f"{mds_30km_unw_n:4.0f}({mds_30km_unw_p:5.1f}%)"
        p3 = f"{mds_30km_w_n:4.0f}({mds_30km_w_p:5.1f}%)\n"
        f.write(
            f"          MDS<30km        : {p1},                            {p3}")

        f.write("#---------Classes---------#---------------Unweighted---------------#----------------Weighted---------------#\n")
        for k, v in classes.items():
            eq_unw_n, eq_unw_p, eq_w_n, eq_w_p = v
            f.write(
                f"  No. events in class {k}:    {eq_unw_n:4.0f}({eq_unw_p:5.1f}%)                            {eq_w_n:4.0f}({eq_w_p:5.1f}%)\n")
