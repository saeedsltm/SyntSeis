#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:13:09 2023

@author: saeed
"""

import os
import sys
from glob import glob
from pathlib import Path

from numpy import array, max, min, nan, sqrt
from obspy.geodetics.base import degrees2kilometers as d2k
from pandas import DataFrame, concat, read_csv, to_datetime
from yaml import SafeLoader, load


def readConfiguration():
    """
    Read configuration file

    Returns
    -------
    config : dict
        configuration parameters.

    """
    if not os.path.exists("config.yaml"):
        msg = "+++ Could not find configuration file! Aborting ..."
        print(msg)
        sys.exit()
    with open("config.yaml") as f:
        config = load(f, Loader=SafeLoader)
    msg = "+++ Configuration file was loaded successfully ..."
    print(msg)
    return config


def getMinMax(*inpList):
    """Get min and max of input list

    Returns:
        tuple: min and max of input list
    """
    Min = min([min(x) for x in inpList])
    Max = max([max(x) for x in inpList])
    return Min, Max


def clearRays():
    rayPath = os.path.join("results", "rays")
    Path(rayPath).mkdir(parents=True, exist_ok=True)
    for ray in glob(os.path.join(rayPath, "*")):
        os.remove(ray)


def roundTo(x, base=5):
    return base * round(x/base)


def handleNone(value, degree=False, dtype="float"):
    """Handle missing values

    Args:
        value (float): a float value
        degree (bool, optional): whether convert to degree or not.
        Defaults to False.

    Returns:
        float: handled value
    """
    if value is None:
        return nan
    else:
        if degree:
            return d2k(value)
        return int(value) if dtype == "int" else value


def getHer(event):
    """Get horizontal error of event

    Args:
        event (obspy.event): an obspy event

    Returns:
        float: event horizontal error
    """
    if event.origins[0].latitude_errors.uncertainty:
        x = event.origins[0].latitude_errors.uncertainty
        y = event.origins[0].longitude_errors.uncertainty
        return round(d2k(sqrt(x**2 + y**2)), 1)
    else:
        return None


def getZer(event):
    """Get depth error of event

    Args:
        event (obspy.event): an obspy event

    Returns:
        float: event depth error
    """
    if event.origins[0].depth_errors.uncertainty:
        return event.origins[0].depth_errors.uncertainty*0.001
    else:
        return None


def getRMS(arrivals):
    time_residuals = array([
        arrival.time_residual for arrival in arrivals if isinstance(
            arrival.time_residual, float)
    ])
    time_weights = array([
        arrival.time_weight for arrival in arrivals if isinstance(
            arrival.time_weight, float)
    ])
    if time_residuals.size:
        weighted_rms = sum(time_weights * time_residuals **
                           2) / sum(time_weights)
        weighted_rms = sqrt(weighted_rms)
    else:
        weighted_rms = nan
    return weighted_rms


def getPick(picks, requestedPickID):
    for pick in picks:
        if pick.resource_id == requestedPickID:
            return pick


def loadxyzm(*xyzmPaths):
    reports = []
    for xyzmPath in xyzmPaths:
        report = read_csv(xyzmPath, delim_whitespace=True)
        reports.append(report)
    return reports


def extractCommons(config, df_ini, df_tar):
    thr_t = config["FGS"]["CommonEventThrT"]
    thr_d = config["FGS"]["CommonEventThrD"]
    thr_m = config["FGS"]["CommonEventThrM"]
    df_com = DataFrame()
    for _, row in df_tar.iterrows():
        diff = DataFrame()
        diff_ort = to_datetime(row["ORT"]) - to_datetime(df_ini["ORT"])
        diff_ort = abs(diff_ort.dt.total_seconds())  # type: ignore
        diff_lon = row["Lon"] - df_ini["Lon"]
        diff_lat = row["Lat"] - df_ini["Lat"]
        diff_loc = d2k(sqrt(diff_lon**2 + diff_lat**2))
        diff_mag = row["Mag"] - df_tar["Mag"]
        diff_mag.fillna(0, inplace=True)
        diff["ort"] = diff_ort
        diff["loc"] = diff_loc
        diff["mag"] = diff_mag
        cond = (diff_ort <= thr_t) & (diff_loc <= thr_d) & (diff_mag <= thr_m)
        if cond.sum():
            commonEventID = diff[cond].sort_values(
                ["ort", "loc", "mag"]).index[0]  # type: ignore
            tar_keys = [k+"_tar" for k in row.keys()]
            tar_vals = [[v] for v in row.values]
            ini_keys = [k+"_ini" for k in df_ini[cond].keys()]
            ini_vals = [[v] for v in df_ini.iloc[commonEventID].values]
            keys = tar_keys + ini_keys
            vals = tar_vals + ini_vals
            result = {k: v for k, v in zip(keys, vals)}
            result = DataFrame(result)
            df_com = concat([df_com, result])
    df_com.reset_index(inplace=True)
    return df_com


def computeClass(config, report_unw, report_w, Class):
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
    neq_unw = len(report_unw)
    neq_w = len(report_w)
    ERH = config["RPS"]["Classes"][Class]["ERH"]
    ERZ = config["RPS"]["Classes"][Class]["ERZ"]
    GAP = config["RPS"]["Classes"][Class]["GAP"]
    RMS = config["RPS"]["Classes"][Class]["RMS"]
    MDS = config["RPS"]["Classes"][Class]["MDS"]
    NuP = config["RPS"]["Classes"][Class]["NuP"]
    NuS = config["RPS"]["Classes"][Class]["NuS"]
    c_unw_h, c_w_h = report_unw["ERH"] <= ERH, report_w["ERH"] <= ERH
    c_unw_z, c_w_z = report_unw["ERZ"] <= ERZ, report_w["ERZ"] <= ERZ
    c_unw_g, c_w_g = report_unw["GAP"] <= GAP, report_w["GAP"] <= GAP
    c_unw_r, c_w_r = report_unw["RMS"] <= RMS, report_w["RMS"] <= RMS
    c_unw_m, c_w_m = report_unw["MDS"] <= MDS, report_w["MDS"] <= MDS
    c_unw_p, c_w_p = report_unw["NuP"] >= NuP, report_w["NuP"] >= NuP
    c_unw_s, c_w_s = report_unw["NuS"] >= NuS, report_w["NuS"] >= NuS
    report_unw_filtered = report_unw[(c_unw_h) & (c_unw_z) & (c_unw_g) & (
        c_unw_r) & (c_unw_m) & (c_unw_p) & (c_unw_s)]
    report_w_filtered = report_w[(c_w_h) & (c_w_z) & (c_w_g) & (
        c_w_r) & (c_w_m) & (c_w_p) & (c_w_s)]
    return (len(report_unw_filtered),
            len(report_unw_filtered)/neq_unw*1e2,
            len(report_w_filtered),
            len(report_w_filtered)/neq_w*1e2)
