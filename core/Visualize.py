#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:44:11 2023

@author: saeed
"""

import os
from pathlib import Path

import proplot as plt
import pykonal
from matplotlib.patches import ConnectionPatch
from numpy import (arange, array, cos, deg2rad, finfo, histogram, isnan, mean,
                   sin)
from obspy.geodetics.base import degrees2kilometers as d2k
from obspy.geodetics.base import locations2degrees as l2d
from pandas import DataFrame, Series, merge, read_csv, to_numeric
from pyproj import Proj

from core.Extra import extractCommons, getMinMax, loadxyzm
from core.Station import loadStationNoiseModel

Path("results").mkdir(parents=True, exist_ok=True)


def plotVelocityModel2D(config):
    print("+++ Plotting 2D velocity model ...")
    velocity = pykonal.fields.read_hdf(os.path.join("model", "vp.mod"))
    anomalyID = config["Model"]["showAnomalyID"]
    anomaliesPos = config["Model"]["anomaliesPos"]
    slice_x = slice_y = 0

    plt.close("all")
    axShape = array(
        [[1],
         [2]]
    )
    if anomalyID != 0:
        slice_y = mean(anomaliesPos[anomalyID-1][2:4], dtype=int)
        slice_x = mean(anomaliesPos[anomalyID-1][0:2], dtype=int)
    qmesh_x = (velocity.nodes[:, slice_y, :, 0],
               velocity.nodes[:, slice_y, :, 2],
               velocity.values[:, slice_y, :])
    qmesh_y = (velocity.nodes[slice_x, :, :, 1],
               velocity.nodes[slice_x, :, :, 2],
               velocity.values[slice_x, :, :])
    fig, axs = plt.subplots(axShape, sharey=True, sharex=False)
    qmesh = None
    for ax, data, label in zip(axs, [qmesh_x, qmesh_y], ["x", "y"]):
        ax.format(
            xlabel=f"{label} (km)",
            ylabel="Depth (km)",
            labelsize=7,
            ticklabelsize=7
        )
        qmesh = ax.pcolormesh(
            *data,
            cmap="Spectral",
            discrete=False,
            vmin=config["Model"]["vpLimits"][0],
            vmax=config["Model"]["vpLimits"][1],
            labels_kw={"fontsize": 5},
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        if anomalyID != 0 and label == "x":
            ax.axvline(x=velocity.nodes[slice_x, 0,
                       0, 0], color="k", linestyle="--", lw=1)
    cbar = fig.colorbar(
        qmesh,
        labelsize=5,
        ticklabelsize=5,
        width=0.1,
    )
    cbar.set_label("Velocity (km/s)")
    cbar.ax.invert_yaxis()

    if anomalyID != 0:
        xyA = (velocity.nodes[slice_x, 0, 0, 0],
               velocity.nodes[slice_x, 0, -1, -1])
        xyB = (velocity.nodes[0, 0, 0, 1], velocity.nodes[0, 0, 0, -1])
        coordsA = "data"
        coordsB = "data"
        con = ConnectionPatch(xyA=xyA, xyB=xyB,
                              coordsA=coordsA, coordsB=coordsB,
                              axesA=axs[0], axesB=axs[1], shrinkB=1, ls=":",
                              in_layout=False)
        axs[0].add_artist(con)
        xyA = (velocity.nodes[slice_x, 0, 0, 0],
               velocity.nodes[slice_x, 0, -1, -1])
        xyB = (velocity.nodes[0, -1, 0, 1], velocity.nodes[0, -1, 0, -1])
        coordsA = "data"
        coordsB = "data"
        con = ConnectionPatch(xyA=xyA, xyB=xyB,
                              coordsA=coordsA, coordsB=coordsB,
                              axesA=axs[0], axesB=axs[1], shrinkB=1, ls=":",
                              in_layout=False)
        axs[0].add_artist(con)
    fig.save(os.path.join("results", f"velocityModel2D_{anomalyID}.png")) # type: ignore

def plotVelocityModel3D(config):
    print("+++ Plotting 3D velocity model ...")
    velocity = pykonal.fields.read_hdf(os.path.join("model", "vp.mod"))
    PCF = 4
    nodes = velocity.nodes
    plt.close("all")
    fig, axs = plt.subplots(proj="three")
    [ax.grid(ls=":") for ax in axs]
    ax = axs[0]
    ax.format(
        labelsize=7,
        ticklabelsize=7,
    )
    pts = ax.scatter(
        nodes[::PCF, ::PCF, ::PCF, 0],
        nodes[::PCF, ::PCF, ::PCF, 1],
        nodes[::PCF, ::PCF, ::PCF, 2],
        c=velocity.values[::PCF, ::PCF, ::PCF],
        cmap="Spectral",
        s=4,
    )
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_zlabel("Depth (km)")
    cbar = ax.colorbar(
        pts,
        labelsize=7,
        ticklabelsize=6,
    )
    cbar.set_label("Velocity (km/s)")
    ax.invert_zaxis()
    fig.save(os.path.join("results", "velocityModel3D.png")) # type: ignore

def plotRays(config, sources, velocity, raysBank, rayType):
    plt.close("all")
    fig, axs = plt.subplots()
    ax = axs[0]
    ax.format(
        xlabel="Horizontal offset (km)",
        ylabel="Depth (km)",
        labelsize=7,
        ticklabelsize=7
    )
    qmesh = ax.pcolormesh(
        velocity.nodes[:, 0, :, 0],
        velocity.nodes[:, 0, :, 2],
        velocity.values[:, 0, :],
        cmap="Spectral",
        discrete=False,
        vmin=config["Model"]["vpLimits"][0],
        vmax=config["Model"]["vpLimits"][1],
        labels_kw={"fontsize": 5},
    )
    cbar = ax.colorbar(
        qmesh,
        labelsize=5,
        ticklabelsize=5,
    )
    for rays in raysBank:
        for ray in rays:
            ax.plot(
                ray[-1, 0],
                0,
                marker=((0, 0), (-2, 4), (2, 4), (0, 0)),
                edgecolor="k",
                facecolor="w",
                mew=0.3,
                ms=5,
                ls="",
                clip_on=False,
                zorder=3
            )
            ax.plot(ray[:, 0], ray[:, 2], color="k", lw=0.2)
    for source in sources:
        ax.plot(
            *source[0:3:2],
            marker="*",
            edgecolor="k",
            facecolor="w",
            s=2.5,
            mew=0.3,
            ls="",
            zorder=3
        )
    cbar.set_label("Velocity (km/s)")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_aspect(1)
    fig.save(os.path.join("results", f"velocityModel_{rayType}.png")) # type: ignore

def plotSeismicityMap(config, locator="hypo71"):
    print(f"+++ Plotting seismicity map, locator is: '{locator}' ...")
    clat = config["StudyArea"]["lat"]
    clon = config["StudyArea"]["lon"]
    # Projection to convert degrees to km relative to the center of the area
    proj = Proj(f"+proj=sterea\
            +lon_0={clon}\
            +lat_0={clat}\
            +units=km")
    catalog_ini_path = os.path.join(
        "results", "location", locator, "xyzm_initial.dat")
    catalog_unw_path = os.path.join(
        "results", "location", locator, "xyzm_unw.dat")
    catalog_w_path = os.path.join(
        "results", "location", locator, "xyzm_w.dat")
    report_ini, report_unw, report_w = loadxyzm(catalog_ini_path,
                                                catalog_unw_path,
                                                catalog_w_path)
    for db in [report_ini, report_unw, report_w]:
        db[["x", "y"]] = db.apply(
            lambda x: Series(
                proj(longitude=x.Lon, latitude=x.Lat)), axis=1)
        db["z"] = db["Dep"]
    stationPath = os.path.join("inputs", "stations.csv")
    station_df = read_csv(stationPath)
    stationNoiseModel = loadStationNoiseModel()
    stationNoiseModel = DataFrame(stationNoiseModel).T
    stationNoiseModel["code"] = stationNoiseModel.index
    station_df = merge(station_df, stationNoiseModel, on="code")
    station_df["probabilityOfOccurrence"] = to_numeric(
        station_df["probabilityOfOccurrence"])
    xMin, xMax = station_df.x.min(), station_df.x.max()
    yMin, yMax = station_df.y.min(), station_df.y.max()
    xMin -= 0.05*(xMax-xMin)
    xMax += 0.05*(xMax-xMin)
    yMin -= 0.05*(yMax-yMin)
    yMax += 0.05*(yMax-yMin)
    zMin, zMax = 0, config["FGS"]["SeismicityMapZMax"]
    axShape = [
        [1]
    ]
    plt.rc.update(
        {'fontsize': 7, 'legend.fontsize': 6, 'label.weight': 'bold'})
    fig, axs = plt.subplots(axShape, share=False)
    axs.format(xlocator=("maxn", 4), ylocator=("maxn", 4))
    [ax.grid(ls=":") for ax in axs]

    axs[0].format(
        xlim=(xMin, xMax),
        ylim=(yMin, yMax),
        xlabel="X (km)",
        ylabel="Y (km)")

    X = [report_ini.x, ]#report_unw.x, report_w.x]
    Y = [report_ini.y, ]#report_unw.y, report_w.y]
    M = [report_ini.Mag, ]#report_unw.Mag, report_w.Mag]
    C = ["green", "red", "blue"]
    L = ["Raw", "Relocated$_{unw}$", "Relocated$_w$"]
    for x, y, m, c, l in zip(X, Y, M, C, L):
        if m.isna().sum() == m.size:
            m = 5
        axs[0].scatter(x.values, y.values, s=m, marker="o", c=c, lw=0.4,
                       edgecolors="k", alpha=.5, label=l,
                       legend="t", legend_kw={'ncol': 3})
    # cs = station_df.probabilityOfOccurrence*100
    cs = [_[0] for _ in station_df.siteNoiseLevel]
    sc = axs[0].scatter(station_df.x, station_df.y, marker="^", lw=0.4,
                        edgecolors="k", s=50, c=cs,
                        cmap="Spectral", vmin=0, vmax=5.0)
    cbar = axs[0].colorbar(
        sc,
        labelsize=7,
        ticklabelsize=6,
        loc="t",
        width=0.1
    )
    # cbar.set_label("Contribution (%)")
    cbar.set_label("Noise level (s)")
    for ii, jj, ss in zip(
            station_df.x, station_df.y, station_df.code):
        dy = 0.05*(station_df.y.max() - station_df.y.min())
        axs[0].text(x=ii, y=jj-dy, s=ss, border=True, borderinvert=True,
                    borderwidth=1,
                    **{"weight": "bold", "size": "xx-small", "ha": "center"})
    for fault in config["FSS"]["Catalog"]["Faults"]:
        n = fault["name"]
        x = fault["dx"]
        y = fault["dy"]
        w = fault["width"]
        s = fault["strike"]
        x += w*cos(deg2rad(90+s))
        y += w*sin(deg2rad(90+s))
        axs[0].text(x=x, y=y, s=n, border=True, borderinvert=True,
                    borderwidth=1, rotation=s,
                    **{"weight": "bold", "size": "medium", "ha": "center"})
    px = axs[0].panel_axes(side="r", width="5em")
    px.grid(ls=":")
    px.format(xlim=(zMin, zMax), ylim=(yMin, yMax),
              xlabel="Depth (km)", xlocator=("maxn", 2), ylocator=("maxn", 4))
    X = [report_ini.z, ]#report_unw.z, report_w.z]
    Y = [report_ini.y, ]#report_unw.y, report_w.y]
    M = [report_ini.Mag, ]#report_unw.Mag, report_w.Mag]
    C = ["green", "red", "blue"]
    for x, y, c, m in zip(X, Y, C, M):
        if m.isna().sum() == m.size:
            m = 5
        px.scatter(x.values, y.values, s=m, marker="o", c=c,
                   lw=0.4, edgecolors="k", alpha=.5)
    # px.invert_yaxis()

    px = axs[0].panel_axes(side="b", width="5em")
    px.grid(ls=":")
    px.format(xlim=(xMin, xMax), ylim=(zMax, zMin),
              xlabel="Longitude (deg)", ylabel="Depth (km)",
              xlocator=("maxn", 4), ylocator=("maxn", 2))
    X = [report_ini.x, ]#report_unw.x, report_w.x]
    Y = [report_ini.z, ]#report_unw.z, report_w.z]
    M = [report_ini.Mag, ]#report_unw.Mag, report_w.Mag]
    C = ["green", "red", "blue"]
    for x, y, c, m in zip(X, Y, C, M):
        if m.isna().sum() == m.size:
            m = 5
        px.scatter(x.values, y.values, s=m, marker="o", c=c,
                   lw=0.4, edgecolors="k", alpha=.5)
    # px.invert_xaxis()
    fig.save(os.path.join("results", f"seismicity_{locator}.png")) # type: ignore

def plotGapRMS(config, locator, basedOn="Nus GAP | Nus | MDS"):
    print(f"+++ Plotting Gap and RMS based on {basedOn}, locator is: '{locator}' ...")
    catalog_ini_path = os.path.join(
        "results", "location", locator, "xyzm_initial.dat")
    catalog_unw_path = os.path.join(
        "results", "location", locator, "xyzm_unw.dat")
    catalog_w_path = os.path.join(
        "results", "location", locator, "xyzm_w.dat")
    report_ini, report_unw, report_w = loadxyzm(catalog_ini_path,
                                                catalog_unw_path,
                                                catalog_w_path)
    df_com = extractCommons(config, report_unw, report_w)
    unit = {"GAP": "$\\degree$",
            "RMS": "s"}
    axShape = [
        [1, 2],
    ]
    fig, axs = plt.subplots(axShape, share=False)
    axs.format(suptitle=f"Differences in Gap-RMS ({len(df_com)} events)",
               xlocator=('maxn', 5),
               ylocator=('maxn', 5)
               )
    [ax.grid(ls=":") for ax in axs]
    X_gap_u = df_com["GAP_ini"]
    Y_gap_w = df_com["GAP_tar"]
    X_rms_u = df_com["RMS_ini"]
    Y_rms_w = df_com["RMS_tar"]
    C = df_com[basedOn+"_tar"]
    vmin, vmax = (config["FGS"]["Colorbar{0}Min".format(basedOn)],
                  config["FGS"]["Colorbar{0}Max".format(basedOn)])
    scr = None
    for i, (x, y, c, l) in enumerate(zip([X_gap_u, X_rms_u],
                                         [Y_gap_w, Y_rms_w],
                                         [C, C],
                                         ["GAP", "RMS"])):
        axsMin = config["FGS"]["{0}Min".format(l.upper())]
        axsMax = config["FGS"]["{0}Max".format(l.upper())]
        axs[i].format(xlabel="{0}$_{1}$ ({2})".format(l, "{unw}", unit[l]),
                      ylabel="{0}$_{1}$ ({2})".format(l, "w", unit[l]),
                      xlim=(axsMin, axsMax),
                      ylim=(axsMin, axsMax))
        scr = axs[i].scatter(
            x, y, s=50, marker="o",  c=c, lw=0.4, edgecolors="k",
            cmap="Gray_r", vmin=vmin, vmax=vmax)
        axs[i].plot([0, 1], [0, 1], transform=axs[i].transAxes,
                    ls="--", lw=1.0, c="red")
        ix = axs[i].inset([0.125, 0.6, 0.3, 0.3], transform="axes", zoom=False)
        ix.grid(ls=":")
        ix.spines["right"].set_visible(False)
        ix.spines["top"].set_visible(False)
        data = x - y
        data = data[~isnan(data)]
        M = data.mean()
        Std = data.std() + finfo(float).eps
        ix.format(
            title="M={M:.2f}, $\\mu$={Std:.2f}".format(M=M, Std=Std),
            xlocator=('maxn', 3), xlabel=f"({unit[l]})")
        ix.hist(
            data, arange(-Std*4, Std*4, Std/2), lw=0.3, histtype="bar",
            filled=True, alpha=0.7, edgecolor="k", color="gray")
    fig.colorbar(
        scr, row=1, loc="r", extend="both",
        label="Number of used stations", shrink=0.9)
    fig.save(os.path.join("results", f"compare_GapRMS_{locator}.png")) # type: ignore

def plotHypoPairs(config, locator, feature, basedOn):
    print(f"+++ Plotting hypocenter comparison for '{feature}' based on '{basedOn}', locator is: '{locator}' ...")
    catalog_ini_path = os.path.join(
        "results", "location", locator, "xyzm_initial.dat")
    catalog_unw_path = os.path.join(
        "results", "location", locator, "xyzm_unw.dat")
    catalog_w_path = os.path.join(
        "results", "location", locator, "xyzm_w.dat")
    report_ini, report_unw, report_w = loadxyzm(catalog_ini_path,
                                                catalog_unw_path,
                                                catalog_w_path)
    axShape = [
        [1, 2],
    ]
    unit = {"Lon": "$\\degree$",
            "Lat": "$\\degree$",
            "Dep": "km"}
    fig, axs = plt.subplots(axShape)
    axs.format(suptitle="Dislocations",
               xlabel="{0}-Raw ({1})".format(feature, unit[feature]),
               ylabel="{0}-Relocated ({1})".format(feature, unit[feature]),
               xlocator=('maxn', 5),
               ylocator=('maxn', 5)
               )
    [ax.grid(ls=":") for ax in axs]
    df_ini_unw_com = extractCommons(config, report_ini, report_unw)
    df_ini_w_com = extractCommons(config, report_ini, report_w)
    X_u = df_ini_unw_com[feature+"_ini"]
    X_w = df_ini_w_com[feature+"_ini"]
    Y_u = df_ini_unw_com[feature+"_tar"]
    Y_w = df_ini_w_com[feature+"_tar"]
    C_u = df_ini_unw_com[basedOn+"_tar"]
    C_w = df_ini_w_com[basedOn+"_tar"]
    vmin, vmax = (config["FGS"]["Colorbar{0}Min".format(basedOn)],
                  config["FGS"]["Colorbar{0}Max".format(basedOn)])
    axMin, axMax = getMinMax(X_u, X_w, Y_u, Y_w)
    scr = None
    for i, (x, y, c, l) in enumerate(zip([X_u, X_w],
                                         [Y_u, Y_w],
                                         [C_u, C_w],
                                         ["Unweighted", "Weighted"])):
        axs[i].format(title="{0}".format(l+f" ({len(x)} events)"),
                      xlim=(axMin, axMax),
                      ylim=(axMin, axMax))
        scr = axs[i].scatter(
            x, y, s=25, marker="o",  c=c, lw=0.4, edgecolors="k",
            cmap="Gray_r", vmin=vmin, vmax=vmax)
        axs[i].plot([0, 1], [0, 1], transform=axs[i].transAxes,
                    ls="--", lw=1.0, c="red")
        ix = axs[i].inset([0.125, 0.6, 0.3, 0.3], transform="axes", zoom=False)
        ix.grid(ls=":")
        ix.spines["right"].set_visible(False)
        ix.spines["top"].set_visible(False)
        data = x.values - y.values # type: ignore        data = data[~isnan(data)]
        if feature in ["Lon", "Lat"]:
            data = d2k(data)
        M = data.mean()
        Std = data.std()
        ix.format(
            title="M={M:.2f}, $\\mu$={Std:.2f}".format(M=M, Std=Std),
            xlocator=('maxn', 3))
        ix.hist(
            data, arange(-Std*4, Std*4, Std/2), histtype="bar", ew=0.3,
            filled=True, alpha=0.7, edgecolor="w", color="gray")
    fig.colorbar(
        scr, row=1, loc="r", extend="both",
        label="{0} ({1})".format(basedOn, unit[feature]), shrink=0.9)
    fig.save(os.path.join("results", f"compare_{feature}_{locator}.png")) # type: ignore

def plotHistPairs(config, locator, feature):
    print(f"+++ Plotting histograms for {feature}, locator is: '{locator}' ...")
    catalog_ini_path = os.path.join(
        "results", "location", locator, "xyzm_initial.dat")
    catalog_unw_path = os.path.join(
        "results", "location", locator, "xyzm_unw.dat")
    catalog_w_path = os.path.join(
        "results", "location", locator, "xyzm_w.dat")
    report_ini, report_unw, report_w = loadxyzm(catalog_ini_path,
                                                catalog_unw_path,
                                                catalog_w_path)
    (HistERHMax, HistERZMax) = (config["FGS"]["HistERHMax"],
                                config["FGS"]["HistERZMax"])
    (HistERHInc, HistERZInc) = (config["FGS"]["HistERHInc"],
                                config["FGS"]["HistERZInc"])
    df_ini_unw_com = extractCommons(config, report_ini, report_unw)
    df_ini_w_com = extractCommons(config, report_ini, report_w)
    df_unw_w_com = extractCommons(config, report_unw, report_w)
    calError_unw_h = df_unw_w_com["ERH_ini"]
    calError_unw_z = df_unw_w_com["ERZ_ini"]
    calError_w_h = df_unw_w_com["ERH_tar"]
    calError_w_z = df_unw_w_com["ERZ_tar"]
    plt.rc.update({"legend.fontsize": 6})
    axShape = [
        [1],
    ]
    fig, axs = plt.subplots(axShape)
    axs.format(suptitle="Absolute error",
               xlabel="{0} error (km)".format(feature),
               ylabel="Number of events",
               xlocator=('maxn', 5),
               ylocator=('maxn', 5),
               lrtitle=f"unw (#): {len(df_ini_unw_com)}\nw (#): {len(df_ini_w_com)}"
               )
    [ax.grid(ls=":") for ax in axs]
    labels = ["$raw-rel_{unw} (km)$", "$raw-rel_{w} (km)$"]
    colors = ["teal7", "orange7"]
    colors = ["gray7", "gray5"]
    dislocations = calError_unw = calError_w = []
    errMax = errInc = None
    if feature == "Horizontal":
        dislocation_u = d2k(l2d(df_ini_unw_com["Lat_ini"],
                                df_ini_unw_com["Lon_ini"],
                                df_ini_unw_com["Lat_tar"],
                                df_ini_unw_com["Lon_tar"]))
        dislocation_w = d2k(l2d(df_ini_w_com["Lat_ini"],
                                df_ini_w_com["Lon_ini"],
                                df_ini_w_com["Lat_tar"],
                                df_ini_w_com["Lon_tar"]))
        dislocations = [dislocation_u, dislocation_w]
        errMax = HistERHMax
        errInc = HistERHInc
        calError_unw = histogram(calError_unw_h, arange(0, errMax, errInc))
        calError_w = histogram(calError_w_h, arange(0, errMax, errInc))
    elif feature == "Depth":
        dislocation_u = abs(
            df_ini_unw_com["Dep_ini"] - df_ini_unw_com["Dep_tar"])
        dislocation_w = abs(
            df_ini_w_com["Dep_ini"] - df_ini_w_com["Dep_tar"])
        dislocations = [dislocation_u, dislocation_w]
        errMax = HistERZMax
        errInc = HistERZInc
        calError_unw = histogram(calError_unw_z, arange(0, errMax, errInc))
        calError_w = histogram(calError_w_z, arange(0, errMax, errInc))
    x, y, dy = 0.7, 0.7, 0.05
    for i, (dislocation, color, label) in enumerate(zip(dislocations,
                                                        colors,
                                                        labels)):

        axs[0].hist(
            dislocation, arange(0, errMax, errInc),
            filled=True, density=False, alpha=0.7, edgecolor="w",
            color=color, histtype="bar", labels=label, legend="ur",
            legend_kw={"ncol": 1})
        q25 = DataFrame({"D": dislocation}).quantile(0.25)[0]
        q50 = DataFrame({"D": dislocation}).quantile(0.50)[0]
        y += -dy*i
        if i == 0:
            label = "unw"
            axs[0].plot(calError_unw[1][:-1], calError_unw[0],
                        color="k", lw=2.2, alpha=0.7)
            axs[0].plot(calError_unw[1][:-1], calError_unw[0], color=color, lw=2.0, alpha=0.7,
                        label="$error_{unw}$", legend="ur", legend_kw={"ncol": 1})
        else:
            label = "w"
            axs[0].plot(calError_w[1][:-1], calError_w[0],
                        color="k", lw=2.2, alpha=0.7)
            axs[0].plot(calError_w[1][:-1], calError_w[0], color=color, lw=2.0, alpha=0.7,
                        label="$error_{w}$", legend="ur", legend_kw={"ncol": 1})
        axs[0].text(x, y, f"$Q_{{25\\%-{label}}} = {q25:0.1f}km$",
                    transform=axs[0].transAxes, size=6)
        y += -dy
        axs[0].text(x, y, f"$Q_{{50\\%-{label}}} = {q50:0.1f}km$",
                    transform=axs[0].transAxes, size=6)
    fig.save(os.path.join("results", f"hist_{feature}_{locator}.png")) # type: ignore

def plotNoise(config):
    noise_df = read_csv(os.path.join("results", "noise.csv"))
    axShape = [
        [1, 2],
    ]
    fig, axs = plt.subplots(axShape, share=True)
    axs.format(suptitle="Noise - Weight",
               xlabel="Residual (s)",
               ylabel="Number of phase",
               xlocator=('maxn', 5),
               ylocator=('maxn', 5),
               xlim=(-config["FGS"]["noiseMax"], config["FGS"]["noiseMax"])
               )
    [ax.grid(ls=":") for ax in axs]
    df_P = [noise_df[(noise_df.phase == "P") & (noise_df.wet == i)]
            for i in range(5)]
    df_S = [noise_df[(noise_df.phase == "S") & (noise_df.wet == i)]
            for i in range(5)]
    M_P = noise_df[noise_df.phase == "P"].noise.mean()
    Std_P = noise_df[noise_df.phase == "P"].noise.std()
    M_S = noise_df[noise_df.phase == "S"].noise.mean()
    Std_S = noise_df[noise_df.phase == "S"].noise.std()
    colors = ["green7", "cyan7", "grape7", "orange7", "black"]
    labels = [str(i) for i in range(5)]
    for df, color, label in zip(df_P, colors, labels):
        axs[0].format(ultitle=f"M={M_P:.2f}, $\\mu$={Std_P:.2f}")
        axs[0].hist(df.noise,
                    arange(-2, 2, 0.2),
                    filled=True, density=False,
                    alpha=0.7, edgecolor="w",
                    color=color, histtype="bar",
                    labels=label)
    for df, color, label in zip(df_S, colors, labels):
        axs[1].format(ultitle=f"M={M_S:.2f}, $\\mu$={Std_S:.2f}")
        axs[1].hist(df.noise,
                    arange(-2, 2, 0.2),
                    filled=True, density=False,
                    alpha=0.7, edgecolor="w",
                    color=color, histtype="bar",
                    labels=label, legend="r",
                    legend_kw={"title": "Weight", "ncol": 1})
    fig.save(os.path.join("results", "hist_noise.png")) # type: ignore

def plotStatistics(config, locator):
    plotHypoPairs(config, locator, "Lon", "GAP")
    plotHypoPairs(config, locator, "Lat", "GAP")
    plotHypoPairs(config, locator, "Dep", "MDS")
    plotHistPairs(config, locator, "Horizontal")
    plotHistPairs(config, locator, "Depth")
    plotGapRMS(config, locator, "Nus")
