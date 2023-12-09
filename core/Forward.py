import os
import sys

import pykonal
from numpy import array, save
from obspy.geodetics.base import degrees2kilometers as d2k
from obspy.taup.taup_geo import calc_dist_azi as cda
from pandas import Series, DataFrame

# def find_closest(velocity, point):
#     x = velocity.nodes[:, 0, 0, 0]
#     y = velocity.nodes[0, :, 0, 1]
#     z = velocity.nodes[0, 0, :, 2]
#     idx = abs(x - point[0]).argmin()
#     idy = abs(y - point[1]).argmin()
#     idz = abs(z - point[2]).argmin()
#     return [idx, idy, idz]


def saveRayPath(ray, source_id):
    rayFile = os.path.join("results", "rays", f"src_{source_id}.npz")
    save(rayFile, array(ray))


def trace(velocity, event, source_id, receivers, phaseType) -> DataFrame:
    if len(receivers) == 0:
        return DataFrame()
    source = [event.x, event.y, event.z]
    travelTime_db = receivers.copy()
    travelTime_db["eid"] = source_id
    dists = receivers.apply(lambda r: Series(
        [cda(event.Latitude,
             event.Longitude,
             r.lat, r.lon,
             6378.137, 0.0033528106647474805)[0]]), axis=1)
    travelTime_db["Dist"] = dists
    azims = receivers.apply(lambda r: Series(
        [cda(event.Latitude,
             event.Longitude,
             r.lat, r.lon,
             6378.137, 0.0033528106647474805)[1]]), axis=1)
    travelTime_db["Azim"] = azims
    receivers = receivers.apply(lambda r: Series(
        [r.x, r.y, r.z]), axis=1)
    solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    solver.velocity.min_coords = velocity.min_coords
    solver.velocity.npts = velocity.npts
    solver.velocity.node_intervals = velocity.node_intervals
    solver.velocity.values = velocity.values
    solver.src_loc = source
    # source_indx = find_closest(velocity, source)
    # solver.traveltime.values[source_indx] = 0
    # solver.unknown[source_indx] = False
    # solver.trial.push(*source_indx)
    solver.solve()
    # rays = map(solver.trace_ray, receivers.values)
    # saveRayPath(rays, source_id)
    travelTimes = map(solver.traveltime.value, receivers.values)
    travelTime_db[f"TT{phaseType}"] = list(travelTimes)
    travelTimes_nan = travelTime_db[f"TT{phaseType}"].isna().sum().sum()
    if travelTimes_nan:
        longest_tt_dist = travelTime_db[travelTime_db[f"TT{phaseType}"].isna()]
        longest_tt_dist = longest_tt_dist.sort_values(
            "Dist", ascending=False).Dist.index[0]
        longest_tt_dist = travelTime_db.Dist[longest_tt_dist]
        max_coords = velocity.max_coords
        print("")
        print("! > Found some travel times with NaN values! check if velocity model")
        print("    could cover the entire area of stations and event distributions!")
        print(
            f" - The longest station-event distance is: {d2k(longest_tt_dist):.1f} km, but we allowed {max_coords[0:2]} in x and y.")
        print(" - Increase velocity model dimensions 'Model>(nx, ny, nz)' parameter ...")
        print("or:")
        print(" - Decrease area including stations 'FSS>Stations>radius' parameter ...")
        print(" - Decrease area including events 'Catalog>Faults>(dx, dy)' parameter ...")
        sys.exit()
    return travelTime_db
