from numpy import linspace
from numpy.random import RandomState


def addNoiseWeight(
        config,
        stationNoiseLevel,
        staCode,
        phaseType,
        noiseID,
        weighting=True):
    S_P_Error = stationNoiseLevel[staCode]["S_P_Error"]
    pickingErrorMax = config["FSS"]["Catalog"]["pickingErrorMax"]
    pickingErrorMax *= S_P_Error
    rng = RandomState(config["FSS"]["Catalog"]["rndID"]+noiseID)
    noise = 0
    m, sigma = stationNoiseLevel[staCode]["siteNoiseLevel"]
    if phaseType == "S":
        sigma *= S_P_Error
    noise += rng.normal(m, sigma)
    if not weighting:
        return noise, 0
    for i, j, w in zip(linspace(0, 75, 4), linspace(25, 100, 4), range(4)):
        c1 = i*pickingErrorMax*1e-2 <= abs(noise)
        c2 = abs(noise) < j*pickingErrorMax*1e-2
        if (c1) & (c2):
            return noise, w
    else:
        return noise, 4
