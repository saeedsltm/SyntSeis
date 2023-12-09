from numpy import log10, power


def getMagnitude(amp, dist):
    """
    Calculate Local magnitude Ml for California [Hutton and Boore, 1987]
    SEISAN V 10.0 P-110

    Parameters
    ----------
    amp : float
        maximum ground amplitude (zero−peak) in nm.
    dist : float
        hypocentral distance in km.

    Returns
    -------
    Ml : float
        Local magnitude Ml.

    """
    Ml = log10(amp) + 1.11*log10(dist) + 0.00189*dist - 2.09
    return Ml


def getAmplitude(mag, dist):
    """
    Calculate amplitude for a given local magnitude
    SEISAN V 10.0 P-110

    Parameters
    ----------
    mag : float
        Local magnitude Ml.
    dist : float
        hypocentral distance in km.

    Returns
    -------
    amp : float
        maximum ground amplitude (zero−peak) in nm.

    """
    amp = power(10, mag - 1.11*log10(dist) - 0.00189*dist + 2.09)
    return amp
