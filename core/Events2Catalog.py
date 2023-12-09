from obspy.core import event
from obspy.core.event.origin import Pick
from obspy.core.event.magnitude import Amplitude
from tqdm import tqdm
from datetime import timedelta as td
from numpy import isnan
from core.Noise import addNoiseWeight
from core.Magnitude import getAmplitude
from obspy.geodetics.base import degrees2kilometers as d2k
import os


class feedCatalog():
    """An obspy Catalog Contractor
    """

    def __init__(self):
        pass

    def setPick(self,
                staCode,
                onset,
                phaseHint,
                weight,
                time):
        """Fill the Obspy pick object

        Args:
            staCode (str): station code,
            onset (str): onset of phase
            phaseHint (str): define P or S phase
            weight (int): dedicated weight to the phase
            time (time): arrival time of phase

        Returns:
            obspy.pick: an obspy pick object
        """
        pick = event.Pick()
        pick.onset = onset
        pick.phase_hint = phaseHint
        pick.update({"extra": {"nordic_pick_weight": {"value": 0}}})
        pick.extra.nordic_pick_weight.value = weight
        pick.time = time
        pick.evaluation_mode = "automatic"
        pick.waveform_id = event.WaveformStreamID("BI", staCode)
        return pick

    def setArrival(self,
                   phase,
                   distance,
                   azimuth,
                   pick_id):
        """Fill Obspy arrival object

        Args:
            phase (str): phase code
            time (time): phase arrival time
            distance (float): epicentral distance
            azimuth (float): azimuth between event and station
            pick_id (str): string representing associated pick

        Returns:
            obspy.event.arrival: an obspy arrival object
        """
        arrival = event.Arrival()
        arrival.phase = phase
        arrival.distance = distance
        arrival.azimuth = azimuth
        arrival.pick_id = pick_id
        return arrival

    def setMagnitude(self,
                     eventInfo,
                     magType,
                     origin):
        """Fill Obspy magnitude object

        Args:
            mag (float): magnitude of event
            magType (str): type of magnitude
            origin (obspy.event.origin): an obspy event origin object

        Returns:
            obspy.event.magnitude: an obspy magnitude object
        """
        magnitude = event.Magnitude()
        magnitude.mag = eventInfo.Mag
        magnitude.magnitude_type = magType
        magnitude.origin_id = origin.resource_id
        return magnitude

    def setAmplitudes(self,
                      mag,
                      picks,
                      arrivals):
        amplitudes = []
        PICKS = {pick.resource_id: pick for pick in picks}
        for arrival in arrivals:
            amplitude = Amplitude()
            arrival.update({"pick": PICKS[arrival.pick_id]})
            if "P" == arrival.pick.phase_hint:
                new_amp_pick = Pick()
                new_amp_pick.time = arrival.pick.time
                new_amp_pick.waveform_id = arrival.pick.waveform_id
                new_amp_pick.onset = "impulsive"
                new_amp_pick.phase_hint = "AML"
                new_amp_pick.evaluation_mode = arrival.pick.evaluation_mode
                picks.append(new_amp_pick)
                amplitude.pick_id = new_amp_pick.resource_id
                amplitude.waveform_id = arrival.pick.waveform_id
                amplitude.generic_amplitude = getAmplitude(mag,
                                                           d2k(arrival.distance))
                amplitude.type = "AML"
                amplitude.magnitude_hint = "ML"
                amplitudes.append(amplitude)
        picks = picks
        amplitudes = amplitudes
        return picks, amplitudes

    def getPicksArrivals(self,
                         config,
                         stationNoiseModel,
                         eventInfo,
                         ttimes_db,
                         eid,
                         weighting,
                         noisePath):
        """Make pick and arrival from input dictionary

        Args:
            arrivalsDict (dict): a dictionary contains arrival and pick

        Returns:
            tuple: a tuple contains obspy pick and arrival objects
        """
        picks, arrivals = [], []
        ort = eventInfo.OriginTime
        for r, row in ttimes_db.iterrows():
            if not isnan(row.TTP):
                noise, wet = addNoiseWeight(config,
                                            stationNoiseModel,
                                            row.code,
                                            "P",
                                            r,
                                            weighting)
                if weighting:
                    ttimes_db.loc[r, "phase"] = "P"
                    ttimes_db.loc[r, "noise"] = noise
                    ttimes_db.loc[r, "wet"] = wet
                art = ort+td(seconds=row.TTP+noise)
                pick = self.setPick(row.code, "impulsive", "P", wet, art)
                arrival = self.setArrival(
                    "P", row.Dist, row.Azim, pick.resource_id)
            elif "TTS" in row and not isnan(row.TTS):
                noise, wet = addNoiseWeight(config,
                                            stationNoiseModel,
                                            row.code,
                                            "S",
                                            r,
                                            weighting)
                if weighting:
                    ttimes_db.loc[r, "phase"] = "S"
                    ttimes_db.loc[r, "noise"] = noise
                    ttimes_db.loc[r, "wet"] = wet
                art = ort+td(seconds=row.TTS+noise)
                pick = self.setPick(row.code, "emergent", "S", wet, art)
                arrival = self.setArrival(
                    "S", row.Dist, row.Azim, pick.resource_id)
            else:
                continue
            picks.append(pick)
            arrivals.append(arrival)
        if weighting:
            with open(noisePath, "a") as noiseFile:
                ttimes_db.to_csv(noiseFile,
                                 columns=["eid", "code", "phase", "noise", "wet"],
                                 index=False,
                                 float_format="%.3f",
                                 header=False if eid != 0 else True)
        return picks, arrivals

    def setOrigin(self,
                  eventInfo,
                  arrivals):
        """Fill Obspy origin object

        Args:
            eventInfoDict (dict): a dictionary contains event information
            arrivals (obspy.arrival): an obspy arrivals

        Returns:
            obspy.origin: an obspy origin object
        """
        origin = event.Origin()
        origin.time = eventInfo.OriginTime
        origin.latitude = eventInfo.Latitude
        origin.longitude = eventInfo.Longitude
        origin.depth = eventInfo.Depth*1e3
        origin.arrivals = arrivals
        return origin

    def setEvent(self,
                 config,
                 stationNoiseModel,
                 eventInfo,
                 ttimes_db,
                 eid,
                 weighting,
                 noisePath):
        """Fill Obspy event object

        Args:
            eventInfoDict (dict): a dictionary contains event information
            arrivalsDict (dict): a dictionary contains event arrivals

        Returns:
            obspy.event: an obspy event
        """
        picks, arrivals = self.getPicksArrivals(config,
                                                stationNoiseModel,
                                                eventInfo,
                                                ttimes_db,
                                                eid,
                                                weighting,
                                                noisePath)
        Event = event.Event()
        origin = self.setOrigin(eventInfo, arrivals)
        magnitude = self.setMagnitude(eventInfo, "Ml", origin)
        picks, amplitudes = self.setAmplitudes(magnitude.mag, picks, arrivals)
        Event.origins.append(origin)
        Event.magnitudes.append(magnitude)
        Event.picks = picks
        Event.amplitudes = amplitudes
        return Event

    def setCatalog(self,
                   config,
                   hypocenter_db,
                   bulletin_db,
                   stationNoiseModel,
                   weighting):
        """Fill Obspy catalog object

        Args:
            eventsInfoList (list): a list contains event information
            eventArrivals (list): a list contains event information

        Returns:
            _type_: _description_
        """
        catalog = event.Catalog()
        noisePath = os.path.join("results", "noise.csv")
        with open(noisePath, "w") as _:
            pass
        desc = f"+++ Generating Obspy catalog weighting={weighting} ..."
        for eid, eventInfo in tqdm(hypocenter_db.iterrows(), desc=desc):
            ttimes_db = bulletin_db[bulletin_db.eid == eid]
            if "TTS" in ttimes_db:
                ttimes_db = ttimes_db.sort_values(by=["Dist", "code", "TTP", "TTS"])
            else:
                ttimes_db = ttimes_db.sort_values(by=["Dist", "code", "TTP"])
            Event = self.setEvent(config,
                                  stationNoiseModel,
                                  eventInfo,
                                  ttimes_db,
                                  eid,
                                  weighting,
                                  noisePath)
            catalog.append(Event)
        return catalog
