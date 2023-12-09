from core.Catalog import generateCatalog
from core.Extra import readConfiguration
from core.Report import prepareReport
from core.Station import generateStationNoiseModel, generateStations
from core.VelocityModel import createVelocityModel
from core.Visualize import (plotNoise, plotSeismicityMap, plotStatistics,
                            plotVelocityModel2D, plotVelocityModel3D)
from hypo71.Locate import locateHypo71
from hypocenter.Locate import locateHypocenter
from hypodd.Locate import locateHypoDD


class Main():
    def __init__(self):
        self.config = self.readConfig()
        self.locators = ["hypocenter", "hypo71", "hypoDD"]

    def readConfig(self):
        config = readConfiguration()
        return config

    def prepareVelocityModel(self):
        createVelocityModel(self.config)

    def prepareStations(self):
        generateStations(self.config)
        generateStationNoiseModel(self.config)

    def prepareCatalog(self):
        generateCatalog(self.config)

    def locateCatalog(self):
        locateHypocenter(self.config)
        locateHypo71(self.config)
        locateHypoDD(self.config)

    def visualizeVelocityModel(self):
        plotVelocityModel2D(self.config)
        plotVelocityModel3D(self.config)

    def visualizeCatalog(self):
        for locator in self.locators:
            plotSeismicityMap(self.config, locator)
            plotStatistics(self.config, locator)
        plotNoise(self.config)

    def reportCatalog(self):
        for locator in self.locators:
            prepareReport(self.config, locator)


if "__main__" == __name__:
    app = Main()
    app.prepareVelocityModel()
    app.prepareStations()
    app.prepareCatalog()
    app.locateCatalog()
    app.visualizeVelocityModel()
    app.visualizeCatalog()
    app.reportCatalog()
