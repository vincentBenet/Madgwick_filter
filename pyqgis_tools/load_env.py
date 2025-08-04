import sys
import os

from qgis import core, analysis
from qgis.core import QgsApplication
from qgis.core import QgsProviderRegistry
from qgis.analysis import QgsNativeAlgorithms

qgis_path = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
sys.path.append(os.path.join(qgis_path, "apps", "qgis-ltr", "python"))
sys.path.append(os.path.join(qgis_path, "apps", "qgis-ltr", "python", "plugins"))
qgis_bin_path = os.path.join(qgis_path, "bin")
os.environ['PATH'] = qgis_bin_path + ";" + os.environ['PATH']
sys.path.append(qgis_bin_path)

QgsApplication.setPrefixPath(qgis_path, True)
qgis_app = QgsApplication([], False)
qgis_app.initQgis()

from plugins.processing.core import Processing

Processing.Processing.initialize()
core.QgsApplication.processingRegistry().addProvider(analysis.QgsNativeAlgorithms())
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())


def list_tools():
    tools = []
    for alg in QgsApplication.processingRegistry().algorithms():
        tools.append([alg, alg.id(), alg.displayName()])
    return tools
