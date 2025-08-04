import os

import numpy
from qgis._core import QgsDoubleRange, QgsGeometry, QgsProviderRegistry
from qgis.core import QgsPointCloudLayer

try:  # From QGIS
    from ..pyqgis_tools import layer as layer_tools
except ImportError:
    from pyqgis_tools import layer as layer_tools
    from . import load_env


from . import crs


def load_layer(path):
    if "pdal" not in QgsProviderRegistry.instance().providerList():
        raise Exception("PDAL provider not set")
    if not os.path.exists(path):
        raise FileNotFoundError
    layer_cloud = QgsPointCloudLayer(path, "pointcloud", "pdal")
    if not layer_cloud.isValid():
        raise Exception("File not valid")
    provider = layer_cloud.dataProvider()
    if not provider.hasValidIndex():
        provider.generateIndex()
    provider.loadIndex()
    print(f"File {path} loaded with {provider.pointCount()} points", )
    return layer_cloud, provider


def load_points(
    crs_utm=None,
    path=None,
    layer=None,
    provider=None,
    error=0,
    inbound=None,
    feature_num=1,
    min_z=None,
    max_z=None,
    n=None,
    sample=1,
    attributes=None
):
    if path is not None and provider is None:
        layer, provider = load_layer(path)
    elif layer is not None and provider is None:
        provider = layer.dataProvider()
    elif layer is not None and provider is not None:
        pass
    else:
        print("Data source is needed")
        return
    crs_cloud = layer.crs()
    if not provider.hasValidIndex():
        provider.generateIndex()
    provider.loadIndex()
    if n is None:
        n = provider.pointCount()
    if min_z is None:
        min_z = -float('inf')
    if max_z is None:
        max_z = float('inf')
    if inbound is None:
        geometry = provider.polygonBounds()
    else:
        polygon = layer_tools.get_polygon_feature(inbound, feature_num)
        inbound_crs_cloud = crs.project_polygon(polygon, crs_utm, crs_cloud)
        geometry = QgsGeometry.fromWkt(inbound_crs_cloud.wkt)
        crs_utm = inbound.crs()
        print(f"{crs_utm.authid() = }")
    print(f"{crs_cloud.authid() = }")
    print(f"Provider extent: {provider.extent().toString()}")
    print(f"Point count: {provider.pointCount()}")
    print(f"{geometry.intersects(QgsGeometry.fromRect(provider.extent())) = }")
    print(f"Loading point cloud:\n\t{error = }\n\t{geometry = }\n\t{n = }\n\textentZRange = {[min_z, max_z]}")

    points = provider.identify(
        maxErrorInMapCoords=error,
        extentGeometry=geometry,
        extentZRange=QgsDoubleRange(min_z, max_z),
        pointsLimit=max(1, n-1))
        
    if not len(points):
        raise Exception("No points loaded")
        
    print(f"Points loaded: {len(points)}")
    print(f"Parsing points")
    if attributes is None:
        attributes = list(points[0].keys())
    values = {attribute: [] for attribute in attributes}
    for i, point in enumerate(points):
        if i % sample != 0:
            continue
        for attribute in attributes:
            values[attribute].append(point[attribute])
    values = {attribute: numpy.array(value) for attribute, value in values.items()}
    return values


def get_attributes(path=None, layer=None, provider=None):
    return [key for key in load_points(path=path, layer=layer, provider=provider, n=1)]
