from qgis.core import QgsMapLayerProxyModel

from . import layer


def only_vectors(widget):
    widget.setFilters(QgsMapLayerProxyModel.VectorLayer)


def only_rasters(widget):
    widget.setFilters(QgsMapLayerProxyModel.RasterLayer)


def only_points_vectors(widget):
    widget.setFilters(QgsMapLayerProxyModel.PointLayer)


def only_lines_vectors(widget):
    widget.setFilters(QgsMapLayerProxyModel.LineLayer)


def only_polygons_vectors(widget):
    widget.setFilters(QgsMapLayerProxyModel.PolygonLayer)


def only_cloud(widget):
    widget.setFilters(QgsMapLayerProxyModel.PointCloudLayer)


def get_actual_layer(widget):
    return widget.currentLayer()


def filter_empty_vectors(widget):  # Remove empty geometry vectors
    widget.setExceptedLayerList(layer.get_all_empty())


def empty_layer(widget):
    widget.setLayer(None)

def only_raster_and_cloud(widget):
    widget.setFilters(QgsMapLayerProxyModel.PointCloudLayer|QgsMapLayerProxyModel.RasterLayer)