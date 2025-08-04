import numpy
import shapely
from qgis._core import QgsCoordinateReferenceSystem, QgsProject, QgsCoordinateTransform, QgsPointXY


def get_crs_utm(point):
    return QgsCoordinateReferenceSystem(f"EPSG:32{'6' if point.y > 0 else 7}{int(30 + point.x / 6) + 1}")


def project_polygons(polygons, crs_source, crs_target):
    return shapely.MultiPolygon([
        project_polygon(polygon, crs_source, crs_target)
        for polygon in polygons.geoms
    ])


def project_polygon(polygon, crs_source, crs_target):
    return shapely.Polygon([
        (point.x, point.y)
        for point in
        project_points(shapely.MultiPoint(polygon.exterior.coords), crs_source, crs_target).geoms
    ])


def project_lines(lines, crs_source, crs_target):
    result = []
    for line in lines.geoms:
        result.append(project_line(line, crs_source, crs_target))
    return shapely.MultiLineString(result)


def project_line(line, crs_source, crs_target):
    points = shapely.MultiPoint([shapely.Point(point) for point in line.coords])
    points_projected = project_points(points, crs_source, crs_target)
    return shapely.LineString(points_projected.geoms)


def project_points(points, crs_source, crs_target):
    transformer = get_transformer(crs_source, crs_target)
    return shapely.MultiPoint([project_point(point, transformer) for point in points.geoms])


def project_point(point, transformer):
    x = point.x
    y = point.y
    point = QgsPointXY(x, y)
    point_projected = transformer.transform(point)
    return shapely.Point(point_projected)


def get_transformer(crs_source=None, crs_target=None):
    if crs_source is None:
        crs_source = QgsCoordinateReferenceSystem.fromEpsgId(4326)
    if crs_target is None:
        crs_target = QgsCoordinateReferenceSystem.fromEpsgId(4326)
    return QgsCoordinateTransform(crs_source, crs_target, QgsProject.instance().transformContext())
