import copy
import os
import tempfile
from pathlib import Path
import numpy
import shapely
from PyQt5.QtCore import QVariant
from qgis import processing
from qgis._core import QgsPoint, QgsPointXY, QgsProcessingFeedback, QgsCoordinateTransform, QgsMessageLog, QgsRectangle, \
    QgsUnitTypes
from qgis.core import QgsRasterLayer
from qgis.analysis import QgsRasterCalculatorEntry
from qgis.analysis import QgsRasterCalculator
from qgis.core import (QgsProject, QgsCoordinateReferenceSystem, QgsVectorLayer, QgsVectorFileWriter, QgsField,
                       QgsFeature, QgsGeometry, QgsLineString)
from qgis.utils import iface

from . import crs as crs_tools, geotiff


def get_xyzm_point(layer, default_z=float("nan"), defaut_m=float("nan")):
    if layer is None:
        return None
    layer_crs = get_crs(layer)
    projecting = False
    if layer_crs.mapUnits() != QgsUnitTypes.DistanceMeters:
        crs_utm = get_crs_utm(layer)
        transformer = crs_tools.get_transformer(layer_crs, crs_utm)
        projecting = True
    points = []
    for i, feature in enumerate(layer.getFeatures()):
        geom = feature.geometry()
        point = geom.constGet()
        x = point.x()
        y = point.y()
        if projecting:
            point_xy = QgsPointXY(x, y)
            transformed_point = transformer.transform(point_xy)
            x = transformed_point.x()
            y = transformed_point.y()
        z = point.z() if hasattr(point, "z") else default_z
        m = point.m() if hasattr(point, "m") else defaut_m
        points.append([x, y, z, m])
    return numpy.array(points)


def get_xy(layer):
    if layer is None:
        return None

    layer_type = get_type(layer)
    layer_crs = get_crs(layer)

    # Extract coordinates based on layer type
    if layer_type == "Points":
        coords = get_xy_points(layer)
    elif layer_type == "Lines":
        coords = get_xy_lines(layer)
    elif layer_type == "Polygons":
        coords = layer_to_multipolygon(layer)
    elif layer_type == "Raster":
        # For raster, return the extent as a polygon
        extent = layer.extent()
        coords = shapely.Polygon([
            (extent.xMinimum(), extent.yMinimum()),
            (extent.xMaximum(), extent.yMinimum()),
            (extent.xMaximum(), extent.yMaximum()),
            (extent.xMinimum(), extent.yMaximum()),
            (extent.xMinimum(), extent.yMinimum())
        ])
    else:
        return None

    # Check if projection is needed
    if layer_crs.mapUnits() != QgsUnitTypes.DistanceMeters:
        # Get UTM CRS for the layer
        crs_utm = get_crs_utm(layer)

        # Project coordinates to UTM
        if layer_type == "Points":
            return crs_tools.project_points(coords, layer_crs, crs_utm)
        elif layer_type == "Lines":
            return crs_tools.project_lines(coords, layer_crs, crs_utm)
        elif layer_type == "Polygons":
            return crs_tools.project_polygons(coords, layer_crs, crs_utm)
        elif layer_type == "Raster":
            # For raster, project the extent polygon
            return crs_tools.project_polygon(coords, layer_crs, crs_utm)

    return coords


def create_raster(path, dem_x, dem_y, dem_z, crs):
    if not path.endswith(".tif"):
        path += ".tif"
    n1 = len(dem_x)
    m1 = len(dem_y)
    n2, m2 = numpy.shape(dem_z.T)
    assert n1 == n2
    assert m1 == m2
    print(f"Creating raster {path} with {n1}x{m1} pixels")
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    layer = geotiff.write_geotiff(
        x_data=dem_x,
        y_data=dem_y,
        z_data=dem_z,
        file_path=Path(path),
        crs=crs
    )
    return layer


def get_max_raster(layer):
    _, _, z = load_raster_band(layer)
    return numpy.max(z)


def crop_raster(raster, inbound, crs_utm=None, path=None):
    if raster is None:
        return None

    print(f"Cropping raster {raster.name()}")

    # Convert Shapely polygon to QgsGeometry
    qgis_polygon = QgsGeometry.fromWkt(inbound.wkt)

    # Get raster CRS
    crs_dest = raster.crs()

    # Reproject polygon if needed
    if crs_utm and crs_utm != crs_dest:
        transform = QgsCoordinateTransform(crs_utm, crs_dest, QgsProject.instance())
        qgis_polygon.transform(transform)

    # Create a temporary vector layer for the polygon
    temp_vector_path = os.path.join(tempfile.gettempdir(), "temp_mask.geojson")
    vector_layer = QgsVectorLayer("Polygon?crs=" + crs_dest.authid(), "mask_layer", "memory")
    provider = vector_layer.dataProvider()

    # Define fields and add geometry
    provider.addAttributes([QgsField("id", QVariant.Int)])
    vector_layer.updateFields()

    feature = QgsFeature()
    feature.setGeometry(qgis_polygon)
    feature.setAttributes([1])
    provider.addFeature(feature)

    # Save the temporary vector file
    QgsVectorFileWriter.writeAsVectorFormat(vector_layer, temp_vector_path, "UTF-8", crs_dest, "GeoJSON")

    # Define output file
    if path is None:
        _, final_output = tempfile.mkstemp(suffix=".tif")
    else:
        final_output = path
        print(f"Export cropped raster to {final_output}")

    print(f"Extent: {qgis_polygon.boundingBox().toString()}")
    print(f"CRS source: {crs_utm.authid() if crs_utm else 'Not specified'}")
    print(f"CRS destination: {raster.crs().authid()}")

    assert raster.isValid(), "Target raster is invalid"
    assert raster.bandCount() >= 1, "Target raster does not have at least 1 band"

    # Run clipping process
    feedback = QgsProcessingFeedback()
    result = processing.run("gdal:cliprasterbymasklayer", {
        'INPUT': raster.source(),
        'MASK': temp_vector_path,  # Use the saved vector file
        'SOURCE_CRS': raster.crs(),
        'TARGET_CRS': raster.crs(),
        'CROP_TO_CUTLINE': True,
        'KEEP_RESOLUTION': True,
        'NODATA': None,
        'DATA_TYPE': 0,  # Keep original
        'OUTPUT': final_output
    }, feedback=feedback)

    if not result or 'OUTPUT' not in result:
        print("Raster cropping failed")
        return None

    return QgsRasterLayer(result['OUTPUT'], f"{raster.name()}_cropped")


def raster_substract(raster_up, raster_low, inbound=None, crs_utm=None, path=None):
    if raster_up is None or raster_low is None:
        return None
    print(f"Raster Substraction")
    target = QgsRasterCalculatorEntry()
    target.raster = raster_up
    target.bandNumber = 1
    target.ref = 'target_raster@1'
    initial = QgsRasterCalculatorEntry()
    initial.raster = raster_low
    initial.bandNumber = 1
    initial.ref = 'initial_raster@1'
    if path is None:
        _, final_output = tempfile.mkstemp()
    else:
        final_output = path
        print(f"Export DEM substract to {final_output}")
    if inbound is None:
        extend = raster_up.extent()
    else:
        crs_dest = raster_up.crs()
        inbound_crs = crs_tools.project_polygon(inbound, crs_utm, crs_dest)
        extend = QgsGeometry.fromWkt(inbound_crs.wkt).boundingBox()
    # print(f"Extent: {extend.toString()}")
    # print(f"CRS source: {crs_utm.authid()}")
    # print(f"CRS destination: {raster_up.crs().authid()}")
    assert raster_up.isValid(), "Target raster is invalid"
    assert raster_low.isValid(), "Initial raster is invalid"
    assert raster_up.bandCount() >= 1, "Target raster does not have at least 1 band"
    assert raster_low.bandCount() >= 1, "Initial raster does not have at least 1 band"
    calc = QgsRasterCalculator(
        formulaString='%s - %s' % (target.ref, initial.ref),
        outputFile=final_output,
        outputFormat='GTiff',
        outputExtent=extend,
        nOutputColumns=raster_up.width(),
        nOutputRows=raster_up.height(),
        rasterEntries=[target, initial])
    result = calc.processCalculation()
    if result != 0:  # 0 indique un succès
        raise Exception(f"Raster calculation failed with error code {result}")
    return QgsRasterLayer(final_output, "DEM_Obstacles")


def get_raster_steps(layer):
    nb_x = layer.width()
    nb_y = layer.height()
    extent = layer.extent()
    x_size = extent.width()
    y_size = extent.height()
    step_x = x_size / nb_x
    step_y = y_size / nb_y
    return step_x, step_y


def get_raster_step_min(layer):
    return min(get_raster_steps(layer))


def load_raster_band(layer, band=1, mini=-1e+9):
    if layer is None:
        return None, None, None
    provider = layer.dataProvider()
    extent = layer.extent()
    x_min = extent.xMinimum()
    y_min = extent.yMinimum()
    x_size = extent.width()
    y_size = extent.height()
    nb_x = layer.width()
    nb_y = layer.height()
    n = nb_x * nb_y
    block = provider.block(band, extent, nb_x, nb_y)
    x = numpy.linspace(x_min, x_min + x_size, nb_x)
    y = numpy.linspace(y_min, y_min + y_size, nb_y)
    data = block.data()
    ratio = len(data) / n
    if ratio == 4:
        z = numpy.frombuffer(data, dtype=numpy.float32)
    elif ratio == 8:
        z = numpy.frombuffer(data, dtype=numpy.float64)
    else:
        raise Exception("Raster dtype to enter")
    z_reshaped = z.reshape((nb_y, nb_x))
    z_filtered = numpy.where(z_reshaped < mini, numpy.nan, z_reshaped)  # Patch wrong nan loading
    return x, y, z_filtered


def get_xy_points(layer):
    result = []
    for i, feature in enumerate(layer.getFeatures()):
        geom = feature.geometry()
        if geom.isEmpty():
            continue
        point = geom.asPoint()
        result.append(point)
    return shapely.MultiPoint(result)


def get_xy_lines(layer):
    result = []
    for i, feature in enumerate(layer.getFeatures()):
        geom = feature.geometry()
        if geom.isEmpty():
            continue
        shape = shapely.from_wkt(geom.asWkt())  # Use of shapely to extract geometry
        if isinstance(shape, shapely.MultiLineString):
            result.extend(shape.geoms)  # Extract each LineString separately
        elif isinstance(shape, shapely.LineString):
            result.append(shape)  # Single LineString
    return shapely.MultiLineString(result)


def get_attributes(layer):
    attributes = {}
    fields = layer.fields()
    for i, feature in enumerate(layer.getFeatures()):
        for field in fields:
            key = field.name()
            if key not in attributes:
                attributes[key] = []
            value = feature[key]
            attributes[key].append(value)
    return attributes


def get_feature_geometry(layer, feature_id):
    feature = layer.getFeature(feature_id)
    geometry = feature.geometry()
    return geometry


def get_polygon_feature(layer, feature_id):
    geometry = get_feature_geometry(layer, feature_id)
    if geometry.isEmpty():
        return shapely.Polygon()
    poly = geometry.asPolygon()
    return shapely.Polygon(poly[0], poly[1:])


def layer_to_multipolygon(layer):
    polygons = []
    if layer is not None:
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom.isEmpty():
                continue

            # Handle MultiPolygon geometries
            if geom.isMultipart():
                multi_geom = geom.asMultiPolygon()
                for sub_geom in multi_geom:
                    if sub_geom:
                        polygons.append(shapely.Polygon(sub_geom[0], sub_geom[1:]))  # Ensure correct structure
            else:
                poly = geom.asPolygon()
                if poly:
                    polygons.append(shapely.Polygon(poly[0], poly[1:]))  # Convert to Shapely Polygon

    shape = shapely.MultiPolygon(polygons)
    return shape


def project(layer, crs_source, crs_target):
    type_layer = get_type(layer)
    if type_layer == "Points":
        return crs_tools.project_points(get_xy_points(layer), crs_source, crs_target)
    elif type_layer == "Lines":
        return crs_tools.project_lines(get_xy_lines(layer), crs_source, crs_target)
    elif type_layer == "Polygons":
        return crs_tools.project_polygons(layer_to_multipolygon(layer), crs_source, crs_target)
    else:
        type_layer_num = layer.wkbType()
        print(f"Layer type of {layer} not supported ({type_layer = } : {type_layer_num = })")
        raise NotImplemented


def get_xy_degree(layer):
    return project(layer, layer.crs(), QgsCoordinateReferenceSystem("EPSG:4326"))


def get_xy_meter(layer):
    if layer is None:
        return None
    return project(layer, layer.crs(), get_crs_utm(layer))


def get_valid_geometries(layer):
    result = []
    for i, feature in enumerate(layer.getFeatures()):
        geom = feature.geometry()
        result.append(not geom.isEmpty())
    return numpy.array(result)


def get_crs_utm(layer):
    return crs_tools.get_crs_utm(get_xy_degree(layer).centroid)


def get_type(layer):
    if layer is None:
        return None
    elif not hasattr(layer, "wkbType"):
        return "Raster"
    type_layer_num = layer.wkbType()
    if layer.wkbType() is None:
        print("none")
        return None
    elif type_layer_num in [1, 4, 1001]:
        return "Points"
    elif type_layer_num in [2, 5, 1002]:
        return "Lines"
    elif type_layer_num in [3, 6, 1003]:
        return "Polygons"
    else:
        print(f"{type_layer_num = }")
        print(f"{type(type_layer_num) = }")
        QgsMessageLog.logMessage(f"Layer type of {layer} not supported ({type_layer_num = })")
        return None


def get_all():
    return QgsProject.instance().mapLayers().values()


def get_all_vectors():
    return [layer for layer in get_all() if get_type(layer) in ["Points", "Lines", "Polygons"]]


def get_crs(layer):
    return layer.crs()


def get_all_empty():
    empty_layers = []
    for layer in get_all_vectors():
        is_empty = True
        for _ in layer.getFeatures():
            is_empty = False
            break
        if is_empty:
            empty_layers.append(layer)
    return empty_layers


def get_path(layer):
    return layer.dataProvider().dataSourceUri()


def get_dir(layer):
    return os.path.dirname(get_path(layer))


def get_filename(layer):
    return ".".join(os.path.basename(get_path(layer)).split(".")[:-1])


def multilinestring_to_layer(multilinestring, path, crs):
    if isinstance(multilinestring, shapely.LineString):
        multilinestring = shapely.MultiLineString([multilinestring])
    geometry_data = [numpy.array(route.coords.xy).T for route in multilinestring.geoms]
    layer = create_gpkg_linestrings(path=path, geometry_data=geometry_data, crs=crs)
    return layer


def multipoint_to_layer(multipoint, path, crs):
    if isinstance(multipoint, shapely.Point):
        multipoint = shapely.MultiPoint([multipoint])

    geometry_data = [numpy.array(point.coords.xy).T[0] for point in multipoint.geoms]
    layer = create_gpkg_point(path=path, geometry_data=geometry_data, crs=crs)
    return layer


def create_gpkg_point(path=None, fields=None, geometry_data=None, crs=None, crs_code=None):
    if fields is None:
        fields = {}
    geometries = []

    if not len(geometry_data):
        raise ValueError("No geometry to export")

    for i, pos in enumerate(geometry_data):
        x, y = pos
        point = QgsPoint(x, y)
        geometry = QgsGeometry(point)
        geometries.append(geometry)

    if crs is None:
        crs = QgsCoordinateReferenceSystem(crs_code)

    fields2 = copy.deepcopy(fields)
    print(f"{fields2 = }")

    return create_gpkg(path, fields=fields2, geometries=geometries, crs=crs, layer_type="Point")


def layer_to_multilinestring(layer: QgsVectorLayer):
    if layer is None:
        return shapely.MultiLineString([])
    lines = []
    for feature in layer.getFeatures():
        geom = feature.geometry()
        if geom.isMultipart():
            multiline = geom.asMultiPolyline()
            for line in multiline:
                lines.append(shapely.LineString(line))
        else:
            singleline = geom.asPolyline()
            lines.append(shapely.LineString(singleline))
    return shapely.MultiLineString(lines)


def create_gpkg_linestring(path=None, fields=None, geometry_data=None, crs=None, crs_code=None):
    if fields is None:
        fields = {}
    geometries = []
    if not len(geometry_data):
        return None
    for i in range(len(geometry_data)-1):
        p1 = geometry_data[i]
        p2 = geometry_data[i+1]

        x1, y1 = p1
        x2, y2 = p2

        if "route_z" in fields:
            z1 = fields["route_z"][i]
            z2 = fields["route_z"][i+1]

            point1 = QgsPoint(x1, y1, z1)
            point2 = QgsPoint(x2, y2, z2)
        else:
            point1 = QgsPoint(x1, y1)
            point2 = QgsPoint(x2, y2)

        geometry = QgsGeometry(QgsLineString([point1, point2]))
        geometries.append(geometry)
    if crs is None:
        crs = QgsCoordinateReferenceSystem(crs_code)
    fields2 = copy.deepcopy(fields)
    return create_gpkg(path, fields=fields2, geometries=geometries, crs=crs, layer_type="LineStringZ")


def create_gpkg_linestrings(path=None, fields=None, geometry_data=None, crs=None, crs_code=None):
    if fields is None:
        fields = {}
    geometries = []
    if not len(geometry_data):
        return None
    for i, line in enumerate(geometry_data):
        points = []
        for j, pos in enumerate(line):
            x, y = pos
            if "route_z" in fields:
                z = fields["route_z"][i][j]
                point = QgsPoint(x, y, z)
            else:
                point = QgsPoint(x, y)
            points.append(point)
        geometry = QgsGeometry(QgsLineString(points))
        geometries.append(geometry)
    if crs is None:
        crs = QgsCoordinateReferenceSystem(crs_code)
    fields2 = copy.deepcopy(fields)
    print(f"{fields2 = }")
    if "route_z" in fields2:
        del fields2["route_z"]
    return create_gpkg(path, fields=fields2, geometries=geometries, crs=crs, layer_type="LineString")


def multipolygon_to_layer(multipolygon, crs_utm, zone_id, path, route_name, name):
    if isinstance(multipolygon, shapely.Polygon):
        multipolygon = shapely.MultiPolygon([multipolygon])
    layer = create_gpkg_polygon(
        path=os.path.join(path, f"{route_name}-{zone_id}_{name}.gpkg"),
        geometry_data=[[list(poly.exterior.coords)] + [list(ring.coords) for ring in poly.interiors] for poly in
                       multipolygon.geoms],
        crs=crs_utm)
    return layer


def create_gpkg_polygon(path=None, fields=None, geometry_data=None, crs=None, crs_code=None):
    if fields is None:
        fields = {}
    if not len(geometry_data):
        print(f"Empty Geometry for {path}: No export")
        return
    geometries = []
    for i, polygon in enumerate(geometry_data):
        rings = []
        for ring in polygon:
            points = [QgsPointXY(pos[0], pos[1]) for pos in ring]
            rings.append(points)
        line = [point.toQgsPointXY() if isinstance(point, QgsPoint) else point for point in rings]
        geometry = QgsGeometry.fromPolygonXY(line)
        geometries.append(geometry)
    if crs is None:
        crs = QgsCoordinateReferenceSystem(crs_code)
    fields2 = copy.deepcopy(fields)
    return create_gpkg(path, fields=fields2, geometries=geometries, crs=crs, layer_type="Polygon")


def create_gpkg(
        path=None, fields=None, geometry_data=None, geometry_func=None, geometries=None, crs=None, crs_code=None,
        layer_type=None,
):
    if crs is None or not crs.isValid():
        crs = QgsCoordinateReferenceSystem(crs_code or "EPSG:4326")
    # Default empty fields and geometries if none provided
    if fields is None:
        fields = {}
    if geometries is None and geometry_func is not None:
        geometries = geometry_func(geometry_data)
    elif geometries is None:
        raise Exception("No geometries provided. Either pass 'geometries' or a 'geometry_func'.")

    # Create a memory layer with the specified CRS
    if path is None:
        layer_name = "Virtual_layer"
    else:
        layer_name = Path(path).stem
    layer = QgsVectorLayer(f"{layer_type}?crs={crs.authid()}", layer_name, "memory")
    layer.setCrs(crs)
    provider = layer.dataProvider()

    # Define fields in the layer
    attributes_declaration = []

    for name, field_data in fields.items():
        if isinstance(field_data, dict) and "type" in field_data:
            attributes_declaration.append(QgsField(name, field_data["type"]))
        elif isinstance(field_data, list) and field_data:
            first_data = field_data[0]
            data_type = type(first_data)
            # print(f"{fields = }")
            # print(f"{name = }")
            # print(f"{field_data = }")
            # print(f"{data_type = }")
            if data_type in [int]:
                field_type = QVariant.Int
            elif data_type in [float]:
                field_type = QVariant.Double
            elif data_type in [str]:
                field_type = QVariant.String
            elif data_type is bool:
                field_type = QVariant.Bool
            else:
                msg = f"field {name}: Type {data_type} unknown. Data = {first_data}"
                print(msg)
                # raise Exception(msg)
                continue
            attributes_declaration.append(QgsField(name, field_type))
        else:
            print(f"Invalid format for field '{name}' with data: {field_data}")

    provider.addAttributes(attributes_declaration)
    layer.updateFields()

    # Populate layer with features and their geometries and attributes
    for i, geometry in enumerate(geometries):
        if not geometry or not geometry.isGeosValid():
            print(f"Skipping invalid geometry at index {i}")
            continue  # Skip invalid geometry

        feature = QgsFeature()
        feature.setGeometry(geometry)

        # Collect attributes based on fields structure
        attributes = []
        for name, field_data in fields.items():
            if isinstance(field_data, list):
                try:
                    attributes.append(field_data[i])
                except IndexError:
                    attributes.append(None)  # Handle missing data gracefully
            elif isinstance(field_data, dict) and "values" in field_data:
                try:
                    attributes.append(field_data["values"][i])
                except IndexError:
                    attributes.append(None)
            else:
                attributes.append(None)

        # Set attributes and add feature to the provider
        feature.setAttributes(attributes)
        provider.addFeature(feature)

    # Update the layer’s extents after adding all features
    layer.updateExtents()
    print(f"Feature count after adding: {layer.featureCount()}")
    print(f"Layer validity: {layer.isValid()}")

    if path is not None:
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = layer_name
        transform_context = QgsProject.instance().transformContext()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, path,
            transform_context,  # Replace with crs of layer because qgis is bugged
            options
        )
        if result[0] != QgsVectorFileWriter.NoError:
            raise Exception(f"Error writing GPKG file: {result[1]}")
        print(f"GPKG file created at: {path} with {layer.featureCount()} features")
        layer = QgsVectorLayer(path, layer_name)
    return layer


def reproject_utm(layer):
    crs_layer = get_crs(layer)
    crs_utm = get_crs_utm(layer)
    if crs_layer == crs_utm:
        return
    # TODO: Projection in UTM


def set_style(layer, path_style):
    if os.path.isfile(path_style):
        layer.loadNamedStyle(path_style)


def add_fields(layer, fields):
    for field, type_field in fields.items():
        type_attribute = {
            float: QVariant.Double,
            str: QVariant.String,
            bool: QVariant.Bool,
        }.get(type_field)
        layer.dataProvider().addAttributes([QgsField(field, type_attribute)])
        layer.updateFields()


def get_selected_layers():
    return iface.layerTreeView().selectedLayers()


def get_selected_layer():
    layers_selected = get_selected_layers()
    if not len(layers_selected):
        return None
    return layers_selected[0]


def add_layer(layer):
    if layer is None:
        print(f"Layer not added because layer is None")
        return
    crs = layer.crs()
    if not crs.isValid():
        print(f"Layer not added because CRS is Not Valid")
        return
    crs_extent = crs.bounds()
    if crs_extent.isEmpty():
        print(f"Layer not added because CRS Extend is Empty")
        return
    layer_extent = layer.extent()
    if not crs_extent.xMinimum() <= layer_extent.xMinimum() <= crs_extent.xMaximum() and \
            crs_extent.xMinimum() <= layer_extent.xMaximum() <= crs_extent.xMaximum():
        print(f"Layer not added because corrdinates outside bounds of extend on X")
        return
    if not crs_extent.yMinimum() <= layer_extent.yMinimum() <= crs_extent.yMaximum() and \
            crs_extent.yMinimum() <= layer_extent.yMaximum() <= crs_extent.yMaximum():
        print(f"Layer not added because corrdinates outside bounds of extend on Y")
        return
    QgsProject.instance().addMapLayer(layer)


def get_raster_value_at_xy(layer, x, y, band=1):
    if layer is None or not layer.isValid():
        return None

    provider = layer.dataProvider()
    if not provider:
        return None

    # Check if coordinates are within layer extent
    extent = layer.extent()
    if (x < extent.xMinimum() or x > extent.xMaximum() or
            y < extent.yMinimum() or y > extent.yMaximum()):
        return None

    # Get value at point
    result = provider.sample(QgsPointXY(x, y), band)

    # Check if the value is valid
    if result[1]:  # Success flag is True
        return result[0]
    else:
        return None


def get_raster_value(layer, x, y, radius=None, band=1):
    """
    Get the maximum raster value within a radius around the specified x,y coordinates.

    Parameters:
    - layer: QgsRasterLayer - The raster layer to sample
    - x, y: float - The coordinates to check
    - radius: float - The radius around the point to search for maximum value
    - band: int - The raster band to sample (default: 1)

    Returns:
    - float or None: The maximum value found within the radius, or None if no valid value found
    """
    if layer is None or not layer.isValid():
        return None

    provider = layer.dataProvider()
    if not provider:
        return None

    if radius is None:
        radius = get_raster_step_min(layer)

    max_value = None

    while max_value is None:

        extent = layer.extent()
        search_extent = QgsRectangle(
            x - radius, y - radius,
            x + radius, y + radius
        )

        # Clip search extent to layer extent
        search_extent = search_extent.intersect(extent)

        # If after clipping the search extent is empty, return None
        if search_extent.isEmpty():
            return None

        # Calculate the number of pixels to check based on layer resolution
        pixel_size_x = layer.rasterUnitsPerPixelX()
        pixel_size_y = layer.rasterUnitsPerPixelY()

        # Determine the number of points to check in each direction
        x_points = max(1, int(radius / pixel_size_x))
        y_points = max(1, int(radius / pixel_size_y))

        # Check all points in the grid
        for i in range(-x_points, x_points + 1):
            for j in range(-y_points, y_points + 1):
                # Calculate point coordinates
                point_x = x + i * pixel_size_x
                point_y = y + j * pixel_size_y

                # Check if this point is within the radius
                distance = ((point_x - x) ** 2 + (point_y - y) ** 2) ** 0.5
                if distance > radius:
                    continue

                # Sample the value at this point
                result = provider.sample(QgsPointXY(point_x, point_y), band)

                # Check if the value is valid and update max if needed
                if result[1]:  # Success flag is True
                    if max_value is None or result[0] > max_value:
                        max_value = result[0]
        radius *= 2

    return max_value


def check_tol_z(xi, yi, xj, yj, tol_xy, tol_z, raster, step):
    dist = ((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5
    m = int(dist / step) + 1
    print(f"{m = }")
    if m < 3:
        print("Skip because too close")
        return None
    ratios = numpy.linspace(0, 1, m)[1:-1]
    x_interp = xi + ratios * (xj - xi)
    y_interp = yi + ratios * (yj - yi)
    zi = get_raster_value(raster, xi, yi, radius=tol_xy)
    zj = get_raster_value(raster, xj, yj, radius=tol_xy)
    z_interp_line = zi + ratios * (zj - zi)
    z_interp = [get_raster_value(raster, x_interp[i], y_interp[i], radius=tol_xy) for i in range(m - 2)]
    dists_to_ij_z = numpy.array(z_interp) - z_interp_line
    abs_dists_to_ij_z = abs(dists_to_ij_z)
    max_dist = numpy.max(abs_dists_to_ij_z)
    if max_dist > tol_z:
        index_point = numpy.argmax(abs_dists_to_ij_z)
        print(f"{index_point = }")
        x_new = x_interp[index_point]
        y_new = y_interp[index_point]
        return x_new, y_new
    return None


def rec_insert_waypoint(waypoints, tol_xy, tol_z, step, raster, index=0):
    """Ajoute des waypoints sur la route en fonction de la variation du terrain"""
    if index == len(waypoints)-1:
        return waypoints
    xi, yi = waypoints[index]
    xj, yj = waypoints[index+1]
    next_waypoint = check_tol_z(xi, yi, xj, yj, tol_xy, tol_z, raster, step)
    if next_waypoint is None:
        return rec_insert_waypoint(waypoints, tol_xy, tol_z, step, raster, index+1)
    waypoints = numpy.insert(waypoints, index+1, next_waypoint, axis=0)
    return rec_insert_waypoint(waypoints, tol_xy, tol_z, step, raster, index)
