"""Export formatted data to GEOTIFF format

This module needs to be executed with PyQgis.

Please refer to _write_geotiff for information about the arguments
"""
from __future__ import annotations

import contextlib
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qgis._core import QgsRasterLayer

if TYPE_CHECKING:
    import numpy.typing as npt


with contextlib.suppress(ImportError):
    import qgis.core

NO_VALUE_DATA = 0


def _get_raster_writer(filename: Path) -> qgis.core.QgsRasterFileWriter:
    """https://qgis.org/pyqgis/3.22/core/QgsRasterFileWriter.html"""
    return qgis.core.QgsRasterFileWriter(str(filename))  # Create the writer object


def _get_raster_provider(  # noqa: PLR0913
    writer: qgis.core.QgsRasterFileWriter,
    width: int,
    height: int,
    x_data: npt.NDArray[np.float_],
    y_data: npt.NDArray[np.float_],
    crs
) -> qgis.core.QgsRasterDataProvider:
    provider = writer.createOneBandRaster(
        # Create the provider object
        # https://qgis.org/pyqgis/3.22/core/QgsRasterDataProvider.html#qgis.core.QgsRasterDataProvider
        dataType=qgis.core.Qgis.Float64,
        width=width,  # Number of points of X axis
        height=height,  # Number of points of Y axis
        # Defining bounding box https://qgis.org/pyqgis/3.2/core/other/QgsRectangle.html
        extent=qgis.core.QgsRectangle(
            x_data.min(),
            y_data.min(),
            x_data.max(),
            y_data.max(),
        ),
        crs=crs,  # Attribution of EPSG code
    )
    provider.setNoDataValue(1, NO_VALUE_DATA)
    provider.setEditable(enabled=True)  # Enable modification and start writing on the file
    return provider


def _get_block(
    provider: qgis.core.QgsRasterDataProvider,
    band_number: int,
) -> qgis.core.QgsRasterBlock:

    return provider.block(
        # Creating a block https://qgis.org/pyqgis/3.22/core/QgsRasterBlock.html#qgis.core.QgsRasterBlock
        bandNo=band_number,  # Band number (single band here)
        boundingBox=provider.extent(),  # Get the QgsRectangle from extend of the provider
        width=provider.xSize(),  # Get the width of the provider
        height=provider.ySize(),  # Get the height of the provider
    )


def _set_block(
        block: qgis.core.QgsRasterBlock,
        provider: qgis.core.QgsRasterDataProvider,
        z_data: np.ndarray,
        band_number: int,
) -> None:
    expected_shape = (block.height(), block.width())  # Expected (H, W)

    # Auto-fix mismatch by transposing if needed
    if z_data.shape != expected_shape:
        if z_data.shape[::-1] == expected_shape:  # Check if transposing fixes it
            print("Warning: Mismatch detected, transposing array to match block dimensions.")
            z_data = z_data.T
        else:
            raise ValueError(
                f"Shape mismatch: Expected {expected_shape}, but got {z_data.shape}. "
                "Cannot automatically fix."
            )

    # Ensure data type and contiguous memory layout
    z_data = np.ascontiguousarray(z_data, dtype=np.float64)

    # Convert to bytes and set data
    data = z_data.tobytes()
    block.setData(data)  # Should no longer crash

    # Write block to provider
    provider.writeBlock(block, band_number, 0, 0)


def write_geotiff(
    x_data: npt.NDArray[np.float_],
    y_data: npt.NDArray[np.float_],
    z_data: npt.NDArray[np.float_],
    file_path: Path,
    crs
) -> QgsRasterLayer:
    """Main function to write a GEOTIFF file

    :param x_data: X nodes
    :param y_data: Y nodes
    :param z_data: 2D array containing the data to export
    :param file_path: Path to create the GEOTIFF file. Should include filename.
    :param crs: EPSG code for the X and Y nodes
    :param multi_band: If True, the data will be exported as a multi band GEOTIFF file.
    """
    qgis.core.QgsApplication.setPrefixPath("/usr", useDefaultPaths=True)
    nx, ny = x_data.size, y_data.size
    nz = z_data.shape

    if (ny, nx) != nz:
        msg = f"Dimension Error of Z data:\n\tX: {nx}\n\tY: {ny}\n\tZ: {nz} expected {(nx, ny)}"
        raise TypeError(msg)

    writer = _get_raster_writer(file_path)
    provider = _get_raster_provider(
        writer,
        nx,
        ny,
        x_data,
        y_data,
        crs
    )
    z_data[np.isnan(z_data)] = NO_VALUE_DATA
    block = _get_block(provider, band_number=1)
    _set_block(
        block,
        provider,
        np.array(z_data, dtype=np.float64),
        band_number=1,
    )
    provider.setEditable(enabled=False)  # Disable modification and stop writing on the file
    layer = QgsRasterLayer(str(file_path), file_path.stem, "gdal")
    print(f"Create raster {layer} at {file_path}")
    return layer
