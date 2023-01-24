# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)


"""
CodraFT Image I/O module
"""

import os
import re
import struct
import time

import numpy as np
from guiqwt.io import _imread_pil, _imwrite_pil, iohandler

from codraft.config import _
from codraft.utils.misc import to_string


# ==============================================================================
# SIF I/O functions
# ==============================================================================
# Original code:
# --------------
# Zhenpeng Zhou <zhenp3ngzhou cir{a} gmail dot com>
# Copyright 2017 Zhenpeng Zhou
# Licensed under MIT License Terms
#
# Changes:
# -------
# * Calculating header length using the line beginning with "Counts"
# * Calculating wavelenght info line number using line starting with "65538 "
# * Handling wavelenght info line ending with "NM"
# * Calculating data offset by detecting the first line containing NUL character after
#   header
#
class SIFFile:
    """
    A class that reads the contents and metadata of an Andor .sif file.
    Compatible with images as well as spectra.
    Exports data as numpy array or xarray.DataArray.

    Example: SIFFile('my_spectrum.sif').read_all()

    In addition to the raw data, SIFFile objects provide a number of meta
    data variables:
    :ivar x_axis: the horizontal axis (can be pixel numbers or wvlgth in nm)
    :ivar original_filename: the original file name of the .sif file
    :ivar date: the date the file was recorded
    :ivar model: camera model
    :ivar temperature: sensor temperature in degrees Celsius
    :ivar exposuretime: exposure time in seconds
    :ivar cycletime: cycle time in seconds
    :ivar accumulations: number of accumulations
    :ivar readout: pixel readout rate in MHz
    :ivar xres: horizontal resolution
    :ivar yres: vertical resolution
    :ivar width: image width
    :ivar height: image height
    :ivar xbin: horizontal binning
    :ivar ybin: vertical binning
    :ivar gain: EM gain level
    :ivar vertical_shift_speed: vertical shift speed
    :ivar pre_amp_gain: pre-amplifier gain
    :ivar stacksize: number of frames
    :ivar filesize: size of the file in bytes
    :ivar m_offset: offset in the .sif file to the actual data
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-statements

    def __init__(self, filepath):
        self.filepath = filepath
        self.original_filename = None
        self.filesize = None
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        self.width = None
        self.height = None
        self.grating = None
        self.stacksize = None
        self.datasize = None
        self.xres = None
        self.yres = None
        self.xbin = None
        self.ybin = None
        self.cycletime = None
        self.pre_amp_gain = None
        self.temperature = None
        self.center_wavelength = None
        self.readout = None
        self.gain = None
        self.date = None
        self.exposuretime = None
        self.m_offset = None
        self.accumulations = None
        self.vertical_shift_speed = None
        self.model = None
        self.grating_blaze = None
        self._read_header(filepath)

    def __repr__(self):
        info = (
            ("Original Filename", self.original_filename),
            ("Date", self.date),
            ("Camera Model", self.model),
            ("Temperature (deg.C)", f"{self.temperature:f}"),
            ("Exposure Time", f"{self.exposuretime:f}"),
            ("Cycle Time", f"{self.cycletime:f}"),
            ("Number of accumulations", f"{self.accumulations:d}"),
            ("Pixel Readout Rate (MHz)", f"{self.readout:f}"),
            ("Horizontal Camera Resolution", f"{self.xres:d}"),
            ("Vertical Camera Resolution", f"{self.yres:d}"),
            ("Image width", f"{self.width:d}"),
            ("Image Height", f"{self.height:d}"),
            ("Horizontal Binning", f"{self.xbin:d}"),
            ("Vertical Binning", f"{self.ybin:d}"),
            ("EM Gain level", f"{self.gain:f}"),
            ("Vertical Shift Speed", f"{self.vertical_shift_speed:f}"),
            ("Pre-Amplifier Gain", f"{self.pre_amp_gain:f}"),
            ("Stacksize", f"{self.stacksize:d}"),
            ("Filesize", f"{self.filesize:d}"),
            ("Offset to Image Data", f"{self.m_offset:f}"),
        )
        desc_len = max([len(d) for d in list(zip(*info))[0]]) + 3
        res = ""
        for description, value in info:
            res += ("{:" + str(desc_len) + "}{}\n").format(description + ": ", value)

        res = object.__repr__(self) + "\n" + res
        return res

    def _read_header(self, filepath):
        """Read SIF file header"""
        with open(filepath, "rb") as sif_file:
            i_wavelength_info = None
            headerlen = None
            i = 0
            self.m_offset = 0
            while True:
                raw_line = sif_file.readline()
                line = raw_line.strip()
                if i == 0:
                    if line != b"Andor Technology Multi-Channel File":
                        sif_file.close()
                        raise Exception(f"{filepath} is not an Andor SIF file")
                elif i == 2:
                    tokens = line.split()
                    self.temperature = float(tokens[5])
                    self.date = time.strftime("%c", time.localtime(float(tokens[4])))
                    self.exposuretime = float(tokens[12])
                    self.cycletime = float(tokens[13])
                    self.accumulations = int(tokens[15])
                    self.readout = 1 / float(tokens[18]) / 1e6
                    self.gain = float(tokens[21])
                    self.vertical_shift_speed = float(tokens[41])
                    self.pre_amp_gain = float(tokens[43])
                elif i == 3:
                    self.model = to_string(line)
                elif i == 5:
                    self.original_filename = to_string(line)
                if i_wavelength_info is None and i > 7:
                    if line.startswith(b"65538 ") and len(line) == 17:
                        i_wavelength_info = i + 1
                if i_wavelength_info is not None and i == i_wavelength_info:
                    wavelength_info = line.split()
                    self.center_wavelength = float(wavelength_info[3])
                    self.grating = float(wavelength_info[6])
                    blaze = wavelength_info[7]
                    if blaze.endswith(b"NM"):
                        blaze = blaze[:-2]
                    self.grating_blaze = float(blaze)
                if headerlen is None:
                    if line.startswith(b"Counts"):
                        headerlen = i + 3
                else:
                    if i == headerlen - 2:
                        if line[:12] == b"Pixel number":
                            line = line[12:]
                        tokens = line.split()
                        if len(tokens) < 6:
                            raise Exception("Not able to read stacksize.")
                        self.yres = int(tokens[2])
                        self.xres = int(tokens[3])
                        self.stacksize = int(tokens[5])
                    elif i == headerlen - 1:
                        tokens = line.split()
                        if len(tokens) < 7:
                            raise Exception("Not able to read Image dimensions.")
                        self.left = int(tokens[1])
                        self.top = int(tokens[2])
                        self.right = int(tokens[3])
                        self.bottom = int(tokens[4])
                        self.xbin = int(tokens[5])
                        self.ybin = int(tokens[6])
                    elif i >= headerlen:
                        if b"\x00" in line:
                            break
                i += 1
                self.m_offset += len(raw_line)

        width = self.right - self.left + 1
        mod = width % self.xbin
        self.width = int((width - mod) / self.ybin)
        height = self.top - self.bottom + 1
        mod = height % self.ybin
        self.height = int((height - mod) / self.xbin)

        self.filesize = os.path.getsize(filepath)
        self.datasize = self.width * self.height * 4 * self.stacksize

    def read_all(self):
        """
        Returns all blocks (i.e. frames) in the .sif file as a numpy array.
        :return: a numpy array with shape (blocks, y, x)
        """
        with open(self.filepath, "rb") as sif_file:
            sif_file.seek(self.m_offset)
            block = sif_file.read(self.width * self.height * self.stacksize * 4)
            data = np.fromstring(block, dtype=np.float32)
        return data.reshape(self.stacksize, self.height, self.width)


def imread_sif(filename):
    """Open a SIF image"""
    sif_file = SIFFile(filename)
    return sif_file.read_all()


# ==============================================================================
# SPIRICON I/O functions
# ==============================================================================


class SCORFile:
    """Object representing a SPIRICON .scor-data file"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = None
        self.width = None
        self.height = None
        self.m_offset = None
        self.filesize = None
        self.datasize = None
        self._read_header()

    def __repr__(self):
        info = (
            ("Image width", f"{self.width:d}"),
            ("Image Height", f"{self.height:d}"),
            ("Filesize", f"{self.filesize:d}"),
            ("Datasize", f"{self.datasize:d}"),
            ("Offset to Image Data", f"{self.m_offset:f}"),
        )
        desc_len = max([len(d) for d in list(zip(*info))[0]]) + 3
        res = ""
        for description, value in info:
            res += ("{:" + str(desc_len) + "}{}\n").format(description + ": ", value)

        res = object.__repr__(self) + "\n" + res
        return res

    def _read_header(self):
        """Read file header"""
        with open(self.filepath, "rb") as data_file:
            metadata = {}
            key1 = None
            while True:
                bline = data_file.readline().strip()
                key1_match = re.match(b"\\[(\\S*)\\]", bline)
                if key1_match is not None:
                    key1 = key1_match.groups()[0].decode()
                    metadata[key1] = {}
                elif b"=" in bline:
                    key2, value = bline.decode().split("=")
                    metadata[key1][key2] = value
                else:
                    break

        capture_size = metadata["Capture"]["CaptureSize"]
        self.width, self.height = [int(val) for val in capture_size.split(",")]

        self.filesize = os.path.getsize(self.filepath)
        self.datasize = self.width * self.height * 2
        self.m_offset = self.filesize - self.datasize - 8

    def read_all(self):
        """Read all data"""
        with open(self.filepath, "rb") as data_file:
            data_file.seek(self.m_offset)
            block = data_file.read(self.datasize)
            data = np.fromstring(block, dtype=np.int16)
        return data.reshape(self.height, self.width)


def imread_scor(filename):
    """Open a SPIRICON image"""
    scor_file = SCORFile(filename)
    return scor_file.read_all()


# ==============================================================================
# FXD I/O functions
# ==============================================================================


class FXDFile:
    """Class implementing FXD Image file reading feature"""

    HEADER = "<llllllffl"

    def __init__(self, fname=None, debug=False):
        self.__debug = debug
        self.file_format = None  # long
        self.nbcols = None  # long
        self.nbrows = None  # long
        self.nbframes = None  # long
        self.pixeltype = None  # long
        self.quantlevels = None  # long
        self.maxlevel = None  # float
        self.minlevel = None  # float
        self.comment_length = None  # long
        self.fname = None
        self.data = None
        if fname is not None:
            self.load(fname)

    def __repr__(self):
        info = (
            ("Image width", f"{self.nbcols:d}"),
            ("Image Height", f"{self.nbrows:d}"),
            ("Frame number", f"{self.nbframes:d}"),
            ("File format", f"{self.file_format:d}"),
            ("Pixel type", f"{self.pixeltype:d}"),
            ("Quantlevels", f"{self.quantlevels:d}"),
            ("Min. level", f"{self.minlevel:f}"),
            ("Max. level", f"{self.maxlevel:f}"),
            ("Comment length", f"{self.comment_length:d}"),
        )
        desc_len = max([len(d) for d in list(zip(*info))[0]]) + 3
        res = ""
        for description, value in info:
            res += ("{:" + str(desc_len) + "}{}\n").format(description + ": ", value)

        res = object.__repr__(self) + "\n" + res
        return res

    def load(self, fname):
        """Load header and image pixel data"""
        with open(fname, "rb") as data_file:
            header_s = struct.Struct(self.HEADER)
            record = data_file.read(9 * 4)
            unpacked_rec = header_s.unpack(record)
            (
                self.file_format,
                self.nbcols,
                self.nbrows,
                self.nbframes,
                self.pixeltype,
                self.quantlevels,
                self.maxlevel,
                self.minlevel,
                self.comment_length,
            ) = unpacked_rec
            if self.__debug:
                print(unpacked_rec)
                print(self)
            data_file.seek(128 + self.comment_length)
            if self.pixeltype == 0:
                size, dtype = 4, np.float32
            elif self.pixeltype == 1:
                size, dtype = 2, np.uint16
            elif self.pixeltype == 2:
                size, dtype = 1, np.uint8
            else:
                raise NotImplementedError(f"Unsupported pixel type: {self.pixeltype}")
            block = data_file.read(self.nbrows * self.nbcols * size)
        data = np.fromstring(block, dtype=dtype)
        self.data = data.reshape(self.nbrows, self.nbcols)


def imread_fxd(filename):
    """Open an FXD image"""
    fxd_file = FXDFile(filename)
    return fxd_file.data


def imread_xyz(filename):
    """Open XYZ file image and return a NumPy array"""
    with open(filename, "rb") as fdesc:
        cols = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
        rows = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
        arr = np.fromfile(fdesc, dtype=np.uint16, count=cols * rows)
        arr = arr.reshape((rows, cols))
    return np.fliplr(arr)


# ==============================================================================
# Registering I/O functions
# ==============================================================================
iohandler.add(_("Andor SIF files"), "*.sif", read_func=imread_sif)
iohandler.add(_("SPIRICON files"), "*.scor-data", read_func=imread_scor)
iohandler.add(_("FXD files"), "*.fxd", read_func=imread_fxd)
iohandler.add(_("XYZ files"), "*.xyz", read_func=imread_xyz)
iohandler.add(
    _("Bitmap images"),
    "*.bmp",
    read_func=_imread_pil,
    write_func=_imwrite_pil,
    data_types=(np.uint8,),
)
