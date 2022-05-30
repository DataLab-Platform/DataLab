# -*- coding: utf-8 -*-
#
# Copyright © 2022 Codra
# Pierre Raybaut

"""
Module checking dependencies with respect to a reference
"""

import hashlib
import os
import os.path as osp
import sys

from guidata.utils import get_module_path

BUFFER_SIZE = 65536


def get_file_hash(filename, dhash=None, bufsize=None):
    """Return SHA256 hash for file"""
    if dhash is None:
        dhash = hashlib.sha256()
    buffer = bytearray(128 * 1024 if bufsize is None else bufsize)
    # using a memoryview so that we can slice the buffer without copying it
    buffer_view = memoryview(buffer)
    with open(filename, "rb", buffering=0) as fobj:
        while True:
            n_chunk = fobj.readinto(buffer_view)
            if not n_chunk:
                break
            dhash.update(buffer_view[:n_chunk])
    return dhash


def get_directory_hash(directory, extensions=None, verbose=False):
    """Return SHA256 hash for whole directory"""
    dhash = hashlib.sha1()
    if not os.path.exists(directory):
        return -1

    assert extensions is None or isinstance(extensions, (tuple, list))
    extlist = []
    if extensions is not None:
        for ext in extensions:
            if not ext.startswith("."):
                ext = "." + ext
            extlist.append(ext)
    filenb = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if extensions is None or osp.splitext(name)[1] in extlist:
                filepath = os.path.join(root, name)
                if verbose:
                    filenb += 1
                    print(f"[{filenb:07d}] Hashing: {filepath}")
                dhash = get_file_hash(filepath, dhash)
    return dhash.hexdigest()


def get_module_hash(module_name, verbose=False):
    """Return SHA256 hash for Python module"""
    path = get_module_path(module_name)
    return get_directory_hash(path, extensions=(".py",), verbose=verbose)


def get_dependencies_hash(dependencies):
    """Return SHA256 hash for Python module dependencies"""
    return {name: get_module_hash(name) for name in dependencies}


DEPFILENAME = f"dependencies-py{sys.version_info.major}.txt"


def check_dependencies_hash(datapath, hash_dict=None):
    """Check dependencies hash dictionnary"""
    if hash_dict is None:
        hash_dict = {}
        with open(osp.join(datapath, DEPFILENAME), "rb") as filehandle:
            for line in filehandle.readlines():
                key, value = line.decode("utf-8").strip().split(":")
                hash_dict[key] = value
    return {name: get_module_hash(name) == hash_dict[name] for name in hash_dict}


def create_dependencies_file(datapath, dependencies):
    """Create dependencies.txt file"""
    hdict = get_dependencies_hash(dependencies)
    with open(osp.join(datapath, DEPFILENAME), "wb") as filehandle:
        for name in hdict:
            line = name + ":" + hdict[name] + os.linesep
            filehandle.write(line.encode("utf-8"))


if __name__ == "__main__":
    # datapath = osp.join(osp.dirname(__file__), os.pardir, "data")
    # hdict = get_dependencies_hash(("guidata", "guiqwt"))
    # print(hdict)
    # print(check_dependencies_hash(datapath, hdict))
    create_dependencies_file(
        osp.join(osp.dirname(__file__), os.pardir, "data"), ("guidata", "guiqwt")
    )
    # print(check_dependencies_hash(datapath))
