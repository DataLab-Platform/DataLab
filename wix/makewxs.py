# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""Make a WiX Toolset .wxs file for the DataLab Windows installer.

History:

- Version 1.0.0 (2024-04-01)
    Initial version.
"""

# TODO: Localization, eventually.

import argparse
import os
import os.path as osp
import uuid
import xml.etree.ElementTree as ET

COUNT = 0


def generate_id() -> str:
    """Generate an ID for a WiX Toolset XML element."""
    global COUNT
    COUNT += 1
    return f"ID_{COUNT:04d}"


def insert_text_after(text: str, containing: str, content: str) -> str:
    """Insert line of text after the line containing a specific text."""
    if os.linesep in content:
        linesep = os.linesep
    elif "\r\n" in content:
        linesep = "\r\n"
    else:
        linesep = "\n"
    lines = content.splitlines()
    for i_line, line in enumerate(lines):
        if containing in line:
            lines.insert(i_line + 1, text)
            break
    return linesep.join(lines)


def make_wxs(product_name: str, version: str) -> None:
    """Make a .wxs file for the DataLab Windows installer."""
    wix_dir = osp.dirname(__file__)
    proj_dir = osp.join(wix_dir, os.pardir)
    dist_dir = osp.join(proj_dir, "dist", product_name)
    wxs_path = osp.join(wix_dir, f"generic-{product_name}.wxs")
    output_path = osp.join(wix_dir, f"{product_name}-{version}.wxs")

    dir_ids: dict[str, str] = {}
    file_ids: dict[str, str] = {}

    files_dict: dict[str, list[str]] = {}

    dir_str_list: list[str] = []
    comp_str_list: list[str] = []

    for pth in os.listdir(dist_dir):
        root_dir = osp.join(dist_dir, pth)

        if not osp.isdir(root_dir):
            continue

        for root, dirs, filenames in os.walk(root_dir):
            for dpath in dirs:
                relpath = osp.relpath(osp.join(root, dpath), proj_dir)
                dir_ids[relpath] = generate_id()
                files_dict.setdefault(osp.dirname(relpath), [])
            for filename in filenames:
                relpath = osp.relpath(osp.join(root, filename), proj_dir)
                file_ids[relpath] = generate_id()
                files_dict.setdefault(osp.dirname(relpath), []).append(relpath)

        # Create the base directory structure in XML:
        base_name = osp.basename(root_dir)
        base_path = osp.relpath(root_dir, proj_dir)
        base_id = dir_ids[base_path] = generate_id()
        dir_xml = ET.Element("Directory", Id=base_id, Name=base_name)

        # Nesting directories, recursively, in XML:
        for dpath in sorted(dir_ids.keys())[1:]:
            dname = osp.basename(dpath)
            parent = dir_xml
            for element in parent.iter():
                if element.get("Id") == dir_ids[osp.dirname(dpath)]:
                    parent = element
                    break
            else:
                raise ValueError(f"Parent directory not found for {dpath}")
            ET.SubElement(parent, "Directory", Id=dir_ids[dpath], Name=dname)
        space = " " * 4
        ET.indent(dir_xml, space=space, level=4)
        dir_str_list.append(space * 4 + ET.tostring(dir_xml, encoding="unicode"))

        # Create additionnal components for each file in the directory structure:
        for dpath in sorted(dir_ids.keys()):
            did = dir_ids[dpath]
            files = files_dict.get(dpath, [])
            if files:
                # This is a directory with files, so we need to create components:
                for path in files:
                    fid = file_ids[path]
                    guid = str(uuid.uuid4())
                    comp_xml = ET.Element("Component", Id=fid, Directory=did, Guid=guid)
                    ET.SubElement(comp_xml, "File", Source=path, KeyPath="yes")
                    ET.indent(comp_xml, space=space, level=3)
                    comp_str = space * 3 + ET.tostring(comp_xml, encoding="unicode")
                    comp_str_list.append(comp_str)
            elif dpath != base_path:
                # This is an empty directory, so we need to create a folder:
                guid = str(uuid.uuid4())
                cdid = f"CreateFolder_{did}"
                comp_xml = ET.Element("Component", Id=cdid, Directory=did, Guid=guid)
                ET.SubElement(comp_xml, "CreateFolder")
                ET.indent(comp_xml, space=space, level=3)
                comp_str = space * 3 + ET.tostring(comp_xml, encoding="unicode")
                comp_str_list.append(comp_str)

    dir_str = "\n".join(dir_str_list).replace("><", ">\n<")
    # print("Directory structure:\n", dir_str)
    comp_str = "\n".join(comp_str_list).replace("><", ">\n<")
    # print("Component structure:\n", comp_str)

    # Create the .wxs file:
    with open(wxs_path, "r", encoding="utf-8") as fd:
        wxs = fd.read()
    wxs = insert_text_after(dir_str, "<!-- Automatically inserted directories -->", wxs)
    wxs = insert_text_after(comp_str, "<!-- Automatically inserted components -->", wxs)
    wxs = wxs.replace("{version}", version)
    with open(output_path, "w", encoding="utf-8") as fd:
        fd.write(wxs)
    print("Modified .wxs file has been created:", output_path)


def run() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Make a WiX Toolset .wxs file for the DataLab Windows installer."
    )
    parser.add_argument("product_name", help="Product name")
    parser.add_argument("version", help="Product version")
    args = parser.parse_args()
    make_wxs(args.product_name, args.version)


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # For testing purposes:
        make_wxs("DataLab", "0.14.2")
    else:
        run()

    # After making the .wxs file, run the following command to create the .msi file:
    #   wix build .\wix\DataLab.wxs -ext WixToolset.UI.wixext
