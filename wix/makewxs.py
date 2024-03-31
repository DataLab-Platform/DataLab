# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""Make a WiX Toolset .wxs file for the DataLab Windows installer."""

import os
import os.path as osp
import uuid
import xml.etree.ElementTree as ET

from cdl import __version__


def get_guid() -> str:
    """Generate a unique GUID."""
    return f"ID_{uuid.uuid4().hex}"


def make_wxs() -> None:
    """Make a .wxs file for the DataLab Windows installer."""
    wix_dir = osp.dirname(__file__)
    root_dir = osp.join(wix_dir, os.pardir)
    dist_dir = osp.join(root_dir, "dist", "DataLab", "_internal")
    wxs_path = osp.join(wix_dir, "DataLab-generic.wxs")
    output_path = osp.join(wix_dir, "DataLab.wxs")

    dir_ids: dict[str, str] = {}
    file_ids: dict[str, str] = {}

    files_dict: dict[str, list[str]] = {}
    for root, dirs, filenames in os.walk(dist_dir):
        for dpath in dirs:
            relpath = osp.relpath(osp.join(root, dpath))
            dir_ids[relpath] = get_guid()
            files_dict.setdefault(osp.dirname(relpath), [])
        for filename in filenames:
            relpath = osp.relpath(osp.join(root, filename))
            file_ids[relpath] = get_guid()
            files_dict.setdefault(osp.dirname(relpath), []).append(relpath)

    dir_xml = None
    # Nesting directories, recursively, in XML:
    for dpath in sorted(files_dict.keys()):
        dname = osp.basename(dpath)
        if dir_xml is None:
            dir_ids[dpath] = get_guid()
            dir_xml = ET.Element("Directory", Id=dir_ids[dpath], Name=dname)
        else:
            parent = dir_xml
            for element in parent.iter():
                if element.get("Id") == dir_ids[osp.dirname(dpath)]:
                    parent = element
                    break
            else:
                raise ValueError(f"Parent directory not found for {dpath}")
            ET.SubElement(parent, "Directory", Id=dir_ids[dpath], Name=dname)
    dir_str = ET.tostring(dir_xml, encoding="unicode").replace("><", ">\n<")
    # print("Directory structure:\n", dir_str)

    # Create additionnal components for each directory:
    comp_str_list: list[str] = []
    for dpath, files in files_dict.items():
        did = dir_ids[dpath]
        for path in files:
            fid = file_ids[path]
            guid = str(uuid.uuid4())
            comp_xml = ET.Element("Component", Id=fid, Directory=did, Guid=guid)
            ET.SubElement(comp_xml, "File", Source=path, KeyPath="yes")
            comp_str_list.append(ET.tostring(comp_xml, encoding="unicode"))
    comp_str = "\n".join(comp_str_list).replace("><", ">\n<")
    # print("Component structure:\n", comp_str)

    # Read the .wxs file:
    with open(wxs_path, "r", encoding="utf-8") as fd:
        wxs_content = fd.read()

    # Insert the directory structure into the .wxs file:
    start = wxs_content.find('<Directory Id="INSTALLFOLDER" Name="DataLab">')
    end = wxs_content.find("</Directory>", start)
    wxs_content = wxs_content[:end] + dir_str + "\n" + wxs_content[end:]
    # Insert the component structure into the .wxs file:
    start = wxs_content.find(
        r'<File Source=".\dist\DataLab\DataLab.exe" KeyPath="yes" />'
    )
    start = wxs_content.find("</Component>", start) + len("</Component>")
    wxs_content = wxs_content[:start] + "\n" + comp_str + wxs_content[start:]

    # Replace the version number in the .wxs file:
    wxs_content = wxs_content.replace("{version}", __version__)

    # Write the modified .wxs file:
    with open(output_path, "w", encoding="utf-8") as fd:
        fd.write(wxs_content)

    print("Modified .wxs file has been created:", output_path)


if __name__ == "__main__":
    make_wxs()
    # After making the .wxs file, run the following command to create the .msi file:
    #   wix build .\wix\DataLab.wxs -ext WixToolset.UI.wixext
