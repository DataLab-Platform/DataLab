# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""Make a WiX Toolset .wxs file for the DataLab Windows installer."""

# TODO: Localization?
# TODO: Uninstall previous version is not working.
# TODO: Remove everything regarding NSIS over the whole project.
# TODO: Icon for the installer?

import os
import os.path as osp
import uuid
import xml.etree.ElementTree as ET

from cdl import __version__

COUNT = 0


def generate_id() -> str:
    """Generate an ID for a WiX Toolset XML element."""
    global COUNT
    COUNT += 1
    return f"ID_{COUNT:04d}"


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
            relpath = osp.relpath(osp.join(root, dpath), root_dir)
            dir_ids[relpath] = generate_id()
            files_dict.setdefault(osp.dirname(relpath), [])
        for filename in filenames:
            relpath = osp.relpath(osp.join(root, filename), root_dir)
            file_ids[relpath] = generate_id()
            files_dict.setdefault(osp.dirname(relpath), []).append(relpath)

    # Create the base directory structure in XML:
    base_name = osp.basename(dist_dir)
    base_id = dir_ids[osp.relpath(dist_dir, root_dir)] = generate_id()
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
    dir_str = ET.tostring(dir_xml, encoding="unicode").replace("><", ">\n<")
    # print("Directory structure:\n", dir_str)

    # Create additionnal components for each file in the directory structure:
    comp_str_list: list[str] = []
    for dpath in sorted(dir_ids.keys())[1:]:
        did = dir_ids[dpath]
        files = files_dict.get(dpath, [])
        if files:
            # This is a directory with files, so we need to create components:
            for path in files:
                fid = file_ids[path]
                guid = str(uuid.uuid4())
                comp_xml = ET.Element("Component", Id=fid, Directory=did, Guid=guid)
                ET.SubElement(comp_xml, "File", Source=path, KeyPath="yes")
                comp_str_list.append(ET.tostring(comp_xml, encoding="unicode"))
        else:
            # This is an empty directory, so we need to create a folder:
            guid = str(uuid.uuid4())
            cdid = f"CreateFolder_{did}"
            comp_xml = ET.Element("Component", Id=cdid, Directory=did, Guid=guid)
            ET.SubElement(comp_xml, "CreateFolder")
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
