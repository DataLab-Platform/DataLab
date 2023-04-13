import configparser as cp
import os
import os.path as osp
import re


def generate_requirement_tables():
    """Generate CobraDataLab install requirements RST table.
    This table is inserted into 'installation.rst' when
    building documentation"""
    path = osp.dirname(__file__)
    config = cp.ConfigParser()
    config.read(osp.join(path, os.pardir, "setup.cfg"))
    ireq = config["options"]["install_requires"].strip().splitlines(False)
    requirements = [
        ".. list-table::",
        "    :header-rows: 1",
        "",
        "    * - Name",
        "      - Version (min.)",
    ]
    ireq = ["Python>=3.8", "PyQt=5.15"] + ireq
    for req in ireq:
        mod, _comp, ver = re.split("(>=|<=|=|<|>)", req)
        requirements.append("    * - " + mod)
        requirements.append("      - " + ver)
    with open(osp.join(path, "install_requires.txt"), "w") as fdesc:
        fdesc.write("\n".join(requirements))


if __name__ == "__main__":
    generate_requirement_tables()
