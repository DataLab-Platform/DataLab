# -*- coding: utf-8 -*-
#
# Distributed under the BSD 3-clause license (see guidata/LICENSE for details).

"""Script to compile translation files using Babel.

This script uses the `pybabel` command-line tool to compile translation files (.po) into
binary format (.mo) for use in applications.
"""

import os
import subprocess
import typing


def postprocess(
    name: str,
    directory: typing.Union[str, os.PathLike],
) -> None:
    """Compile translated strings using Babel.

    This function performs the following steps:
    1. Check the project root directory.
    2. Check the Babel configuration file.
    3. Verify the presence of the translation directory.
    4. Compile the translation files.

    Args:
        name: The name of the project.
        directory: The root directory of the project.

    Raises:
        FileNotFoundError: The Babel configuration file or the translation directory
        does not exist.
        RuntimeError: The compilation failed.

    """
    # Check the project root directory
    if not os.path.exists(directory):
        print(f"Error: Project root directory {directory} does not exist.")
        raise FileNotFoundError(f"Project root directory {directory} does not exist.")
    print(f"Project root directory: {directory}")

    # Set the Babel configuration file path
    babel_cfg = os.path.join(directory, "babel.cfg")
    if not os.path.exists(babel_cfg):
        print(f"Error: Babel configuration file {babel_cfg} does not exist.")
        raise FileNotFoundError(f"Babel configuration file {babel_cfg} does not exist.")

    # Set the translation directory
    translation_dir = os.path.join(directory, name, "locale")
    if not os.path.exists(translation_dir):
        print(f"Error: Translation directory {translation_dir} does not exist.")
        raise FileNotFoundError(
            f"Translation directory {translation_dir} does not exist."
        )
    print(f"Translation directory {translation_dir}")

    # Compile the translation files
    try:
        subprocess.run(
            [
                "pybabel",
                "compile",
                "--directory",
                translation_dir,
                "--domain",
                name,
                "--statistics",
            ],
            check=True,
        )
        print("Compilation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error: Compilation failed.")
        print(e.stderr)
        raise RuntimeError(f"Compilation failed: {e.stderr}") from e


if __name__ == "__main__":
    import argparse

    import dotenv

    env = dotenv.dotenv_values()

    parser = argparse.ArgumentParser(
        description="Compile translation files using Babel."
    )
    parser.add_argument(
        "--name",
        required="NAME" not in env,
        default=env.get("NAME"),
        help="Project name",
    )
    parser.add_argument(
        "--directory",
        required="DIRECTORY" not in env,
        default=env.get("DIRECTORY"),
        help="Project root directory",
    )
    args = parser.parse_args()

    postprocess(args.name, args.directory)
