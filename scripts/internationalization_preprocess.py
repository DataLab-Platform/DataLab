# -*- coding: utf-8 -*-
#
# Distributed under the BSD 3-clause license (see guidata/LICENSE for details).

"""Script to extract and process translatable strings from Python files using Babel.

This script uses the `pybabel` command-line tool to extract translatable strings from
Python files. It generates a template catalog (.pot) containing the extracted strings,
initializes the translation file (.po), if need be, and updates it.
"""

import os
import subprocess
import typing


def scan(  # noqa: PLR0913 # pylint: disable=too-many-positional-arguments
    name: str,
    directory: typing.Union[str, os.PathLike],
    version: str,
    copyright_holder: str,
    email: str,
    locale: typing.List[str],
) -> None:
    """Extract and process translatable strings from Python files using Babel.

    This function performs the following steps:
    1. Check the project root directory.
    2. Check the Babel configuration file.
    3. Set up the translation directory.
    3. Extract translatable strings from the project and generate a template file.
    4. For each specified locale, check that the necessary directory structure exists
    and generate or update the corresponding translation file.

    Args:
        name: The name of the project, used for directory and domain naming.
        directory: The root directory of the project.
        version: The version of the project.
        copyright_holder: The copyright holder's name.
        email: The contact email for reporting translation bugs.
        locale: A list of locale codes (e.g., ['fr', 'it']) for which translation files
        should be generated or updated.

    Raises:
        FileNotFoundError: The Babel configuration file does not exist.
        RuntimeError: Extraction or translation file generation failed.

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
        os.makedirs(translation_dir)
        print(f"Created translation directory: {translation_dir}")

    # Set the template file path
    catalog_template = os.path.join(translation_dir, f"{name}.pot")
    print(f"Output template file: {catalog_template}")

    # Run pybabel extract
    print("Extracting translatable strings...")
    try:
        subprocess.run(
            [
                "pybabel",
                "extract",
                directory,
                "--mapping-file",
                babel_cfg,
                "--output-file",
                catalog_template,
                "--no-location",
                "--no-wrap",
                "--sort-by-file",
                "--header-comment",
                """\
# Translations template for PROJECT.
# Copyright (C) YEAR ORGANIZATION
# This file is distributed under the same license as the PROJECT project.
#""",
                "--project",
                name,
                "--version",
                version,
                "--copyright-holder",
                copyright_holder,
                "--msgid-bugs-address",
                email,
            ],
            check=True,
        )
        print(f"Extraction completed successfully. Output file: {catalog_template}")
    except subprocess.CalledProcessError as e:
        print("Error: Extraction failed.")
        print(e.stderr)
        raise RuntimeError(f"Extraction failed: {e.stderr}") from e

    # Initialize the folder structure if it does not exist
    for loc in locale:
        locale_dir = os.path.join(translation_dir, loc, "LC_MESSAGES")
        if not os.path.exists(locale_dir):
            os.makedirs(locale_dir)
            print(f"Created translation directory: {locale_dir}")

    # Generate or update the translation file
    print(f"Translation directory: {translation_dir}")
    try:
        for loc in locale:
            subprocess.run(
                [
                    "pybabel",
                    "update",
                    "--input-file",
                    catalog_template,
                    "--output-dir",
                    translation_dir,
                    "--locale",
                    loc,
                    "--domain",
                    name,
                    "--ignore-obsolete",
                    "--init-missing",
                    "--no-wrap",
                    "--update-header-comment",
                ],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        print("Error: Translation file generation failed.")
        print(e.stderr)
        raise RuntimeError(f"Translation file generation failed: {e.stderr}") from e


if __name__ == "__main__":
    import argparse

    import dotenv

    env = dotenv.dotenv_values()

    parser = argparse.ArgumentParser(
        description="Extract and process translatable strings using Babel."
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
    parser.add_argument(
        "--version",
        required="VERSION" not in env,
        default=env.get("VERSION"),
        help="Project version",
    )
    parser.add_argument(
        "--copyright-holder",
        required="COPYRIGHT" not in env,
        default=env.get("COPYRIGHT"),
        help="Copyright holder",
    )
    parser.add_argument(
        "--email",
        required="EMAIL" not in env,
        default=env.get("EMAIL"),
        help="Contact email for msgid bugs address",
    )
    locale_env = env.get("LOCALE")
    parser.add_argument(
        "--locale",
        required="LOCALE" not in env,
        default=locale_env.split() if locale_env is not None else None,
        nargs="+",
        help="Locales for the translation (space-separated list, e.g., en fr de)",
    )

    args = parser.parse_args()

    scan(
        args.name,
        args.directory,
        args.version,
        args.copyright_holder,
        args.email,
        locale=args.locale,
    )
