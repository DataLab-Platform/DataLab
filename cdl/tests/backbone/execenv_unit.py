# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
CDLExecEnv test
---------------

Checking DataLab execution environment management.
"""

# guitest: skip (only needed for regression tests)

from __future__ import annotations

import os
import sys

from cdl import __version__
from cdl.env import VerbosityLevels, execenv
from cdl.utils.tests import get_script_output

ARGV_TEST = "--execenvtest"


def print_execenv() -> None:
    """Print CDL execution environment"""
    sys.argv.remove(ARGV_TEST)
    print(str(execenv.to_dict()))


def get_subprocess_execenv_dict(args: list[str], env: dict | None = None) -> dict:
    """Get CDL execution environment dict from subprocess

    Args:
        args (list[str]): command-line arguments
        env (dict, optional): environment variables to pass to subprocess

    Returns:
        dict[str, str | int | bool | None]: CDL execution environment dict
    """
    output = get_script_output(__file__, args=args + [ARGV_TEST], env=env)
    return eval(output)


def assert_two_dicts_are_equal(
    dict1: dict, dict2: dict, exceptions: tuple[str] | None = None
) -> None:
    """Assert two dicts are equal

    Args:
        dict1 (dict): first dict
        dict2 (dict): second dict
        exceptions (tuple[str], optional): keys to ignore
    """
    diff_keys = []
    for key in dict1:
        if key in exceptions:
            continue
        if dict1[key] != dict2[key]:
            diff_keys.append(key)
    if diff_keys:
        assert False, "Dictionaries differ on keys: %s" % str(diff_keys)


def test_cli():
    """Test CDL execution environment from command-line"""
    remove_all_cdl_envvars()

    # Test default values
    execenv.print("Testing command-line arguments:")
    execenv.print("  Default values: ", end="")
    execenvdict = get_subprocess_execenv_dict([])
    execenv.print("OK")
    assert execenvdict == execenv.to_dict()
    # Testing boolean arguments
    execenv.print("  Testing boolean arguments:")
    for argname in ("unattended", "screenshot"):
        execenv.print("    %s:" % argname, end="")
        for val in (True, False):
            execenv.print(" %s" % str(val), end="")
            if val:
                args = [f"--{argname}"]
            else:
                args = []  # Default value is False
            execenvdict = get_subprocess_execenv_dict(args)
            assert_two_dicts_are_equal(execenvdict, execenv.to_dict(), (argname,))
            assert (
                execenvdict[argname] is val
            ), f"execenvdict[{argname}] = {execenvdict[argname]} != {val}"
        execenv.print()
    # Testing integer arguments
    execenv.print("  Testing integer arguments:")
    for argname in ("delay", "xmlrpcport"):
        execenv.print("    %s:" % argname, end="")
        for val in (None, 0, 1, 2):
            if val is None:
                args = []
            else:
                args = [f"--{argname}", str(val)]
            execenv.print(f" {val}", end="")
            execenvdict = get_subprocess_execenv_dict(args)
            assert_two_dicts_are_equal(execenvdict, execenv.to_dict(), (argname,))
            if val is None:
                if argname == "delay":
                    # Default value is 0 for delay
                    assert execenvdict[argname] == 0
                else:
                    # Default value is None for xmlrpcport
                    assert execenvdict[argname] is None
            else:
                assert execenvdict[argname] == val
        execenv.print()
    # Testing choice arguments
    execenv.print("  Testing choice arguments:")
    for argname in ("verbose",):
        execenv.print("    %s:" % argname, end="")
        choices = {"verbose": [verb.value for verb in VerbosityLevels]}
        defaultval = {"verbose": VerbosityLevels.NORMAL.value}[argname]
        for val in [None] + choices[argname]:
            if val is None:
                args = []
            else:
                args = [f"--{argname}", val]
            execenv.print(f" {val}", end="")
            execenvdict = get_subprocess_execenv_dict(args)
            assert_two_dicts_are_equal(execenvdict, execenv.to_dict(), (argname,))
            if val is None:
                assert execenvdict[argname] == defaultval
            else:
                assert execenvdict[argname] == val
        execenv.print()
    # Testing special arguments
    execenv.print("  Testing special arguments:")
    execenv.print("    version: ", end="")
    args = ["--version"]
    output = get_script_output(__file__, args=args + [ARGV_TEST])
    execenv.print(output)
    assert __version__ in output
    execenv.print("    h5 positionnal argument: ", end="")
    for h5files in (None, ["test.h5"], ["toto.h5", "tata.h5"]):
        if h5files is None:
            args = []
        else:
            args = [";".join(h5files)]
        execenv.print(f" {h5files}", end="")
        execenvdict = get_subprocess_execenv_dict(args)
        assert execenvdict["h5files"] == h5files
    execenv.print()
    execenv.print("    h5browser argument:")
    for argname in ("-b", "--h5browser"):
        execenv.print(f"      {argname}", end="")
        for val in (None, "test.h5"):
            if val is None:
                args = []
            else:
                args = [argname, val]
            execenv.print(f" {val}", end="")
            execenvdict = get_subprocess_execenv_dict(args)
            assert execenvdict["h5browser_file"] == val
        execenv.print()
    execenv.print("=> Everything is OK")


def iterate_over_attrs_envvars():
    """Iterate over CDL environment variables"""
    for attrname in dir(execenv):
        if attrname.endswith("_ENV"):
            envvar = getattr(execenv, attrname)
            attrname = envvar[4:].lower()
            yield attrname, envvar


def remove_all_cdl_envvars():
    """Remove all CDL environment variables"""
    for _attrname, envvar in iterate_over_attrs_envvars():
        os.environ.pop(envvar, None)


def get_attr_to_envvar(
    vartype: type,
    default: int | str | None | bool,
    values: list[int | str | None | bool] | None = None,
) -> tuple:
    """Get ATTR_TO_ENVVAR tuple for a given type"""
    if vartype is bool:
        if default is False:
            ate = ((True, ("1", "true")), (False, (None, "", "0", "false")))
        else:
            ate = ((True, (None, "", "1", "true")), (False, ("0", "false")))
    elif vartype is int:
        ate = []
        for val in values:
            if val == default:
                ate.append((val, (None, "", str(val))))
            else:
                ate.append((val, (str(val),)))
        ate = tuple(ate)
        if default is None:
            ate = ((None, (None, "")),) + ate
    elif vartype is str:
        ate = ((None, (None, "")),)
    elif vartype is list:
        ate = []
        for choice in values:
            if choice == default:
                ate.append((choice, (None, "", choice)))
            else:
                ate.append((choice, (choice,)))
        ate = tuple(ate)
    else:
        raise ValueError(f"Unknown type {vartype}")
    return ate


ATTR_TO_ENVVAR = {
    "unattended": get_attr_to_envvar(bool, False),
    "screenshot": get_attr_to_envvar(bool, False),
    "do_not_quit": get_attr_to_envvar(bool, False),
    "catcher_test": get_attr_to_envvar(bool, False),
    "delay": get_attr_to_envvar(int, 0, [0, 10, 20]),
    "xmlrpcport": get_attr_to_envvar(int, None, [9854, 1020, 213]),
    "verbose": get_attr_to_envvar(
        list, VerbosityLevels.NORMAL.value, [verb.value for verb in VerbosityLevels]
    ),
}


def test_envvar():
    """Testing DataLab configuration file"""
    assert execenv.unattended is False, "This test must be run with unattended=False"
    print("Testing DataLab execution environment:")
    for attrname, envvar in iterate_over_attrs_envvars():
        attr_to_envvar = ATTR_TO_ENVVAR[attrname]
        print(f"  Testing {attrname}:")
        for value, envvals in attr_to_envvar:
            print(f"    {value}: [attr->env]", end="")
            remove_all_cdl_envvars()
            if value is not None:
                setattr(execenv, attrname, value)
            assert os.environ.get(envvar) in envvals, "os.environ[%s] = %s != %s" % (
                envvar,
                os.environ.get(envvar),
                envvals,
            )
            print(f" [env->attr]", end="")
            remove_all_cdl_envvars()
            for envval in envvals:
                if envval is not None:
                    os.environ[envvar] = envval
                assert (
                    getattr(execenv, attrname) == value
                ), "execenv.%s = %s != %s (envval = %r)" % (
                    attrname,
                    getattr(execenv, attrname),
                    value,
                    envval,
                )
            print(f" [env->subprocess->attr]", end="")
            remove_all_cdl_envvars()
            for envval in envvals:
                if envval is not None:
                    os.environ[envvar] = envval
                execenvdict = get_subprocess_execenv_dict([])
                assert (
                    execenvdict[attrname] == value
                ), "execenvdict[%s] = %s != %s (envval = %r)" % (
                    attrname,
                    execenvdict[attrname],
                    value,
                    envval,
                )
            print()
    print("=> Everything is OK")


if __name__ == "__main__":
    if ARGV_TEST in sys.argv:
        print_execenv()
    else:
        test_envvar()
        test_cli()
