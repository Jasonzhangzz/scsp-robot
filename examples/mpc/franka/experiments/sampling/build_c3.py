"""Compile the standalone C3 extension next to this package."""

import subprocess
import sys
import sysconfig
from pathlib import Path


def extension_path():
    here = Path(__file__).resolve().parent
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    return here / ("c3_core" + suffix)


def build():
    here = Path(__file__).resolve().parent
    source = here / "csrc" / "c3_core.cpp"
    output = extension_path()
    include = Path(sysconfig.get_path("include"))
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-I/usr/include/eigen3",
        "-I/usr/include",
        "-I" + str(include),
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


if __name__ == "__main__":
    print(build())
    sys.stdout.flush()
