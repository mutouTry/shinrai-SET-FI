# The C++ extension is declared here; everything else is in pyproject.toml.
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "setfi._engine",
            ["cpp/engine.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
