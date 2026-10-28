#  Copyright (C) 2024
#  Smithsonian Astrophysical Observatory
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utility routines for interfacing with XSPEC models.

This is used by `xspec-models-cxc` to build the code needed
to interface with XSPEC models. It relies on the
`parse-xspec module <https://pypi.org/project/parse-xspec/>`_
to identify models from a model.dat or lmodel.dat file.

"""

import glob
from importlib import resources
from io import StringIO
import logging
import os
from pathlib import Path
import subprocess
import sysconfig
from typing import Mapping, Sequence

from parse_xspec.models import ModelDefinition


__all__ = ("get_xspec_model_path",
           "get_xspec_include_path", "get_xspec_library_path",
           "get_xspec_libs",
           "select_models", "wrapmodel_compiled", "wrapmodel_python",
           "find_fortran_files", "find_c_files", "find_cplusplus_files"
           )


info = logging.getLogger(__name__).info
warning = logging.getLogger(__name__).warning

# Access to the data location via importlib.resources.files
DATALOC: str = "xspec_models_cxc_helpers.data"

# In case it ever becomes useful to tweak this list after loading the
# module. See get_xspec_libs().
#
LIBNAMES: list[str] = ["XSFunctions", "XSUtil", "XS", "hdsp",
                       "cfitsio", "CCfits", "wcs",
                       "XSCore", "XSData", "XSFit", "XSFunctions", "XSMinuit", "XSModel", "XSPlot", "XSPy", "XSSim", "XSStat", "XSUser", "XSUtil"]



def get_headas() -> Path:
    """Return the HEADAS environment variable as a path."""

    HEADAS_ENV = os.getenv('HEADAS')
    if HEADAS_ENV is None:
        raise OSError("Unable to find HEADAS environment variable")

    return Path(HEADAS_ENV)


def get_xspec_model_path() -> Path:
    """Return the location of the XSPEC model.dat file."""

    HEADAS = get_headas()
    modelfile = HEADAS / '../spectral/manager/model.dat'
    modelfile = modelfile.resolve()
    if not modelfile.is_file():
        raise OSError(f"Unable to find {modelfile}")

    return modelfile


def get_xspec_path(path: str) -> Path:
    """Return the location of the given XSPEC directory.

    This should be easy but the CXC xspec-modelsonly conda
    package has a different layout to XSPEC.

    """

    HEADAS = get_headas()
    out = HEADAS / path

    # This looks like XSPEC
    if out.is_dir():
        return out

    # Is this xspec-modelsonly?
    out = (HEADAS / '..').resolve() / path
    if not out.is_dir():
        raise IOError(f"Unable to find XSPEC {path} directory: {out}")

    return out


def get_xspec_include_path() -> Path:
    """Return the location of the XSPEC include directory."""

    return get_xspec_path("include")


def get_xspec_library_path() -> Path:
    """Return the location of the XSPEC library directory."""

    return get_xspec_path("lib")


def get_xspec_libs(path: Path | None = None) -> list[str]:
    """Return the XSPEC libraries that are needed to compile the model code.

    Parameters
    ----------
    path
       The location of the XSPEC library directory. If not set then
       get_xspec_library_path() is used to determine the location.

    Returns
    -------
    libs
       The library names, without the "lib" prefix or any suffix.

    """

    if path is None:
        xspec_libdir = get_xspec_library_path()
    else:
        xspec_libdir = path

    # There's some attempt to be platform independent, but
    # is it worth it?
    #
    if sysconfig.get_config_var("WITH_DYLD"):
        suffix = ".dylib"
    else:
        suffix = sysconfig.get_config_var('SHLIB_SUFFIX')

    # The tricky thing is that we have XSFunctions, XSUtil, and XS as
    # arguments. So we can not just look for XS*, as that will match
    # multiple libraries. We also don't want to include all matches to XS
    # as there are a number of matches we do not need.
    #
    def match(name: str) -> str:
        head = f"lib{name}{suffix}"
        if (xspec_libdir / head).is_file():
            return name

        head = f"lib{name}_*{suffix}"
        ms = glob.glob(str(xspec_libdir / head))
        if len(ms) == 1:
            return Path(ms[0]).stem[3:]

        head = f"lib{name}-*{suffix}"
        ms = glob.glob(str(xspec_libdir / head))
        if len(ms) == 1:
            return Path(ms[0]).stem[3:]

        raise OSError(f"Unable to find a unique match for lib{name}*{suffix} in {xspec_libdir}")

    out = []
    for libname in LIBNAMES:
        # Note: not all names are versioned
        out.append(match(libname))

    return out


def select_models(models: Sequence[ModelDefinition]
                  ) -> tuple[list[ModelDefinition], list[ModelDefinition]]:
    """Identify the models that can be used from Python.

    Select those models that this module can provide access to from
    Python.  The reasons for a model being unsuported include: support
    could be added but it just hasn't yet, there is a new model type
    in XSPEC that this package does not yet understand, and models
    that are not really intended for use outside of XSPEC directly
    (such as mixing models).

    Parameters
    ----------
    models
        The models provided in the model.dat file (normally created
        by `parse_xspec.models.parse_xspec_model_description`.

    Returns
    -------
    supported, unsupported
        Those models that can be used and those that can not.

    Notes
    -----
    Only additive, multiplicative, and convolution models are
    currently supported. All four "language styles" - that is C, C++,
    FORTRAN 4-byte real, and FORTRAN 8-byte real are supported.

    """

    # Filter to the ones we care about.
    #
    supported1 = []
    unsupported = []
    for m in models:
        if m.modeltype in ['Add', 'Mul', 'Con']:
            supported1.append(m)
        else:
            unsupported.append(m)

    # A sanity check (at present this should be all supported
    # "language styles").
    #
    allowed_langs = ["Fortran - single precision",
                     "Fortran - double precision",
                     "C style",
                     "C++ style"]
    supported = []
    for m in supported1:
        if m.language in allowed_langs:
            supported.append(m)
        else:
            unsupported.append(m)

    return supported, unsupported


def get_npars(npars: int) -> str:
    """Return the number of parameters."""

    if npars == 0:
        return "no parameters"
    if npars == 1:
        return "1 parameter"

    return f"{npars} parameters"


def wrapmodel_basic(model: ModelDefinition,
                    npars: int,
                    call: str,
                    text: str,
                    inplace: bool = False,
                    convolution: bool = False
                    ) -> str:
    """Create the m.def line for a single model"""

    assert not(inplace and convolution)

    out = f'    m.def("{model.name}", {call}'

    if model.language == 'Fortran - single precision':
        out += f'_f<{model.funcname}_'
    elif model.language == 'Fortran - double precision':
        out += F'_F<{model.funcname}_'
    elif model.language == 'C++ style':
        out += f'_C<C_{model.funcname}'
    elif model.language == 'C style':
        out += f'_C<{model.funcname}'  # should this be 'c_{model.funcname}' (not for compmag...)?
    else:
        raise ValueError("Unsuported language: {model.name} {model.funcname} {model.language}")

    out += f', {npars}>, "{text}",'
    out += '"pars"_a,"energies"_a,'
    if convolution:
        out += '"model"_a,'

    if inplace:
        out += '"out"_a,'

    out += '"spectrum"_a=1'
    if not model.language.startswith('Fortran'):
        out += ',"initStr"_a=""'

    if inplace or convolution:
        out += ',py::return_value_policy::reference'

    out += ');'
    return out


def wrapmodel_cxx(model: ModelDefinition,
                  npars: int,
                  text: str) -> str:
    """Make the C++ version available as name_"""

    if model.language != 'C++ style':
        return ''

    out = f'    m.def("{model.name}_", xspec_models_cxc::wrapper_inplace_CXX<'
    out += f'{model.funcname}, {npars}>, "{text}",'
    out += '"pars"_a,"energies"_a,"out"_a,"spectrum"_a=1,'
    out += '"initStr"_a=""'
    out += ',py::return_value_policy::reference'
    out += ');'
    return out


def wrapmodel_add(model: ModelDefinition,
                  npars: int) -> str:
    """What is the m.def line for this additive model?"""

    npars_str = get_npars(npars)
    out = wrapmodel_basic(model, npars, 'xspec_models_cxc::wrapper',
                          f'The XSPEC additive {model.name} model ({npars_str}).')
    out += '\n'
    out += wrapmodel_basic(model, npars, 'xspec_models_cxc::wrapper_inplace',
                           f'The XSPEC additive {model.name} model ({npars_str}); inplace.',
                           inplace=True)
    out += '\n'
    out += wrapmodel_cxx(model, npars,
                         f"The XSPEC additive {model.name} model ({npars_str}); RealArray, inplace.")
    return out


def wrapmodel_mul(model: ModelDefinition,
                  npars: int) -> str:
    """What is the m.def line for this multiplicative model?"""

    npars_str = get_npars(npars)
    out = wrapmodel_basic(model, npars, 'xspec_models_cxc::wrapper',
                          f'The XSPEC multiplicative {model.name} model ({npars_str}).')
    out += '\n'
    out += wrapmodel_basic(model, npars, 'xspec_models_cxc::wrapper_inplace',
                           f'The XSPEC multiplicative {model.name} model ({npars_str}); inplace.',
                           inplace=True)
    out += '\n'
    out += wrapmodel_cxx(model, npars,
                         f"The XSPEC multiplicative {model.name} model ({npars_str}); RealArray, inplace.")
    return out


def wrapmodel_con(model: ModelDefinition,
                  npars: int) -> str:
    """What is the m.def line for this convolution model?"""

    npars_str = get_npars(npars)
    return wrapmodel_basic(model, npars, 'xspec_models_cxc::wrapper_con',
                           f'The XSPEC convolution {model.name} model ({npars_str}); inplace.',
                           convolution=True)


def wrapmodel_compiled(model: ModelDefinition
                       ) -> tuple[str, str, str]:
    """The C++ code needed to intrface with this model.

    Parameters
    ----------
    model
       The model to use.

    Returns
    -------
    compiled_code, modeltype, description
       The code needed to be included in the C++ module, the model
       type ("Add", "Mul", or "Con"), and a description.

    """

    npars = len(model.pars)
    if model.modeltype == 'Add':
        mdef = wrapmodel_add(model, npars)

    elif model.modeltype == 'Mul':
        mdef = wrapmodel_mul(model, npars)

    elif model.modeltype == 'Con':
        mdef = wrapmodel_con(model, npars)

    else:
        raise ValueError(f"Unknown model: {model.name} {model.modeltype}")

    npars_str = get_npars(npars)
    desc = f"{model.name} - {npars_str}"
    return mdef, model.modeltype, desc


def to_model(mtype: str) -> str:
    """What is the model type enumeration."""
    return {'Add': 'ModelType.Add',
            'Mul': 'ModelType.Mul',
            'Con': 'ModelType.Con' }[mtype]


def to_lang(langtype: str) -> str:
    """What is the language enumeration"""
    return {'C++ style': 'LanguageStyle.CppStyle8',
            'C style': 'LanguageStyle.CStyle8',
            'Fortran - single precision': 'LanguageStyle.F77Style4',
            'Fortran - double precision': 'LanguageStyle.F77Style8'}[langtype]


def to_ptype(ptype):
    """What is the parameter type enumeration"""
    # We don't support periodic yet
    return {'Basic': 'ParamType.Default',
            'Switch': 'ParamType.Switch',
            'Scale': 'ParamType.Scale',
            '?': 'ParamType.Periodic'}[ptype]


def wrapmodel_python(model: ModelDefinition) -> tuple[str, str]:
    """What is the Python code needed to use this model.

    Parameters
    ----------
    model
       The model to use.

    Returns
    -------
    name, python_code
       The name of the model and the python code needed to create
       an instance of the model.

    """

    # pyright claims these can be None, so error out if they are;
    # the assumption is that this should not happen so there is no
    # attempt to make this a "nice" error.
    #
    assert model.modeltype is not None
    assert model.language is not None

    mtype = to_model(model.modeltype)
    lang = to_lang(model.language)
    out = [f"XSPECModel(modeltype={mtype}",
           f"name='{model.name}'",
           f"funcname='{model.funcname}'",
           f"language={lang}",
           f"elo={model.elo}",
           f"ehi={model.ehi}"]

    pars = []
    for p in model.pars:
        ps = [f"XSPECParameter(paramtype={to_ptype(p.paramtype)}"]
        ps.append(f"name='{p.name}'")
        ps.append(f"default={p.default}")
        if p.units is not None:
            ps.append(f"units='{p.units}'")

        try:
            if p.frozen:
                ps.append("frozen=True")
        except AttributeError:
            # Assume that if there's no frozen attribute it is
            # always frozen
            ps.append("frozen=True")

        for t in ['soft', 'hard']:
            for r in ['min', 'max']:
                attr = getattr(p, f'{t}{r}')
                if attr is not None:
                    ps.append(f"{t}{r}={attr}")

        if p.delta is not None:
            ps.append(f"delta={p.delta}")

        pars.append(', '.join(ps) + ')')

    pars = ', '.join(pars)
    out.append(f"parameters=[{pars}]")

    if len(model.flags) > 0 and model.flags[0] > 0:
        out.append("use_errors=True")

    if len(model.flags) > 1 and model.flags[1] > 0:
        out.append("can_cache=False")

    return model.name, ', '.join(out) + ')'


# Try to match XSPEC for finding files to compile for user models.
#
def find_file_types(types: Mapping[str, str]
                    ) -> dict[str, list[str]] | None:
    """Return a dictionary listing the types of files in the types
    dictionary, which has key as the type name and value as the glob
    pattern - e.g.

      types = { "generic": "*.f", "90": "*.f90" }

    Only those types that contain a match are included; if there are
    no matches then None is returned. The return names are sorted
    alphabetically.

    At present it turns out that it is possible to create the pattern
    from the typename, ie a dictionary is not needed as input, but
    leave as is as there is no need to "optimise" this.

    """

    out = {}
    for (typename, pattern) in types.items():
        match = glob.glob(pattern)
        if len(match) > 0:
            # It looks like XSPEC uses alphabetical sorting so do this
            # here (needed for reltrans to compile the F90 code in the
            # correct order).
            out[typename] = sorted(match)

    if len(out) == 0:
        return None

    return out


def find_fortran_files() -> dict[str, list[str]] | None:
    """Return the Fortran files found in the current
    directory, labelled by "type".

      "f":   *.f
      "f03": *.f03
      "f90": *.f90

    The dictionary only contains keys if there was a
    match for that pattern; if there are no matches
    then None is returned.
    """

    return find_file_types({"f": "*.f",
                            "f03": "*.f03",
                            "f90": "*.f90"})


def find_c_files() -> dict[str, list[str]] | None:
    """Return the C files found in the current
    directory, labelled by "type".

      "c": *.c

    The dictionary only contains keys if there was a
    match for that pattern; if there are no matches
    then None is returned.

    Note that on case-insensitive file systems, this
    will match the same files as find_c_files() for
    the "C" and "c" options.
    """

    return find_file_types({"c": "*.c"})


def is_fs_case_sensitive(path: Path) -> bool:
    """Is this filesystem case sensitive?

    This seems a lot of work.... It is taken from
    https://stackoverflow.com/a/36612604
    """

    # We force the filename to have mixed case by virtue of the
    # prefix.
    #
    with tempfile.NamedTemporaryFile(prefix='TmP',
                                     dir=path,
                                     delete=True) as tfile:
        return (not os.path.exists(tfile.name.lower()))


def find_cplusplus_files() -> dict[str, list[str]] | None:
    """Return the Fortran files found in the current
    directory, labelled by "type".

      "cxx" : *.cxx
      "C"   : *.C
      "cc"  : *.cc
      "cpp" : *.cpp

    The dictionary only contains keys if there was a match for that
    pattern; if there are no matches then None is returned.

    Note that on case-insensitive file systems, this will match the
    same files as find_c_files() for the "C" and "c" options.

    Files that match the patterns

        lpack_*.cxx
        *FunctionMap.cxx

    are excluded from the searches, as they are typically created by
    XSPEC (unfortunately it depends on what name the user who called
    initpackage used to know what the '*' should be, so we just ignore
    any matches).

    """

    this_dir = Path(".").resolve()
    is_case_sen = is_fs_case_sensitive(this_dir)

    out = find_file_types({"cxx": "*.cxx",
                           "C": "*.C",
                           "cc": "*.cc",
                           "cpp": "*.cpp"})
    if out is None:
        return None

    if not is_case_sen and "C" in out:
        warning(f"directory {this_dir} is  not case sensitive, so C++ may have found *.c files!")

    try:
        cxx = out["cxx"]
        repl = []
        for fname in cxx:
            if is_case_sen:
                check = fname.startswith("lpack_") or fname.endswith("FunctionMap.cxx")
            else:
                lname = fname.lower()
                check = lname.startswith("lpack_") or lname.endswith("functionmap.cxx")

            if check:
                info(f"Skipping {fname} as assumed to have been created by initpackage")
                continue

            repl.append(fname)

        if len(repl) == 0:
            del out["cxx"]
        elif len(repl) != len(cxx):
            out["cxx"] = repl

    except KeyError:
        pass

    return out


# UDMGET support
#
def create_xspec_init(outpath: Path):
    """Create the xspec.inc file needed for using udmget."""

    # Is this file installed? If so we cna just use it.
    #
    inc_path = get_xspec_include_path()
    infile = inc_path / "xspec.h"
    if not infile.is_file():
        # xspec-modelsonly does not include it, so use a copy
        #
        res_path = resources.files(DATALOC)
        infile = res_path / "xspec.h"
        if not infile.is_file():
            raise OSError(f"Unable to find xspec.h in: {inc_path} {res_path}")

    # Always over-write the output file.
    #
    outfile = outpath / "xspec.inc"
    with outfile.open(mode="wt") as ofh:
        ofh.write(infile.read_text())


def create_xspec_udmget(outpath: Path,
                        dprec: bool = True
                        ) -> Path:
    """Copy over the code implemeting udmget.

    Parameters
    ----------
    outpath
       Output location
    dprec
       If True use the 64-bit version, otherwise the 32 bit.

    Returns
    -------
    path
       The file that was created.

    """

    name = "xsudmget64.cxx" if dprec else "xsudmget.cxx"
    inc_path = get_xspec_include_path()
    infile = inc_path / name
    if not infile.is_file():
        # xspec-modelsonly does not include it, so use a copy
        #
        res_path = resources.files(DATALOC)
        infile = res_path / name
        if not infile.is_file():
            raise OSError(f"Unable to find xspec.h in: {inc_path} {res_path}")

    # Always over-write the output file.
    #
    outfile = outpath / name
    with outfile.open(mode="wt") as ofh:
        ofh.write(infile.read_text())

    return outfile


def compile_xsudmget(udmpath: Path) -> Path:
    """Do we have to compile the xsudmget file?

    We just compile this manually, picking up the compiler
    with the CXX enviroment variable.

    """

    # Should this be OS dependent?
    #
    cxx = os.getenv("CXX")
    if cxx is None:
        cxx = "g++"

    # Where should this go?
    #
    outpath = udmpath.parent / (udmpath.stem + ".o")

    # Taken from an initpackage run with XSPEC 12.13.0
    #
    args = [cxx, "-c", "-o", str(outpath),
            "-O2", "-g", "-fPIC", str(udmpath)]
    info("Compiling xsudmget with: %s", ' '.join(args))
    subprocess.run(args, check=True)
    return outpath
