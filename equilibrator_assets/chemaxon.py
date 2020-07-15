# The MIT License (MIT)
#
# Copyright (c) 2013 The Weizmann Institute of Science.
# Copyright (c) 2018 Institute for Molecular Systems Biology, ETH Zurich.
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""Enrich compounds with cheminformatic properties."""


import logging
import subprocess
from collections import namedtuple
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
from openbabel import pybel
from periodictable import elements

# KMS
import os
# KMS Add environment for jupyter notebook (PATH issue)
MY_ENV = os.environ.copy()
MY_ENV["PATH"] = "/Applications/MarvinSuite/bin:" + MY_ENV["PATH"]


ATOMIC_NUMBER_TO_SYMBOL = {e.number: e.symbol for e in elements}
SYMBOL_TO_ATOMIC_NUMBER = {e.symbol: e.number for e in elements}

logger = logging.getLogger(__name__)

MoleculeData = namedtuple("MoleculeData", ["atom_bag", "smiles"])
# A dictionary from InChIKey to MoleculeData objects
# Here we list a few exceptions, i.e. compounds that are not treated
# correctly by cxcalc, and override them with our own data
COMPOUND_EXCEPTIONS = {
    # H+
    # ChemAxon fails if we try to run it on a single proton.
    # error is "pka: Inconsistent molecular structure."
    "InChI=1S/p+1": MoleculeData(atom_bag={"H": 1}, smiles=None),
    # sulfur
    # ChemAxon gets confused with the structure of sulfur
    #  (returns a protonated form, [SH-], at pH 7).
    "InChI=1S/S": MoleculeData(atom_bag={"S": 1, "e-": 16}, smiles="S"),
    # CO
    # ChemAxon gets confused with the structure of carbon
    # monoxide (returns a protonated form, [CH]#[O+], at pH 7).
    "InChI=1S/CO/c1-2": MoleculeData(
        atom_bag={"C": 1, "O": 1, "e-": 14}, smiles="[C-]#[O+]"
    ),
    # H2
    "InChI=1S/H2/h1H": MoleculeData(atom_bag={"H": 2, "e-": 2}, smiles=None),
    # Metal Cations get multiple pKa values from ChemAxon, which is
    # obviously a bug. We override the important ones here:
    # Ca2+
    "InChI=1S/Ca/q+2": MoleculeData(
        atom_bag={"Ca": 1, "e-": 18}, smiles="[Ca++]"
    ),
    # K+
    "InChI=1S/K/q+1": MoleculeData(atom_bag={"K": 1, "e-": 18}, smiles="[K+]"),
    # Mg2+
    "InChI=1S/Mg/q+2": MoleculeData(
        atom_bag={"Mg": 1, "e-": 10}, smiles="[Mg++]"
    ),
    # Fe2+
    "InChI=1S/Fe/q+2": MoleculeData(
        atom_bag={"Fe": 1, "e-": 24}, smiles="[Fe++]"
    ),
    # Fe3+
    "InChI=1S/Fe/q+3": MoleculeData(
        atom_bag={"Fe": 1, "e-": 23}, smiles="[Fe+++]"
    ),
    # thiocyanate
    "InChI=1S/CHNS/c2-1-3/h3H/p-1": MoleculeData(
        atom_bag={"S": 1, "C": 1, "N": 1, "H": 1, "e-": 31}, smiles="[S-]C#N"
    ),
    # These following compounds don't have a real structure, but we still
    # want to use them for training the component-contribution data
    # ferredoxin(red)
    "metanetx.chemical:MNXM169": MoleculeData(
        atom_bag={"Fe": 1, "e-": 26}, smiles=None
    ),
    # ferredoxin(ox)
    "metanetx.chemical:MNXM178": MoleculeData(
        atom_bag={"Fe": 1, "e-": 25}, smiles=None
    ),
}


CXCALC_BIN = "cxcalc"

class ChemAxonNotAvailableError(OSError):
    """Raise when ``cxcalc`` is not available."""

    pass


def verify_cxcalc():
    """Verify the existence of the ``cxcalc`` command line program."""
    try:
        # KMS added in env for jupyter
        subprocess.run([CXCALC_BIN, "--help"], env=MY_ENV)
        return True
    except OSError:
        return False


def run_cxcalc(mol_string: str, args: List[str]):
    """
    Run cxcalc as a subprocess.

    Parameters
    ----------
    mol_string : str
        A text description of the molecule(s) (SMILES or InChI).
    args : list
        A list of arguments for cxcalc.

    Returns
    -------
    tuple
        str
            The cxcalc standard output.
        str
            The cxcalc standard error.

    Raises
    ------
    subprocess.CalledProcessError
        If the command fails.
    ChemAxonNotAvailableError
        If the cxcalc program is not working as expected.

    """
    command = [CXCALC_BIN] + args
    try:
        logger.debug("Parameters: %s | %s", mol_string, " ".join(command))
        # KMS added in env for jupyter
        result = subprocess.run(
            command,
            input=mol_string,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
            env=MY_ENV,
        )
    except OSError as error:
        raise ChemAxonNotAvailableError(
            f"{error.strerror} (Please ensure that Marvin cxcalc is "
            f"properly installed as described at https://chemaxon.com/.)"
        )
    return result.stdout, result.stderr


def get_molecular_masses(
    molecules: pd.DataFrame, error_log: str
) -> pd.DataFrame:
    """
    Compute the dissociation constants and major microspecies at a defined pH.

    Parameters
    ----------
    molecules: pandas.DataFrame
        A list containing descriptions of the molecules (SMILES or InChI).
    error_log : str
        The base file path for error output.

    Returns
    -------
    view : pandas.DataFrame
        the input molecules data frame joined with the results calculated
        by ``cxcalc``.

    """
    if len(molecules) == 0:
        raise ValueError("Empty list of molecules, cannot calculate pKas.")

    args = [
        "--ignore-error",  # Continue with the next molecule on error.
        "mass",  # calculate molecular mass.
    ]

    output, error = run_cxcalc("\n".join(molecules["inchi"].tolist()), args)
    with open(f"{str(error_log)}.log", "w") as handle:
        # We skip the unhelpful Java stack traces that occur every second line.
        handle.write("\n".join(error.split("\n")[::2]))

    try:
        output_df = pd.read_csv(
            StringIO(output), sep="\t", header=0, index_col="id"
        )
    except pd.errors.EmptyDataError:
        raise ValueError("ChemAxon failed on all molecules")

    output_df.rename(columns={"Molecular weight": "mass"}, inplace=True)
    # We adjust the input index to start at 1 in order to be aligned with the
    # cxcalc standard and error output.
    molecules.index = range(1, 1 + molecules.shape[0])
    # Left join the results on index.
    return molecules.join(output_df)


def get_dissociation_constants(
    molecules: pd.DataFrame,
    error_log: str,
    num_acidic: int,
    num_basic: int,
    mid_ph: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute the dissociation constants and major microspecies at a defined pH.

    Parameters
    ----------
    molecules: pandas.DataFrame
        A list containing descriptions of the molecules (SMILES or InChI).
    error_log : str
        The base file path for error output.
    num_acidic : int
        The maximal number of acidic pKas to calculate.
    num_basic : int
        The maximal number of basic pKas to calculate.
    mid_ph : float
        The pH for which the major microspecies is calculated.

    Returns
    -------
    view, pka_columns : Tuple[pandas.DataFrame, List[str]]
        view - the input molecules data frame joined with the results
               calculated by ``cxcalc``.
        pka_columns - a list of the column names in 'view' containing pKa values

    """
    if len(molecules) == 0:
        raise ValueError("Empty list of molecules, cannot calculate pKas.")

    args = [
        "--ignore-error",  # Continue with the next molecule on error.
        "pka",  # pKa calculation.
        "--na",  # The number of acidic pKa values displayed.
        str(num_acidic),
        "--nb",  # The number of basic pKa values displayed.
        str(num_basic),
        "majorms",  # Major microspecies at given pH.
        "--majortautomer",  # Take the major tautomeric form.
        "true",
        "--pH",  # Get the major microspecies at this pH.
        str(mid_ph),
    ]

    cxcalc_input = "\n".join(molecules["inchi"].tolist())
    output, error = run_cxcalc(cxcalc_input, args)
    with open(f"{str(error_log)}.log", "w") as handle:
        # We skip the unhelpful Java stack traces that occur every second line.
        handle.write("\n".join(error.split("\n")[::2]))

    try:
        output_df = pd.read_csv(
            StringIO(output), sep="\t", header=0, index_col="id", dtype=str
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty output from cxcalc for {input}.")

    # We adjust the input index to start at 1 in order to be aligned with the
    # cxcalc standard and error output.
    molecules.index = range(1, 1 + molecules.shape[0])

    # Create a table output of molecules with calculation errors.
    # when only the pka run fails, the return value has "pka:FAILED" in the
    # first column:
    error_mask_pka = (output_df == "pka:FAILED").any(axis=1)

    # when only the majorms run fails, the return value has "majorms:FAILED" in
    # the second column:
    error_mask_majorms = (output_df == "majorms:FAILED").any(axis=1)

    # when both the pKa and major-ms runs fail:
    # The first acidic pKa column is empty if there was an error.
    # The second acidic pKa column contains an error description.
    error_mask_both = output_df["apKa1"].isnull() & output_df["apKa2"].notnull()

    error_mask = error_mask_pka | error_mask_majorms | error_mask_both
    if error_mask.any(axis=0):
        # write the error report to the log file
        error = molecules.copy()
        error["error"] = "PASSED"
        error.loc[output_df[error_mask_pka].index, "error"] = "pka:FAILED"
        error.loc[
            output_df[error_mask_majorms].index, "error"
        ] = "majorms:FAILED"
        error.loc[output_df[error_mask_both].index, "error"] = output_df.loc[
            error_mask_both, "apKa2"
        ]
        error.to_csv(f"{str(error_log)}.tsv", sep="\t")

    # Left join the results on index.
    result = molecules.join(output_df.loc[~error_mask, :], how="left")
    result.rename(columns={"major-ms": "major_ms"}, inplace=True)

    # create a function that retrieves the dissociation constants data from
    # a list. The relevant indices don't start from 0 (since the pKa values are
    # preceded by the columns in the input DataFrame "molecules").
    pka_columns = [f"apKa{i}" for i in range(1, 1 + num_acidic)] + [
        f"bpKa{i}" for i in range(1, 1 + num_basic)
    ]

    return result, pka_columns


def get_atom_bag(mol_format: str, mol_string: str) -> Dict[str, int]:
    """
    Compute the atom bag and the formal charge of a molecule.

    The formal charge is calculated by summing the formal charge of each atom
    in the molecule.

    Parameters
    ----------
    mol_format : str
        The format in which the molecule is given ("inchi", "smi", etc.)
    mol_string : str
        The molecular descriptor.

    Returns
    -------
    dict
        A dictionary of atom counts.

    """
    molecule = pybel.readstring(mol_format, mol_string)
    # Make all hydrogen atoms explicit so we can properly count them
    molecule.addh()

    # Count charges and atoms.
    atom_bag = dict()
    formal_charge = 0
    for atom in molecule.atoms:
        symbol = ATOMIC_NUMBER_TO_SYMBOL[atom.atomicnum]
        atom_bag[symbol] = atom_bag.get(symbol, 0) + 1
        formal_charge += atom.formalcharge

    # count all protons (from all types of atoms) in the molecule
    total_num_protons = sum(
        count * SYMBOL_TO_ATOMIC_NUMBER[elem]
        for elem, count in atom_bag.items()
    )

    # protons are the only positively charged particles, and electrons
    # are the only negatively charged ones, so the formal_charge is
    # exactly equal to the difference between these two counts.
    atom_bag["e-"] = total_num_protons - formal_charge
    return atom_bag
