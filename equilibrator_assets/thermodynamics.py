# The MIT License (MIT)
#
# Copyright (c) 2018 Institute for Molecular Systems Biology, ETH Zurich.
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark
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


"""Enrich compounds with thermodynamic information."""


import logging
from typing import Iterator, List

import numpy as np
import pandas as pd
from equilibrator_cache import (
    Compound,
    CompoundIdentifier,
    CompoundMicrospecies,
    MagnesiumDissociationConstant,
    Registry,
)
from equilibrator_cache.compound_cache import PROTON_INCHI_KEY
from equilibrator_cache.thermodynamic_constants import (
    default_RT,
    standard_dg_formation_mg,
)
from importlib_resources import files
from sqlalchemy.orm import joinedload, sessionmaker
from tqdm import tqdm

from . import chemaxon
from . import data as assets_data


logger = logging.getLogger(__name__)
Session = sessionmaker()
LOG10 = np.log(10.0)


def populate_magnesium_dissociation_constants(
    session: Session, filename: str = "magnesium_pkds.csv"
) -> None:
    """Copy Mg2+ data from the resource file into the database.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        An active session in order to communicate with a SQL database.

    filename : str
        The name of the CSV file where the raw data is stored.

    """

    # Data from https://onlinelibrary.wiley.com/doi/abs/10.1002/bit.22309
    path = files(assets_data).joinpath(filename)
    mg_df = pd.read_csv(path)

    mappings = []
    for row in mg_df.itertuples():
        namespace, accession = row.compound_id.split(":")

        # fetch the compound object from the database
        query = session.query(Compound)
        query = query.outerjoin(CompoundIdentifier).filter(
            CompoundIdentifier.accession == accession
        )
        query = query.outerjoin(Registry).filter(
            Registry.namespace == namespace
        )
        compound = query.one_or_none()
        if compound is None:
            raise KeyError(f"The compound {row.compound_id} was not found")

        if np.isnan(row.n_h) or np.isnan(row.n_mg) or np.isnan(row.pk_d):
            continue
        mappings.append(
            {
                "compound_id": compound.id,
                "number_protons": int(row.n_h),
                "number_magnesiums": int(row.n_mg),
                "dissociation_constant": float(row.pk_d),
            }
        )

    session.bulk_insert_mappings(MagnesiumDissociationConstant, mappings)
    session.commit()


def map_compound_exception(
    session: Session, compound_id: str
) -> Iterator[Compound]:
    """Return the compound object associated to this compound_id.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        An active session in order to communicate with a SQL database.
    compound_id : str
        The ID can either be the InChI or [namespace]:[accession]

    Returns
    -------
    Compound or None

    """
    if compound_id.find("InChI") == 0:
        # search the DB by inchi
        query = session.query(Compound).filter(Compound.inchi == compound_id)
    else:
        # search the DB by accession
        namespace, accession = compound_id.split(":", 1)
        query = (
            session.query(Compound)
            .outerjoin(CompoundIdentifier)
            .filter(CompoundIdentifier.accession == accession)
            .outerjoin(Registry)
            .filter(Registry.namespace == namespace)
        )

    return query.all()


def populate_compound_exception(session: Session) -> None:
    """Update the database with data for compounds in the "exception" list.

    These compounds do not have known dissociation constants, but we still
    need to correctly set their nH and charge values in order to use them
    in Reactions.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        An active session in order to communicate with a SQL database.

    """
    mappings = []
    for compound_id, mol_data in chemaxon.COMPOUND_EXCEPTIONS.items():
        for compound in map_compound_exception(session, compound_id):
            mappings.append(
                {
                    "id": compound.id,
                    "atom_bag": mol_data.atom_bag,
                    "smiles": mol_data.smiles,
                    "dissociation_constants": [],
                }
            )
    session.bulk_update_mappings(Compound, mappings)
    session.commit()


def populate_dissociation_constants(
    session: Session,
    batch_size: int,
    error_log: str,
    num_acidic: int = 20,
    num_basic: int = 20,
    min_ph: float = 0.0,
    mid_ph: float = 7.0,
    max_ph: float = 14.0,
) -> None:
    """Populate the DB with proton dissociation constants using ChemAxon.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        An active session in order to communicate with a SQL database.
    batch_size : int
        The size of batches of compounds considered at a time.
    error_log : str
        The base file path for error output.
    num_acidic : int, optional
        The maximal number of acidic pKas to calculate (Default value = 20).
    num_basic : int, optional
        The maximal number of basic pKas to calculate (Default value = 20).
    min_ph : float, optional
        The minimal pH to consider (Default value = 0.0).
    mid_ph : float
        The pH for which the major pseudoisomer is calculated
        (Default value = 7.0).
    max_ph : float, optional
        The maximal pH to consider (Default value = 14.0).

    """

    # Select all compounds that do not yet have a list of dissociation constants
    query = session.query(Compound.id, Compound.inchi)
    query = query.filter(
        Compound.dissociation_constants.is_(None),
        Compound.inchi.isnot(None),
        Compound.mass.isnot(None),
    ).order_by(Compound.mass, Compound.inchi)

    molecules_without_pkas_df = pd.read_sql(query.statement, session.bind)

    with tqdm(
        total=len(molecules_without_pkas_df), desc="Analyzed"
    ) as progress_bar:
        for index in range(0, len(molecules_without_pkas_df), batch_size):
            view = molecules_without_pkas_df.iloc[index : index + batch_size, :]
            logger.debug(view)
            constants, pka_columns = chemaxon.get_dissociation_constants(
                view,
                f"{error_log}_batch_{index}",
                num_acidic=num_acidic,
                num_basic=num_basic,
                mid_ph=mid_ph,
            )
            compound_mappings = []
            for row in constants.itertuples(index=False):
                logger.debug(
                    f"Calculating MicroSpecies for compound "
                    f"#{row.id} - {row.major_ms}"
                )
                p_kas = [getattr(row, col) for col in pka_columns]
                p_kas = map(float, p_kas)
                p_kas = filter(lambda p_ka: min_ph < p_ka < max_ph, p_kas)
                dissociation_constants = sorted(p_kas, reverse=True)
                logger.debug(f"list of pKas: {dissociation_constants}")

                if pd.isnull(row.major_ms) or row.major_ms == "":
                    logger.warning(
                        "Failed to calculate major_ms string " f"for {row.id}"
                    )
                    compound_mappings.append(
                        {
                            "id": row.id,
                            "atom_bag": {},
                            "smiles": None,
                            "dissociation_constants": dissociation_constants,
                        }
                    )
                else:
                    atom_bag = chemaxon.get_atom_bag("smi", row.major_ms)
                    compound_mappings.append(
                        {
                            "id": row.id,
                            "atom_bag": atom_bag,
                            "smiles": row.major_ms,
                            "dissociation_constants": dissociation_constants,
                        }
                    )

            session.bulk_update_mappings(Compound, compound_mappings)
            session.commit()
            progress_bar.update(len(view))


def populate_microspecies(session: Session, mid_ph: float = 7.0) -> None:
    """
    Calculate dissociation constants and create microspecies.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        An active session in order to communicate with a SQL database.
    mid_ph : float
        The pH for which the major microspecies is calculated
        (Default value = 7.0).

    """

    # We only create microspecies for compounds that have dissociation_constants
    # (although it could also be an empty list) and an atom_bag (which we need
    # in order to determine the nH and z of the major microspecies).
    query = (
        session.query(Compound)
        .filter(
            Compound.dissociation_constants.isnot(None),
            Compound.atom_bag.isnot(None),
        )
        .options(joinedload(Compound.magnesium_dissociation_constants))
    )

    microspecies_mappings = []
    for compound in tqdm(query.all(), desc="Analyzed"):
        microspecies_mappings.extend(
            create_microspecies_mappings(compound, mid_ph)
        )
    session.bulk_insert_mappings(CompoundMicrospecies, microspecies_mappings)
    session.commit()


def create_microspecies_mappings(
    compound: Compound, mid_ph: float = 7.0
) -> List[dict]:
    """Create the mappings for the microspecies of a Compound.

    Parameters
    ----------
    compound : Compound
        A Compound object, where the atom_bag and dissociation_constants must
        not be None.
    mid_ph : float
        The pH for which the major microspecies is calculated
        (Default value = 7.0).

    Returns
    -------
    list
        A list of mappings for creating the entries in the compound_microspecies
        table.

    """
    # We add an exception for H+ (and put z = nH = 0) in order to
    # eliminate its effect of the Legendre transform.
    if compound.inchi_key == PROTON_INCHI_KEY:
        return [
            {
                "compound_id": compound.id,
                "charge": 0,
                "number_protons": 0,
                "number_magnesiums": 0,
                "ddg_over_rt": 0.0,
                "is_major": True,
            }
        ]

    # Find the index of the major microspecies, by counting how many pKas there
    # are in the range between the given pH and the maximum (typically, 7 - 14).
    # Then make a list of the nH and charge values for all the microspecies
    if not compound.dissociation_constants:
        num_species = 1
        major_ms_index = 0
    else:
        num_species = len(compound.dissociation_constants) + 1
        major_ms_index = sum(
            (1 for p_ka in compound.dissociation_constants if p_ka > mid_ph)
        )

    major_ms_num_protons = compound.atom_bag.get("H", 0)
    major_ms_charge = compound.net_charge

    microspecies_mappings = dict()
    for i in range(num_species):
        charge = i - major_ms_index + major_ms_charge
        num_protons = i - major_ms_index + major_ms_num_protons

        if i == major_ms_index:
            ddg_over_rt = 0.0
        elif i < major_ms_index:
            ddg_over_rt = (
                sum(compound.dissociation_constants[i:major_ms_index]) * LOG10
            )
        elif i > major_ms_index:
            ddg_over_rt = (
                -sum(compound.dissociation_constants[major_ms_index:i]) * LOG10
            )
        else:
            raise IndexError("Major microspecies index mismatch.")

        microspecies_mappings[(num_protons, 0)] = {
            "compound_id": compound.id,
            "charge": charge,
            "number_protons": num_protons,
            "number_magnesiums": 0,
            "ddg_over_rt": ddg_over_rt,
            "is_major": i == major_ms_index,
        }

    standard_dg_formation_mg_over_rt = (
        standard_dg_formation_mg / default_RT
    ).m_as("")

    # iterate through all Mg2+ dissociation constants in an order where
    # the ones with fewest Mg2+ are first, so that their reference MS will
    # already be in `microspecies_mappings`. If there is a gap in this layered
    # approach, we raise an Exception.
    for mg_diss in sorted(
        compound.magnesium_dissociation_constants,
        key=lambda x: (x.number_magnesiums, x.number_protons),
    ):
        dissociation_constant = mg_diss.dissociation_constant
        num_protons = mg_diss.number_protons
        num_magnesiums = mg_diss.number_magnesiums

        # find the reference MS, but looking for one with one less Mg2+ ion.
        # it should already be in the `microspecies_mappings` dictionary
        # since we sorted the pKds by increasing order of Mg2+.
        try:
            ref_ms = microspecies_mappings[(num_protons, num_magnesiums - 1)]
        except KeyError:
            raise KeyError(
                f"Could not find the reference microspecies for "
                f"the [nH={num_protons}, nMg={num_magnesiums}] "
                f"microspecies, for the compound {compound.id}"
            )

        charge = ref_ms["charge"] + 2
        ddg_over_rt = (
            ref_ms["ddg_over_rt"]
            + standard_dg_formation_mg_over_rt
            - dissociation_constant * LOG10
        )

        microspecies_mappings[(num_protons, num_magnesiums)] = {
            "compound_id": compound.id,
            "charge": charge,
            "number_protons": num_protons,
            "number_magnesiums": num_magnesiums,
            "ddg_over_rt": ddg_over_rt,
            "is_major": False,
        }

    return sorted(
        microspecies_mappings.values(),
        key=lambda x: (x["number_magnesiums"], x["number_protons"]),
    )
