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


"""Populate compound information."""


import logging
import typing

import pandas as pd
import requests
from equilibrator_cache import Compound, CompoundIdentifier, Registry
from openbabel import pybel
from openbabel.pybel import readstring
from sqlalchemy import exists, or_
from sqlalchemy.orm.session import Session
from tqdm import tqdm

from .chemaxon import get_atom_bag, get_molecular_masses
from .registry import get_mnx_mapping


logger = logging.getLogger(__name__)


def inchi_to_inchi_key(inchi: str) -> str:
    """Return an InChIKey for the given InChI string."""
    return readstring("inchi", inchi).write("inchikey")


def create_compound_object(row: typing.NamedTuple) -> dict:
    """Generate a minimal compound object for bulk insertion."""
    return {
        "mnx_id": row.mnx_id,
        "inchi_key": row.inchi_key,
        "inchi": row.inchi,
        "smiles": row.smiles,
        "mass": row.mass,
    }


def create_compound_identifier_objects(
    compound_id: int,
    cross_references: pd.DataFrame,
    prefix2registry: typing.Dict[str, Registry],
) -> typing.List[dict]:
    """Generate compound cross-references for bulk insertion."""
    identifiers = []
    for row in cross_references.itertuples(index=False):
        registry = prefix2registry[row.prefix]
        if registry is None:
            continue
        if registry.is_prefixed:
            accession = f"{row.prefix.upper()}:{row.accession}"
        else:
            accession = row.accession
        identifiers.append(
            {
                "compound_id": compound_id,
                "registry_id": registry.id,
                "accession": accession,
            }
        )
    registry = prefix2registry["synonyms"]
    names = cross_references.loc[
        cross_references["description"].notnull(), "description"
    ].unique()
    for name in names:
        identifiers.append(
            {
                "compound_id": compound_id,
                "registry_id": registry.id,
                "accession": name,
            }
        )

    return identifiers


def populate_compounds(
    session: Session,
    properties: pd.DataFrame,
    cross_references: pd.DataFrame,
    batch_size: int,
) -> None:
    """
    Populate the compound and identifier tables using information from MetaNetX.

    Parameters
    ----------
    session : SQLAlchemy.Session
    properties : pd.DataFrame
    cross_references : pd.DataFrame
    batch_size : int

    Warnings
    --------
    The function uses bulk inserts for performance and thus assumes empty
    tables. Do **not** use it for updating content.

    """
    prefix2registry = get_mnx_mapping(session)
    grouped_xref = cross_references[
        (cross_references["prefix"] != "metanetx.chemical")
        & cross_references["accession"].notnull()
    ].groupby("mnx_id", sort=False)
    with tqdm(total=len(properties), desc="Compounds") as pbar:
        for index in range(0, len(properties), batch_size):
            compounds = [
                create_compound_object(row)
                for row in properties[index : index + batch_size].itertuples(
                    index=False
                )
            ]
            session.bulk_insert_mappings(Compound, compounds)
            session.commit()
            pbar.update(len(compounds))
    with tqdm(total=len(properties), desc="Cross-References") as pbar:
        for index in range(0, len(properties), batch_size):
            identifiers = []
            counter = 0
            for row in session.query(Compound.id, Compound.mnx_id).slice(
                index, index + batch_size
            ):
                try:
                    identifiers.extend(
                        create_compound_identifier_objects(
                            row.id,
                            grouped_xref.get_group(row.mnx_id),
                            prefix2registry,
                        )
                    )
                except KeyError:
                    logger.debug(
                        "Compound '%s' has no cross-references.", row.mnx_id
                    )
                counter += 1
            session.bulk_insert_mappings(CompoundIdentifier, identifiers)
            session.commit()
            pbar.update(counter)


def populate_additional_compounds(session: Session, filename: str) -> None:
    """Populate the database with additional compounds."""
    additional_compound_df = pd.read_csv(filename)
    additional_compound_df[additional_compound_df.isnull()] = None
    name_registry = (
        session.query(Registry).filter_by(namespace="synonyms").one()
    )
    coco_registry = session.query(Registry).filter_by(namespace="coco").one()
    for row in tqdm(additional_compound_df.itertuples(index=False)):
        if session.query(exists().where(Compound.inchi == row.inchi)).scalar():
            continue
        logger.info(f"Adding non-MetaNetX compound: {row.name}")
        compound = Compound(
            mnx_id=row.mnx_id,
            inchi=row.inchi,
            inchi_key=inchi_to_inchi_key(row.inchi),
        )
        identifiers = []
        if row.coco_id:
            print(repr(row.coco_id))
            identifiers.append(
                CompoundIdentifier(
                    registry=coco_registry, accession=row.coco_id
                )
            )
        if row.name:
            identifiers.append(
                CompoundIdentifier(registry=name_registry, accession=row.name)
            )
        compound.identifiers = identifiers
        session.add(compound)
    session.commit()


def get_kegg_inchi(accession: str) -> typing.Tuple[str, str]:
    """Retrieve a compound's InChI from KEGG if it exists."""
    try:
        response = requests.get(
            "http://rest.kegg.jp/get/cpd:{}/mol".format(accession)
        )
        response.raise_for_status()
        molstring = str(response.text)
    except requests.exceptions.HTTPError:
        return None, None

    mol = pybel.readstring("mol", molstring)
    inchi = mol.write("InChI")

    if inchi:
        inchi_key = mol.write("InChIKey")
        return inchi, inchi_key
    else:
        return None, None


def fetch_kegg_missing_inchis(session: Session):
    """Retrieve InChI strings from KEGG for all compounds missing those."""
    kegg_registry = (
        session.query(Registry)
        .filter(Registry.namespace == "kegg")
        .one_or_none()
    )

    if kegg_registry is None:
        raise Exception("Cannot find KEGG in the registry table")

    # find all compounds that are in KEGG but don't have an InChI in the
    # MetaNetX database, but the linked KEGG compound does have one.
    query = session.query(Compound, CompoundIdentifier.accession)
    query = query.join(CompoundIdentifier)
    query = query.filter(CompoundIdentifier.registry_id == kegg_registry.id)
    query = query.group_by(Compound.id)
    query = query.filter(
        Compound.inchi.is_(None), CompoundIdentifier.accession.like("C%")
    )

    # For each of these compounds, try to fetch the structure from KEGG using
    # the REST API
    for compound, accession in tqdm(query, total=query.count(), desc="Filled"):
        inchi, inchi_key = get_kegg_inchi(accession)
        if inchi is not None:
            compound.inchi = inchi
            compound.inchi_key = inchi_key
            session.commit()


def fill_missing_values(
    session: Session, only_kegg: bool, batch_size: int, error_log: str
) -> None:
    """
    Complete missing mass and/or atom bag information from InChI strings.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        An active session in order to communicate with a SQL database.
    only_kegg : bool
        Calculate thermodynamic information for compounds contained in KEGG
        only.
    batch_size : int
        The size of batches of compounds considered at a time.
    error_log : str
        The base file path for error output.

    """
    query = session.query(Compound.id, Compound.mnx_id, Compound.inchi)

    if only_kegg:
        # Filter compounds in KEGG or COCO (additional compounds for
        # component-contribution)
        query = query.join(CompoundIdentifier).join(Registry)
        query = query.filter(Registry.namespace.in_(("kegg", "coco")))
        query = query.group_by(Compound.id)

    query = query.filter(
        Compound.inchi.isnot(None),
        or_(Compound.mass.is_(None), Compound.atom_bag.is_(None)),
    )

    logger.debug("calculating mass for compounds with missing values")
    input_df = pd.read_sql_query(query.statement, query.session.bind)

    with tqdm(total=len(input_df), desc="Analyzed") as pbar:
        for index in range(0, len(input_df), batch_size):
            view = input_df.iloc[index : index + batch_size, :]
            try:
                view = get_molecular_masses(view, f"{error_log}_batch_{index}")
                compounds = []
                for row in view.itertuples(index=False):
                    try:
                        atom_bag = get_atom_bag("inchi", row.inchi)
                    except OSError as e:
                        logger.warning(str(e))
                        atom_bag = {}
                    compounds.append(
                        {"id": row.id, "mass": row.mass, "atom_bag": atom_bag}
                    )
                session.bulk_update_mappings(Compound, compounds)
                session.commit()
            except ValueError as e:
                logger.warning(str(e))

            pbar.update(len(view))
