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


"""Populate and fix information on identifier registries."""


import logging

import pandas as pd
import requests
from equilibrator_cache import Registry
from importlib_resources import open_text
from sqlalchemy.orm.session import Session
from tqdm import tqdm

import equilibrator_assets.data


logger = logging.getLogger(__name__)


def load_rest_data(url: str) -> dict:
    """Return the object from a simple GET request."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def create_registry(entry: dict) -> Registry:
    """Create a new registry."""
    registry = Registry(
        namespace=entry["prefix"],
        name=entry.get("name"),
        pattern=entry.get("pattern"),
        identifier=entry.get("id"),
        url=entry.get("url"),
        is_prefixed=bool(entry.get("prefixed", False)),
    )
    try:
        registry.access_url = entry.get("resources", [])[0]["accessURL"]
    except IndexError:
        pass
    return registry


def patch_registry(session: Session, prefix: str) -> None:
    """Create an entry similar to a registry defined by identifiers.org."""
    entry = {"namespace": prefix, "is_prefixed": False}
    if prefix == "envipath":
        entry["name"] = "enviPath"
        entry["pattern"] = r"^.+$"
        entry["access_url"] = "https://envipath.org/package/{$id}"
    elif prefix == "synonyms":
        entry["name"] = "Synonyms"
        entry["pattern"] = r"^.+$"
    elif prefix == "coco":
        entry["name"] = "Component-Contribution Metabolite"
        entry["pattern"] = r"^COCOM\d+$"
    else:
        raise ValueError(f"Unknown registry prefix '{prefix}'.")
    # We use low-level insertion in order to circumvent the validation.
    session.execute(Registry.__table__.insert(), [entry])


def populate_registries(
    session: Session, cross_references: pd.DataFrame
) -> None:
    """Populate the database with registry information."""
    prefixes = set(cross_references["prefix"].unique())
    # Add dummy registry for names.
    prefixes.add("synonyms")
    # Add dummy registry for additional compounds.
    prefixes.add("coco")
    # remove deprecated
    prefixes.remove("deprecated")

    # Load registry information from identifiers.org.
    collections = {
        c["prefix"]: c["id"]
        for c in load_rest_data("http://identifiers.org/rest/collections")
    }
    with open_text(equilibrator_assets.data, "prefix_mapping.tsv") as handle:
        mapping = {
            row.mnx_prefix: row.identifiers_prefix
            for row in pd.read_table(handle, header=0).itertuples(index=False)
        }
    for prefix in tqdm(prefixes, total=len(prefixes), desc="Registry"):
        id_prefix = mapping.get(prefix)

        if id_prefix is None:
            logger.warning(f"Prefix '{prefix}' not present in mapping.")
            patch_registry(session, prefix)
            continue
        elif id_prefix not in collections:
            logger.error(
                f"Prefix '{prefix}' does not exist at identifiers.org."
            )
            continue
        else:
            miriam = collections[id_prefix]
            entry = load_rest_data(
                f"http://identifiers.org/rest/collections/{miriam}"
            )
        session.add(create_registry(entry))
    session.commit()


def get_mnx_mapping(session: Session):
    """Return a mapping from MetaNetX prefixes to MIRIAM registries."""
    with open_text(equilibrator_assets.data, "prefix_mapping.tsv") as handle:
        mapping = {
            row.mnx_prefix: session.query(Registry)
            .filter_by(namespace=row.identifiers_prefix)
            .one_or_none()
            for row in pd.read_csv(handle, sep="\t", header=0).itertuples(
                index=False
            )
        }
    mapping["envipath"] = (
        session.query(Registry).filter_by(namespace="envipath").one_or_none()
    )
    mapping["synonyms"] = (
        session.query(Registry).filter_by(namespace="synonyms").one_or_none()
    )
    mapping["deprecated"] = (
        session.query(Registry)
        .filter_by(namespace="metanetx.chemical")
        .one_or_none()
    )
    return mapping
