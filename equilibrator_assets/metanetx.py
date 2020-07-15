# The MIT License (MIT)
#
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


"""Interact with the MetaNetX FTP server."""


import ftplib
import logging
from datetime import datetime
from os.path import isfile, join

import pandas as pd
from dateutil import parser


logger = logging.getLogger(__name__)


def update_file(
    path: str, ftp: ftplib.FTP, filename: str, last_checked: datetime
) -> None:
    """
    Retrieve a file from an FTP server if it is newer than a local version.

    Parameters
    ----------
    path : str
        Working directory where files are searched and stored.
    ftp : ftplib.FTP
        An instance of the FTP server with an open connection.
    filename : str
        The file to retrieve relative to the working directory on the server.
    last_checked : datetime
        The date and time when this script was last run.

    Raises
    ------
    ftplib.all_errors
        May raise any of the errors defined in ftplib if the file doesn't
        exist or the connection fails for some other reason.

    """
    # Modified time parsing according to https://stackoverflow.com/a/29027386.
    remote_modified = parser.parse(ftp.voidcmd(f"MDTM {filename}")[4:])
    logger.info(
        f"Remote file '{filename}' last modified on "
        f"{remote_modified.isoformat()}."
    )
    local_filename = join(path, filename)
    if (not isfile(local_filename)) or (remote_modified > last_checked):
        logger.info("Retrieving updated file version.")
        with open(local_filename, "wb") as file_handle:
            ftp.retrbinary(f"RETR {filename}", file_handle.write)
    else:
        logger.info("Local file version is up to date.")


def update_tables(
    path: str,
    host: str = "ftp.vital-it.ch",
    directory: str = "databases/metanetx/MNXref/latest",
) -> None:
    """Attempt to download updated files from the MetaNetX FTP server."""
    last = join(path, "last.txt")
    if isfile(last):
        with open(last) as file_handle:
            last_checked = parser.parse(file_handle.read().strip())
    else:
        last_checked = datetime.fromordinal(1)
    logger.info(f"Last ran on {last_checked.isoformat()}.")
    # The files could be downloaded asynchronously, aka, premature optimization.
    files = ["README.md", "chem_xref.tsv", "chem_prop.tsv"]
    with ftplib.FTP(host) as ftp:
        # Anonymous log in is still required.
        ftp.login()
        ftp.cwd(directory)
        for filename in files:
            try:
                update_file(path, ftp, filename, last_checked)
            except ftplib.all_errors as error:
                logger.error(f"Failed to update '{filename}'.", exc_info=error)
    with open(last, "w") as file_handle:
        file_handle.write(datetime.utcnow().isoformat())


def load_compound_cross_references(path: str) -> pd.DataFrame:
    """Load and transform the MetaNetX chemical cross-references."""
    filename = join(path, "chem_xref.tsv")
    df = pd.read_csv(filename, sep="\t", comment="#", header=None)
    df.columns = ["xref", "mnx_id", "evidence", "description"]
    # Cross references have a prefix.
    # We split the prefixes so that we know the actual data sources.
    df[["prefix", "accession"]] = df["xref"].str.split(":", n=1, expand=True)
    # MetaNetX identifiers themselves have no prefix. So we add it.
    mnx_mask = df["accession"].isnull()
    df.loc[mnx_mask, "accession"] = df.loc[mnx_mask, "prefix"]
    df.loc[mnx_mask, "prefix"] = "metanetx.chemical"
    df.loc[df["accession"] == "UNKNOWN", "accession"] = None
    logger.debug(df.head())
    return df


def load_compound_properties(path: str) -> pd.DataFrame:
    """Load and transform the MetaNetX chemical properties."""
    filename = join(path, "chem_prop.tsv")
    df = pd.read_csv(filename, sep="\t", comment="#", header=None)
    df.columns = [
        "mnx_id",
        "description",
        "formula",
        "charge",
        "mass",
        "inchi",
        "smiles",
        "source",
        "inchi_key",
    ]
    logger.debug(df.head())
    return df
