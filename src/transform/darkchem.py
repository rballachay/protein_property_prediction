import multiprocessing as mp
import os
import re

# from openbabel import pybel
from os.path import abspath, dirname

import numpy as np
import pandas as pd
from data.AADesc import BLOSUM62, VHSE, Z3, Z5

from src.data.fixtures import SMI

"""
All functions in section taken AND ADADPTED from :
    https://github.com/pnnl/darkchem 
"""


def generate_encoded_smiles(
    df, name="smiles_strings", output="", canonical=False, shuffle=True, haslabels=True
):
    """
    Assumes dataframe with InChI or SMILES columns and
    optionally a Formula column.  Any additional columns will
    be propagated as labels for prediction.
    """
    if haslabels:
        check = _check_for_encoded_smiles(name)
    else:
        name += "_prediction"
        check = _check_for_encoded_smiles(name)
    if check[0]:
        arr = check[1]
    else:
        # already converted
        if "SMILES" in df.columns and canonical is True:
            pass
        elif "SMILES" in df.columns:
            df["SMILES"] = canonicalize(df["SMILES"].values)
            df.to_csv(
                os.path.join(output, "data/%s_canonical.tsv" % name),
                index=False,
                sep="\t",
            )
        # error
        else:
            raise KeyError('Dataframe must have an "InChI" or "SMILES" column.')

        # vectorize
        # TODO: configurable max length
        # TODO: configurable charsest
        vectors = np.vstack(vectorize(df["SMILES"].values))
        # vectors = np.where(np.all(vectors == 0, axis=1, keepdims=True), np.nan, vectors)

        df["vec"] = vectors.tolist()

        # df.dropna(how='any', axis=0, inplace=True)
        arr = np.vstack(df["vec"].values)

        arr = np.nan_to_num(arr)

        # labels
        if "InChI" in df.columns:
            labels = df.drop(columns=["InChI", "SMILES", "vec"])
        else:
            labels = df.drop(columns=["SMILES", "vec"])

        # save
        np.save(os.path.join(output, "data/%s.npy" % name), arr)

        if len(labels.columns) > 0:
            np.save(os.path.join(output, "data/%s_labels.npy" % name), labels.values)

    return arr


def struct2vec(struct, charset=SMI, max_length=117):
    """
    Takes in structure and returns the encoded version using the default
    or passed in charset.

    Parameters
    ----------
    struct : str
        Structure of compound, represented as an InChI or SMILES string.

    charset : list, optional
        Character set used for encoding.

    max_length : int, optional
        Maximum length of encoding.

    Returns
    -------
    vec : unit8 array
        Encoded structure
    """

    output = _smi2vec(struct, charset, max_length)

    if output is None:
        return np.zeros(max_length)
    else:
        return np.array(output, dtype=np.uint8)


def vectorize(smiles, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as p:
        return p.map(struct2vec, smiles)


def _canonicalize(smi):
    """Canonicalizes SMILES string."""
    return pybel.readstring("smi", smi).write("can").strip()


def canonicalize(smiles, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as p:
        return p.map(_canonicalize, smiles)


def _parse_formula(formula, targets="CHNOPS"):
    atoms = re.findall(r"([A-Z][a-z]?)(\d+)?", formula)

    d = {k: v for k, v in atoms if k in targets}

    for k in targets:
        if k in d.keys():
            if d[k] == "":
                d[k] = 1
            else:
                d[k] = int(d[k])
        else:
            d[k] = 0
    return d


def parse_formulas(formulas, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as p:
        return pd.DataFrame(data=p.map(_parse_formula, formulas))


def _check_for_encoded_smiles(filename):
    filename += ".npy"
    file_path = f"data{os.sep}{filename}"
    file_path = file_path
    if os.path.isfile(file_path):
        embedding = np.load(file_path)
        return (True, embedding)
    else:
        return (False, None)


def _encode(string, charset):
    """
    Encodes string with a given charset.
    Returns None if s contains illegal characters
    If s is empty, returns an empty array
    """

    if pd.isna(string):
        return np.array([])

    vec = np.zeros(len(string))
    for i in range(len(string)):
        s = string[i]
        if s in charset:
            vec[i] = charset.index(s)
        else:  # Illegal character in s
            print("ILLEGAL CHARACTER:")
            print(s)
            return None

    return vec


def _add_padding(l, length):
    """
    Adds padding to l to make it size length.
    """

    ltemp = list(l)
    ltemp.extend([0] * (length - len(ltemp)))
    return ltemp


def _smi2vec(smi, charset, max_length=117):
    # Encode SMILES
    vec = _encode(smi, charset)

    # Check for errors
    if vec is None:
        # print('%s skipped, contains illegal characters' % smi)
        return None
    if len(vec) > max_length:
        print("%s skipped, too long" % smi)
        print("length = %i" % len(smi))
        return None

    # Add padding
    vec = _add_padding(vec, max_length)

    # Return encoded InChI
    return vec
