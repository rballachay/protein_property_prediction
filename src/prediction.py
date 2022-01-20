import os
import pandas as pd
from src.transform.preprocess import AMINO_ACIDS
from rdkit import Chem


def make_prediction_data():
    file_name = "data/prediction_data.csv"
    if os.path.isfile(file_name):
        aa_data = pd.read_csv(file_name, index_col=0)
    else:
        AA_list = _make_all_tetramers()
        AA_smiles = [Chem.MolToSmiles(Chem.MolFromFASTA(a)) for a in AA_list]
        aa_data = pd.DataFrame()
        aa_data["seq"] = AA_list
        aa_data["SMILES"] = AA_smiles
        aa_data.to_csv(file_name)

    return aa_data


def _make_all_tetramers(AMINO_ACIDS=AMINO_ACIDS):
    AA_list = []
    for AA1 in AMINO_ACIDS:
        for AA2 in AMINO_ACIDS:
            for AA3 in AMINO_ACIDS:
                for AA4 in AMINO_ACIDS:
                    AA_list.append(AA1 + AA2 + AA3 + AA4)

    return AA_list
