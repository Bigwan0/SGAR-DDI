import os
from pathlib import Path
import itertools
from collections import defaultdict
from operator import neg
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pandas as pd
import numpy as np
from functional_groups import FunctionalGroupDetector

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')

# Initialize global variables for functional groups
FG_DETECTOR = FunctionalGroupDetector()
FG_ENRICHMENT_SCORES = {}
DRUG_TO_FUNCTIONAL_GROUPS = {}
DRUG_TO_INTRA_ENRICHMENT = {}  # Cache for intra-molecular enrichment scores

def load_fg_enrichment_scores(csv_path):
    global FG_ENRICHMENT_SCORES

    FG_ENRICHMENT_SCORES = {}

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        FG_ENRICHMENT_SCORES[(row['fg1'], row['fg2'])] = float(row['enrichment'])
        FG_ENRICHMENT_SCORES[(row['fg2'], row['fg1'])] = float(row['enrichment'])

    return {
        "rows": len(df),
        "min_enrichment": float(df["enrichment"].min()),
        "median_enrichment": float(df["enrichment"].median()),
        "max_enrichment": float(df["enrichment"].max()),
    }

def get_molecule_functional_groups(mol):
    """Get functional groups for a molecule"""
    if mol is None:
        return []
    return [fg[0] for fg in FG_DETECTOR.detect_functional_groups(mol)]

def precompute_functional_groups():
    global DRUG_TO_FUNCTIONAL_GROUPS, DRUG_TO_INTRA_ENRICHMENT
    print("Precomputing functional groups and enrichment scores...")
    
    for drug_id, mol in drug_to_mol_graph.items():
        # Store functional groups
        DRUG_TO_FUNCTIONAL_GROUPS[drug_id] = get_molecule_functional_groups(mol)
        
        # Precompute intra-molecular enrichment score for this drug
        functional_groups = DRUG_TO_FUNCTIONAL_GROUPS[drug_id]
        max_enrichment = 1.0
        
        if functional_groups:
            for i, fg1 in enumerate(functional_groups):
                for j, fg2 in enumerate(functional_groups):
                    if i != j:  # Don't compare same functional group with itself
                        pair_key = (fg1, fg2)
                        reverse_pair_key = (fg2, fg1)
                        
                        if pair_key in FG_ENRICHMENT_SCORES:
                            max_enrichment = max(max_enrichment, FG_ENRICHMENT_SCORES[pair_key])
                        elif reverse_pair_key in FG_ENRICHMENT_SCORES:
                            max_enrichment = max(max_enrichment, FG_ENRICHMENT_SCORES[reverse_pair_key])
        
        DRUG_TO_INTRA_ENRICHMENT[drug_id] = max_enrichment
    
    print(f"Precomputed enrichment scores for {len(DRUG_TO_INTRA_ENRICHMENT)} drugs.")


# ------------------------------------------------------------
# TWOSIDES molecular-structure source
#
# Resolution order:
#   1. TWOSIDES_DATA_DIR environment variable
#   2. twosides_test/twosides/
#   3. current AutoDL official FG-DDI dataset location
# ------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

_SMILES_CANDIDATES = []

_env_data_dir = os.environ.get(
    "TWOSIDES_DATA_DIR"
)

if _env_data_dir:
    _SMILES_CANDIDATES.append(
        Path(_env_data_dir)
        / "drug_smiles.csv"
    )

_SMILES_CANDIDATES.extend([
    (
        _THIS_DIR
        / "twosides"
        / "drug_smiles.csv"
    ),
    Path(
        "/root/autodl-tmp/FG-DDI/"
        "DSN-DDI-dataset/dataset/"
        "twosides/drug_smiles.csv"
    ),
])

_SMILES_FILE = None

for _candidate in _SMILES_CANDIDATES:
    if _candidate.exists():
        _SMILES_FILE = _candidate.resolve()
        break

if _SMILES_FILE is None:
    raise FileNotFoundError(
        "TWOSIDES drug_smiles.csv not found. "
        "Set TWOSIDES_DATA_DIR or provide "
        "twosides_test/twosides/drug_smiles.csv."
    )

df_drugs_smiles = pd.read_csv(
    _SMILES_FILE
)

DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]

drug_to_mol_graph = {id:Chem.MolFromSmiles(smiles.strip()) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])}


# Gettings information and features of atoms
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})

# Use the correct API to avoid deprecation warnings
try:
    # Try the newer API with which parameter
    max_valence = max(a.GetValence(which=Chem.rdchem.Atom.ValenceType.IMPLICIT) for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
except (AttributeError, TypeError):
    try:
        # Try alternative newer API
        max_valence = max(a.GetTotalValence() - a.GetExplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
    except AttributeError:
        # Fall back to the older API
        max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))

max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):
    
    # Calculate implicit valence using the correct API to avoid deprecation warnings
    try:
        # Try the newer API with which parameter
        implicit_valence = atom.GetValence(which=Chem.rdchem.Atom.ValenceType.IMPLICIT)
    except (AttributeError, TypeError):
        try:
            # Try alternative newer API
            implicit_valence = atom.GetTotalValence() - atom.GetExplicitValence()
        except AttributeError:
            # Fall back to the older API
            implicit_valence = atom.GetImplicitValence()

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, implicit_valence, 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_atom_features(atom, mode='one_hot'):
    
    # Calculate implicit valence using the correct API to avoid deprecation warnings
    try:
        # Try the newer API with which parameter
        implicit_valence = atom.GetValence(which=Chem.rdchem.Atom.ValenceType.IMPLICIT)
    except (AttributeError, TypeError):
        try:
            # Try alternative newer API
            implicit_valence = atom.GetTotalValence() - atom.GetExplicitValence()
        except AttributeError:
            # Fall back to the older API
            implicit_valence = atom.GetImplicitValence()

    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(implicit_valence, AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([implicit_valence]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature

def get_mol_edge_list_and_feat_mtx(mol_graph):
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list 
    # Fix PyTorch deprecation warning: use .mT instead of .T for 2D tensors
    return undirected_edge_list.mT if undirected_edge_list.dim() == 2 else undirected_edge_list.T, n_features

def compute_intra_molecular_enrichment_weights(drug_id, num_edges):
    """
    Fast computation using precomputed enrichment scores
    
    Args:
        drug_id: Drug identifier
        num_edges: Number of edges in the molecule
    
    Returns:
        torch.Tensor: enrichment weights for each edge
    """
    if num_edges == 0:
        return torch.ones(0)
    
    # Use precomputed enrichment score
    enrichment_score = DRUG_TO_INTRA_ENRICHMENT.get(drug_id, 1.0)
    
    # Apply the same enrichment value to all intra-molecular edges
    return torch.ones(num_edges) * enrichment_score

def get_bipartite_graph(mol_graph_1,mol_graph_2):
    x1 = np.arange(0,len(mol_graph_1.GetAtoms()))
    x2 = np.arange(0,len(mol_graph_2.GetAtoms()))
    edge_list = torch.LongTensor(np.meshgrid(x1,x2))
    edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
    return edge_list

MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol) 
                                for drug_id, mol in drug_id_mol_graph_tup}

MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])



##### DDI statistics and counting #######

ALL_DRUG_IDS, _ = zip(*drug_id_mol_graph_tup)
ALL_DRUG_IDS = np.array(list(set(ALL_DRUG_IDS)))

# These statistics MUST be built from the current training split only.
ALL_TRUE_H_WITH_TR = defaultdict(list)
ALL_TRUE_T_WITH_HR = defaultdict(list)
FREQ_REL = defaultdict(int)
ALL_H_WITH_R = defaultdict(dict)
ALL_T_WITH_R = defaultdict(dict)
ALL_TAIL_PER_HEAD = {}
ALL_HEAD_PER_TAIL = {}


def configure_ddi_statistics(tri_list):
    """
    Configure TWOSIDES training-only negative-sampling
    statistics.

    The clean TWOSIDES protocol uses tail corruption only:

        (h, t, r) -> (h, n, r)

    For positive rejection, drug-pair semantics are unordered.
    Therefore, if (A, B, r) is a training positive, both

        B in TRUE_TAILS[(A, r)]
        A in TRUE_TAILS[(B, r)]

    are registered.

    Validation/test triples must never be passed here.
    """
    global ALL_TRUE_H_WITH_TR
    global ALL_TRUE_T_WITH_HR
    global FREQ_REL
    global ALL_H_WITH_R
    global ALL_T_WITH_R
    global ALL_TAIL_PER_HEAD
    global ALL_HEAD_PER_TAIL

    true_h_with_tr = defaultdict(list)
    true_t_with_hr = defaultdict(list)

    freq_rel = defaultdict(int)
    h_with_r = defaultdict(dict)
    t_with_r = defaultdict(dict)

    n_triples = 0

    for h, t, r in tri_list:

        h = str(h)
        t = str(t)
        r = int(r)

        # Unordered-pair positive rejection.
        true_t_with_hr[
            (h, r)
        ].append(t)

        true_t_with_hr[
            (t, r)
        ].append(h)

        # Retain the symmetric reverse map for compatibility
        # with the common clean DrugDataset implementation.
        true_h_with_tr[
            (t, r)
        ].append(h)

        true_h_with_tr[
            (h, r)
        ].append(t)

        freq_rel[r] += 1

        h_with_r[r][h] = 1
        h_with_r[r][t] = 1

        t_with_r[r][h] = 1
        t_with_r[r][t] = 1

        n_triples += 1

    for key in list(
        true_h_with_tr.keys()
    ):
        true_h_with_tr[key] = np.array(
            sorted(
                set(
                    true_h_with_tr[key]
                )
            ),
            dtype=object,
        )

    for key in list(
        true_t_with_hr.keys()
    ):
        true_t_with_hr[key] = np.array(
            sorted(
                set(
                    true_t_with_hr[key]
                )
            ),
            dtype=object,
        )

    head_per_tail = {}
    tail_per_head = {}

    for r in freq_rel:

        h_with_r[r] = np.array(
            sorted(
                h_with_r[r].keys()
            ),
            dtype=object,
        )

        t_with_r[r] = np.array(
            sorted(
                t_with_r[r].keys()
            ),
            dtype=object,
        )

        # Kept only for API compatibility.
        # TWOSIDES clean training does NOT use Bernoulli
        # head/tail corruption.
        denom_h = max(
            len(h_with_r[r]),
            1,
        )

        denom_t = max(
            len(t_with_r[r]),
            1,
        )

        tail_per_head[r] = (
            freq_rel[r]
            / denom_h
        )

        head_per_tail[r] = (
            freq_rel[r]
            / denom_t
        )

    ALL_TRUE_H_WITH_TR = (
        true_h_with_tr
    )

    ALL_TRUE_T_WITH_HR = (
        true_t_with_hr
    )

    FREQ_REL = freq_rel
    ALL_H_WITH_R = h_with_r
    ALL_T_WITH_R = t_with_r

    ALL_TAIL_PER_HEAD = (
        tail_per_head
    )

    ALL_HEAD_PER_TAIL = (
        head_per_tail
    )

    return {
        "num_triples":
            n_triples,

        "num_relations":
            len(freq_rel),

        "num_true_hr_keys":
            len(
                true_t_with_hr
            ),

        "num_true_tr_keys":
            len(
                true_h_with_tr
            ),

        "sampling":
            "tail-only",

        "pair_semantics":
            "unordered",
    }
#######    ****** ###############
class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, enrichment_weights=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.enrichment_weights = enrichment_weights

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class DrugDataset(Dataset):
    def __init__(
        self,
        tri_list,
        ratio=1.0,
        neg_ent=1,
        disjoint_split=True,
        shuffle=True,
        fixed_negative_file=None,
    ):
        self.neg_ent = neg_ent
        self.tri_list = []
        self.ratio = ratio
        self.fixed_negative_map = None

        for h, t, r in tri_list:
            if (
                h in MOL_EDGE_LIST_FEAT_MTX
                and t in MOL_EDGE_LIST_FEAT_MTX
            ):
                self.tri_list.append(
                    (h, t, int(r))
                )

        if disjoint_split:
            d1, d2, *_ = zip(*self.tri_list)
            self.drug_ids = np.array(
                list(set(d1 + d2))
            )
        else:
            self.drug_ids = ALL_DRUG_IDS.copy()

        if shuffle:
            random.shuffle(self.tri_list)

        limit = math.ceil(
            len(self.tri_list) * ratio
        )
        self.tri_list = self.tri_list[:limit]

        # ----------------------------------------------------
        # Fixed negatives are used ONLY for validation/test.
        # Training leaves fixed_negative_file=None and keeps
        # the original dynamic negative sampling behavior.
        # ----------------------------------------------------
        if fixed_negative_file is not None:

            fixed_df = pd.read_csv(
                fixed_negative_file
            )

            required_columns = {
                "pos_d1",
                "pos_d2",
                "type",
                "neg_d1",
                "neg_d2",
            }

            missing = (
                required_columns
                - set(fixed_df.columns)
            )

            if missing:
                raise ValueError(
                    "Missing fixed-negative columns: "
                    f"{sorted(missing)}"
                )

            self.fixed_negative_map = {}

            for row in fixed_df.itertuples(
                index=False
            ):
                key = (
                    str(row.pos_d1),
                    str(row.pos_d2),
                    int(row.type),
                )

                if key in self.fixed_negative_map:
                    raise ValueError(
                        "Duplicate fixed-negative key: "
                        f"{key}"
                    )

                self.fixed_negative_map[key] = (
                    str(row.neg_d1),
                    str(row.neg_d2),
                )

            expected_keys = {
                (
                    str(h),
                    str(t),
                    int(r),
                )
                for h, t, r in self.tri_list
            }

            fixed_keys = set(
                self.fixed_negative_map
            )

            if expected_keys != fixed_keys:
                missing_keys = (
                    expected_keys - fixed_keys
                )
                extra_keys = (
                    fixed_keys - expected_keys
                )

                raise ValueError(
                    "Fixed negatives do not match "
                    "evaluation positives: "
                    f"missing={len(missing_keys)}, "
                    f"extra={len(extra_keys)}"
                )

            fixed_drug_ids = {
                drug_id
                for pair in
                self.fixed_negative_map.values()
                for drug_id in pair
            }

            self.drug_ids = np.array(
                list(
                    set(self.drug_ids.tolist())
                    | fixed_drug_ids
                )
            )

        self.drug_ids = np.array([
            drug_id
            for drug_id in self.drug_ids
            if drug_id
            in MOL_EDGE_LIST_FEAT_MTX
        ])

        self.mol_cache = {
            drug_id:
                MOL_EDGE_LIST_FEAT_MTX[
                    drug_id
                ]
            for drug_id in self.drug_ids
        }

    def __len__(self):
        return len(self.tri_list)
    
    def __getitem__(self, index):
        return self.tri_list[index]

    def __corrupt_tail(
        self,
        h,
        r,
        n=1,
    ):
        """
        TWOSIDES clean TRAIN negative sampling.

        Tail corruption only:

            (h, t, r)
                ->
            (h, negative_tail, r)

        Candidate restrictions:
          1. current training drug universe only;
          2. negative_tail != h;
          3. unordered (h, negative_tail, r) must not be
             a current TRAIN positive.
        """

        h = str(h)
        r = int(r)

        forbidden = (
            ALL_TRUE_T_WITH_HR.get(
                (h, r),
                np.array(
                    [],
                    dtype=object,
                ),
            )
        )

        corrupted = []

        while len(corrupted) < n:

            need = (
                n
                - len(corrupted)
            )

            draw_size = max(
                need * 4,
                8,
            )

            candidates = np.random.choice(
                self.drug_ids,
                draw_size,
                replace=True,
            )

            # Reject self pairs.
            mask = (
                candidates
                != h
            )

            # Reject any same-relation TRAIN positive
            # under unordered pair semantics.
            if len(forbidden) > 0:

                mask &= np.isin(
                    candidates,
                    forbidden,
                    assume_unique=False,
                    invert=True,
                )

            valid = candidates[
                mask
            ]

            if len(valid) == 0:
                continue

            for candidate in valid:

                candidate = str(
                    candidate
                )

                if candidate in corrupted:
                    continue

                corrupted.append(
                    candidate
                )

                if len(corrupted) >= n:
                    break

        return np.asarray(
            corrupted[:n],
            dtype=object,
        )


    def __normal_batch(
        self,
        h,
        t,
        r,
        neg_size,
    ):
        """
        Preserve the common DrugDataset return interface while
        enforcing TWOSIDES tail-only corruption.
        """

        neg_heads = np.asarray(
            [],
            dtype=object,
        )

        neg_tails = (
            self.__corrupt_tail(
                h,
                r,
                neg_size,
            )
        )

        return (
            neg_heads,
            neg_tails,
        )


    def _create_enhanced_data_object(self, drug_id, edge_index, features):
        """
        Create a Data object with intra-molecular enrichment weights - optimized version
        """
        # Get number of edges efficiently
        num_edges = edge_index.size(1) if edge_index.dim() == 2 and edge_index.numel() > 0 else 0
        
        # Fast computation using precomputed values
        intra_enrichment_weights = compute_intra_molecular_enrichment_weights(drug_id, num_edges)
        
        # Create Data object with enrichment weights
        data = Data(x=features, edge_index=edge_index)
        # Always add the attribute, even if empty, to ensure consistent batching
        data.intra_enrichment_weights = intra_enrichment_weights
        
        return data

    def collate_fn(self, batch):
        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []
        pos_b_samples = []

        neg_rels = []
        neg_h_samples = []
        neg_t_samples = []
        neg_b_samples = []

        for h, t, r in batch:
            pos_rels.append(r)
            
            # Use cached mol data but create enhanced Data objects with intra-enrichment
            h_edge_index, h_features = self.mol_cache[h]
            t_edge_index, t_features = self.mol_cache[t]
            
            # Create enhanced Data objects with intra-molecular enrichment weights
            h_data = self._create_enhanced_data_object(h, h_edge_index, h_features)
            t_data = self._create_enhanced_data_object(t, t_edge_index, t_features)
            
            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)
            
            # Create bipartite graph with drug IDs for faster lookup
            pos_b_graph = self._create_b_graph(get_bipartite_graph(drug_to_mol_graph[h], drug_to_mol_graph[t]), 
                                             h_data.x, t_data.x, h, t)
            pos_b_samples.append(pos_b_graph)

            if self.fixed_negative_map is not None:

                key = (
                    str(h),
                    str(t),
                    int(r),
                )

                if key not in self.fixed_negative_map:
                    raise KeyError(
                        "No fixed negative found for "
                        f"{key}"
                    )

                neg_h, neg_t = (
                    self.fixed_negative_map[
                        key
                    ]
                )

                neg_rels.append(r)

                (
                    neg_h_edge_index,
                    neg_h_features,
                ) = self.mol_cache[neg_h]

                (
                    neg_t_edge_index,
                    neg_t_features,
                ) = self.mol_cache[neg_t]

                neg_h_data = (
                    self._create_enhanced_data_object(
                        neg_h,
                        neg_h_edge_index,
                        neg_h_features,
                    )
                )

                neg_t_data = (
                    self._create_enhanced_data_object(
                        neg_t,
                        neg_t_edge_index,
                        neg_t_features,
                    )
                )

                neg_h_samples.append(
                    neg_h_data
                )
                neg_t_samples.append(
                    neg_t_data
                )

                neg_b_graph = (
                    self._create_b_graph(
                        get_bipartite_graph(
                            drug_to_mol_graph[
                                neg_h
                            ],
                            drug_to_mol_graph[
                                neg_t
                            ],
                        ),
                        neg_h_data.x,
                        neg_t_data.x,
                        neg_h,
                        neg_t,
                    )
                )

                neg_b_samples.append(
                    neg_b_graph
                )

            else:
                # Original dynamic negative sampling.
                # This path is used by TRAIN only.
                neg_heads, neg_tails = (
                    self.__normal_batch(
                        h,
                        t,
                        r,
                        self.neg_ent,
                    )
                )

                for neg_h in neg_heads:

                    neg_rels.append(r)

                    (
                        neg_edge_index,
                        neg_features,
                    ) = self.mol_cache[
                        neg_h
                    ]

                    neg_h_data = (
                        self._create_enhanced_data_object(
                            neg_h,
                            neg_edge_index,
                            neg_features,
                        )
                    )

                    neg_h_samples.append(
                        neg_h_data
                    )

                    neg_t_samples.append(
                        t_data
                    )

                    neg_b_graph = (
                        self._create_b_graph(
                            get_bipartite_graph(
                                drug_to_mol_graph[
                                    neg_h
                                ],
                                drug_to_mol_graph[
                                    t
                                ],
                            ),
                            neg_h_data.x,
                            t_data.x,
                            neg_h,
                            t,
                        )
                    )

                    neg_b_samples.append(
                        neg_b_graph
                    )

                for neg_t in neg_tails:

                    neg_rels.append(r)

                    neg_h_samples.append(
                        h_data
                    )

                    (
                        neg_edge_index,
                        neg_features,
                    ) = self.mol_cache[
                        neg_t
                    ]

                    neg_t_data = (
                        self._create_enhanced_data_object(
                            neg_t,
                            neg_edge_index,
                            neg_features,
                        )
                    )

                    neg_t_samples.append(
                        neg_t_data
                    )

                    neg_b_graph = (
                        self._create_b_graph(
                            get_bipartite_graph(
                                drug_to_mol_graph[
                                    h
                                ],
                                drug_to_mol_graph[
                                    neg_t
                                ],
                            ),
                            h_data.x,
                            neg_t_data.x,
                            h,
                            neg_t,
                        )
                    )

                    neg_b_samples.append(
                        neg_b_graph
                    )

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_b_samples = Batch.from_data_list(pos_b_samples)
        pos_rels = torch.LongTensor(pos_rels).unsqueeze(0)
        
        neg_h_samples = Batch.from_data_list(neg_h_samples)
        neg_t_samples = Batch.from_data_list(neg_t_samples)
        neg_b_samples = Batch.from_data_list(neg_b_samples)
        neg_rels = torch.LongTensor(neg_rels).unsqueeze(0)
        
        return (pos_h_samples, pos_t_samples, pos_rels, pos_b_samples), \
               (neg_h_samples, neg_t_samples, neg_rels, neg_b_samples)

    def _create_b_graph(self, edge_index, x_s, x_t, drug_h_id, drug_t_id):
        """Optimized bipartite graph creation using precomputed values"""
        enrichment_weights = None
        
        # Use precomputed functional groups for faster lookup
        fg1 = DRUG_TO_FUNCTIONAL_GROUPS.get(drug_h_id, [])
        fg2 = DRUG_TO_FUNCTIONAL_GROUPS.get(drug_t_id, [])
        
        # Precompute the maximum enrichment score for this molecule pair
        max_enrichment = 1.0
        
        if fg1 and fg2:  # Only proceed if both molecules have functional groups
            for fg_1 in fg1:
                for fg_2 in fg2:
                    pair_key = (fg_1, fg_2)
                    if pair_key in FG_ENRICHMENT_SCORES:
                        max_enrichment = max(max_enrichment, FG_ENRICHMENT_SCORES[pair_key])
        
        # Apply the same enrichment value to all edges at once
        enrichment_weights = torch.ones(edge_index.shape[1], device=edge_index.device) * max_enrichment
    
        return BipartiteData(edge_index, x_s, x_t, enrichment_weights)

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
