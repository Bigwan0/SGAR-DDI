from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
from torch_geometric.data import Data 

class FunctionalGroupDetector:
    def __init__(self):
        self.functional_groups = {
            # Core heterocyclic systems - highest priority
            'indole': '[nR2]1[cR2]2[cR1][cR1][cR1][cR1][cR2]2[cR1][cR1]1',
            'quinoline': '[nR2]1[cR2]2[cR1][cR1][cR1][cR1][cR1][cR2]2[cR1][cR1]1',
            'isoquinoline': '[nR2]1[cR2]2[cR1][cR1][cR1][cR1][cR1][cR2]2[cR1][cR1]1',
            'benzofuran': '[oR2]1[cR2]2[cR1][cR1][cR1][cR1][cR2]2[cR1][cR1]1',
            'benzothiophene': '[sR2]1[cR2]2[cR1][cR1][cR1][cR1][cR2]2[cR1][cR1]1',
            
            # Important nitrogen heterocycles
            'pyrrole': '[nH;R1]1[cR1][cR1][cR1][cR1]1',
            'pyridine': '[nR1]1[cR1][cR1][cR1][cR1][cR1]1',
            'pyrimidine': '[nR1]1[cR1][nR1][cR1][cR1][cR1]1',
            'imidazole': '[nR1]1[cR1][nR1][cR1][cR1]1',
            'pyrazole': '[nR1]1[nR1][cR1][cR1][cR1]1',
            'triazole': '[$([nR1]1[nR1][nR1][cR1][cR1]1),$([nR1]1[nR1][cR1][nR1][cR1]1)]',
            
            # Saturated heterocycles
            'piperidine': '[NX3R2]1[CX4R2][CX4R2][CX4R2][CX4R2][CX4R2]1',
            'piperazine': '[NX3R2]1[CX4R2][CX4R2][NX3R2][CX4R2][CX4R2]1',
            'morpholine': '[NX3R2]1[CX4R2][CX4R2][OX2R2][CX4R2][CX4R2]1',
            'pyrrolidine': '[NX3R2]1[CX4R2][CX4R2][CX4R2][CX4R2]1',
            
            # Carboxylic acid derivatives
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'ester': '[#6][CX3](=O)[OX2][#6]',
            'amide': '[NX3;H2,H1,H0;!$(NC=O)][CX3](=[OX1])[#6]',
            'acid_anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
            
            # Carbonyls
            'ketone': '[#6][CX3](=O)[#6;!$(C[OH])]',
            'aldehyde': '[CX3H1](=O)[#6]',
            
            # Alcohols and phenols
            'primary_alcohol': '[CH2][OH]',
            'secondary_alcohol': '[CH1]([#6])[OH]',
            'tertiary_alcohol': '[CH0]([#6])([#6])[OH]',
            'phenol': '[OH][c]1[c][c][c][c][c]1',
            
            # Amines - with improved specificity
            'primary_amine': '[NH2;!$(NC=O)][#6]',
            'secondary_amine': '[NH1;!$(NC=O)]([#6])[#6]',
            'tertiary_amine': '[NH0;!$(NC=O)]([#6])([#6])[#6]',
            'aniline': '[NH2][c]1[c][c][c][c][c]1',
            
            # Ethers - more specific definitions
            'ether': '[OX2]([#6;!$(C=O)])[#6;!$(C=O)]',
            'methoxy': '[CH3][OX2][#6;!$(C=O)]',
            'aryl_ether': '[OX2]([c])[#6;!$(C=O)]',
            
            # Aromatic systems
            'benzene': '[cR1]1[cR1][cR1][cR1][cR1][cR1]1',
            'naphthalene': '[cR2]1[cR2]2[cR1][cR1][cR1][cR1][cR2]2[cR1][cR1]1',
            
            # Other important groups
            'thiol': '[SH]',
            'sulfide': '[#16X2H0][#6]',
            'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
            'nitrile': '[NX1]#[CX2]',
            'phosphate': '[P](=O)([O][#6,H])([O][#6,H])([O][#6,H])',
            'sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])',
            
            # Specific drug-relevant groups
            'guanidine': '[NX3][CX3](=[NX2])[NX3]',
            'urea': '[NX3][CX3](=[OX1])[NX3]'
        }
        
        # Compile SMARTS patterns
        self.patterns = {
            name: Chem.MolFromSmarts(pattern)
            for name, pattern in self.functional_groups.items()
        }
        
    def detect_functional_groups(self, mol):
        """
        Detect functional groups in a molecule with improved overlap handling
        """
        if mol is None:
            return []
            
        # Track atoms that have been assigned to complex ring systems
        assigned_atoms = set()
        matches = []
        
        # First pass: Detect complex ring systems
        ring_patterns = ['indole', 'quinoline', 'isoquinoline', 'benzofuran', 
                        'benzothiophene', 'naphthalene']
                        
        for fg_name in ring_patterns:
            pattern = self.patterns[fg_name]
            for match in mol.GetSubstructMatches(pattern):
                matches.append((fg_name, match))
                assigned_atoms.update(match)
        
        # Second pass: Detect other groups, considering already assigned atoms
        for fg_name, pattern in self.patterns.items():
            if fg_name not in ring_patterns:
                for match in mol.GetSubstructMatches(pattern):
                    # Check overlap with assigned atoms
                    if len(set(match) & assigned_atoms) < len(match) * 0.5:
                        matches.append((fg_name, match))
                        assigned_atoms.update(match)
        
        return matches
