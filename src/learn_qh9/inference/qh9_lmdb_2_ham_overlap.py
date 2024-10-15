import pickle
import lmdb
import numpy as np
from ase.db.core import connect
from ase.atoms import Atoms

import os
from dftio.io.gaussian.gaussian_tools import cut_matrix, transform_matrix, generate_molecule_transform_indices
import shutil
from tqdm import tqdm
import pyscf
from pyscf import tools, scf


pyscf_def2svp_convention = {'atom_to_dftio_orbitals': {'C': '3s2p1d', 'H': '2s1p', 'O': '3s2p1d'},
                                  'atom_to_simplified_orbitals': {'C': 'sssppd', 'H': 'ssp', 'O': 'sssppd', 'N': 'sssppd', 'F': 'sssppd'},
                                  'atom_to_transform_indices': {'C': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
                                                                'H': [0, 1, 3, 4, 2],
                                                                'O': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
                                                                'N': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
                                                                'F': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
                                                                },
                                  'basis_name': 'def2svp'}


def get_idx_data(lmdb_path: str, idx: int):
    db_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with db_env.begin() as txn:
        data_dict = txn.get(int(idx).to_bytes(length=4, byteorder='big'))
        data_dict = pickle.loads(data_dict)
        _, num_nodes, atoms, pos, Ham = \
            data_dict['id'], data_dict['num_nodes'], \
                np.frombuffer(data_dict['atoms'], np.int32), \
                np.frombuffer(data_dict['pos'], np.float64), \
                np.frombuffer(data_dict['Ham'], np.float64)
        pos = pos.reshape(num_nodes, 3)
        num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
        Ham = Ham.reshape(num_orbitals, num_orbitals)
    db_env.close()
    return atoms,  pos, Ham


def get_overlap_from_atoms(atoms, basis='6-311+g(d,p)'):
    atom_nums = atoms.numbers
    mol = pyscf.gto.Mole()
    t = [[atom_nums[atom_idx], an_atom.position]
         for atom_idx, an_atom in enumerate(atoms)]
    mol.build(verbose=0, atom=t, basis=basis, unit='ang')
    target_overlap = mol.intor("int1e_ovlp")
    # homo_idx = int(sum(atom_nums) / 2) - 1
    return target_overlap


def get_ham_overlap_from_qh9_lmdb(qh9_lmdb_path: str, dump_folder_path: str, size_ood_flag: bool=True, dump_length: int=150000):
    cwd_ = os.getcwd()
    os.makedirs(dump_folder_path, exist_ok=True)
    for an_index in tqdm(range(dump_length)):
        try:
            symbols, pos, ham = get_idx_data(lmdb_path=qh9_lmdb_path, idx=an_index)
        except:
            print('Index out of range.')
            break
        if size_ood_flag:
            if len(symbols) < 23:
                continue
        an_atoms = Atoms(symbols=symbols, positions=pos)
        overlap = get_overlap_from_atoms(atoms=an_atoms)
        os.chdir(dump_folder_path)
        os.makedirs(f'{str(an_index)}')
        os.chdir(f'{str(an_index)}')
        np.save('ham.npy', ham)
        np.save('overlap.npy', overlap)
    os.chdir(cwd_)


get_ham_overlap_from_qh9_lmdb(
    qh9_lmdb_path=r'/share/qh9_data/QH9Stable.lmdb',
    dump_folder_path=r'/root/turn_qh9_into_trainable_size_ood/ham_and_overlap_npy',
    size_ood_flag=True,
    dump_length=150000
)


