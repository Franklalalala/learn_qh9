import pickle
import lmdb
import numpy as np
from ase.db.core import connect
from ase.atoms import Atoms

import os
from dftio.io.gaussian.gaussian_tools import cut_matrix, transform_matrix, generate_molecule_transform_indices
import shutil
from tqdm import tqdm


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


def qh9_lmdb_2_dptb_readable(qh9_lmdb_path: str, dptb_lmdb_out_root: str, dump_length: int):
    if os.path.exists(dptb_lmdb_out_root):
        shutil.rmtree(dptb_lmdb_out_root)
    lmdb_env = lmdb.open(dptb_lmdb_out_root, map_size=1048576000000, lock=True)
    with lmdb_env.begin(write=True) as txn:
        for an_index in tqdm(range(dump_length)):
            try:
                symbols, pos, ham = get_idx_data(lmdb_path=qh9_lmdb_path, idx=an_index)
            except:
                print('Index out of range.')
                break
            an_atoms = Atoms(symbols=symbols, positions=pos)
            molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(atom_types=an_atoms.symbols, atom_to_transform_indices=pyscf_def2svp_convention['atom_to_transform_indices'])
            ham = transform_matrix(matrix=ham, transform_indices=molecule_transform_indices)
            ham = cut_matrix(full_matrix=ham, atom_in_mo_indices=atom_in_mo_indices)
            basis = pyscf_def2svp_convention['atom_to_dftio_orbitals']
            data_dict = {
                "atomic_numbers": an_atoms.numbers,
                "pbc": np.array([False, False, False]),
                "pos": an_atoms.positions.reshape(-1, 3).astype(np.float32),
                "cell": an_atoms.cell.reshape(3, 3).astype(np.float32),
                "hamiltonian": ham,
                'idx': an_index,
                'basis': basis,
                'nf': 1
            }
            data_dict = pickle.dumps(data_dict)
            entries = lmdb_env.stat()["entries"]
            txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)
    lmdb_env.close()


def qh9_lmdb_2_dptb_readable_size_ood(qh9_lmdb_path: str, dptb_lmdb_out_root: str, dump_length: int):
    if os.path.exists(dptb_lmdb_out_root):
        shutil.rmtree(dptb_lmdb_out_root)

    train_dptb_lmdb_out_root = os.path.join(dptb_lmdb_out_root, 'train', "data.{}.lmdb".format(os.getpid()))
    os.makedirs(train_dptb_lmdb_out_root)
    valid_dptb_lmdb_out_root = os.path.join(dptb_lmdb_out_root, 'valid', "data.{}.lmdb".format(os.getpid()))
    os.makedirs(valid_dptb_lmdb_out_root)
    test_dptb_lmdb_out_root = os.path.join(dptb_lmdb_out_root, 'test', "data.{}.lmdb".format(os.getpid()))
    os.makedirs(test_dptb_lmdb_out_root)


    lmdb_env = lmdb.open(train_dptb_lmdb_out_root, map_size=1048576000000, lock=True)
    with lmdb_env.begin(write=True) as txn:
        for an_index in tqdm(range(dump_length)):
            try:
                symbols, pos, ham = get_idx_data(lmdb_path=qh9_lmdb_path, idx=an_index)
            except:
                print('Index out of range.')
                break
            an_atoms = Atoms(symbols=symbols, positions=pos)
            if len(an_atoms) > 20:
                continue

            molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(atom_types=an_atoms.symbols, atom_to_transform_indices=pyscf_def2svp_convention['atom_to_transform_indices'])
            ham = transform_matrix(matrix=ham, transform_indices=molecule_transform_indices)
            ham = cut_matrix(full_matrix=ham, atom_in_mo_indices=atom_in_mo_indices)
            basis = pyscf_def2svp_convention['atom_to_dftio_orbitals']
            data_dict = {
                "atomic_numbers": an_atoms.numbers,
                "pbc": np.array([False, False, False]),
                "pos": an_atoms.positions.reshape(-1, 3).astype(np.float32),
                "cell": an_atoms.cell.reshape(3, 3).astype(np.float32),
                "hamiltonian": ham,
                'idx': an_index,
                'basis': basis,
                'nf': 1
            }
            data_dict = pickle.dumps(data_dict)
            entries = lmdb_env.stat()["entries"]
            txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)
    lmdb_env.close()

    lmdb_env = lmdb.open(valid_dptb_lmdb_out_root, map_size=1048576000000, lock=True)
    with lmdb_env.begin(write=True) as txn:
        for an_index in tqdm(range(dump_length)):
            try:
                symbols, pos, ham = get_idx_data(lmdb_path=qh9_lmdb_path, idx=an_index)
            except:
                print('Index out of range.')
                break
            an_atoms = Atoms(symbols=symbols, positions=pos)
            if len(an_atoms) < 21 or len(an_atoms) > 22:
                continue
            molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(atom_types=an_atoms.symbols, atom_to_transform_indices=pyscf_def2svp_convention['atom_to_transform_indices'])
            ham = transform_matrix(matrix=ham, transform_indices=molecule_transform_indices)
            ham = cut_matrix(full_matrix=ham, atom_in_mo_indices=atom_in_mo_indices)
            basis = pyscf_def2svp_convention['atom_to_dftio_orbitals']
            data_dict = {
                "atomic_numbers": an_atoms.numbers,
                "pbc": np.array([False, False, False]),
                "pos": an_atoms.positions.reshape(-1, 3).astype(np.float32),
                "cell": an_atoms.cell.reshape(3, 3).astype(np.float32),
                "hamiltonian": ham,
                'idx': an_index,
                'basis': basis,
                'nf': 1
            }
            data_dict = pickle.dumps(data_dict)
            entries = lmdb_env.stat()["entries"]
            txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)
    lmdb_env.close()

    lmdb_env = lmdb.open(test_dptb_lmdb_out_root, map_size=1048576000000, lock=True)
    with lmdb_env.begin(write=True) as txn:
        for an_index in tqdm(range(dump_length)):
            try:
                symbols, pos, ham = get_idx_data(lmdb_path=qh9_lmdb_path, idx=an_index)
            except:
                print('Index out of range.')
                break
            an_atoms = Atoms(symbols=symbols, positions=pos)
            if len(an_atoms) < 23:
                continue
            molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(atom_types=an_atoms.symbols, atom_to_transform_indices=pyscf_def2svp_convention['atom_to_transform_indices'])
            ham = transform_matrix(matrix=ham, transform_indices=molecule_transform_indices)
            ham = cut_matrix(full_matrix=ham, atom_in_mo_indices=atom_in_mo_indices)
            basis = pyscf_def2svp_convention['atom_to_dftio_orbitals']
            data_dict = {
                "atomic_numbers": an_atoms.numbers,
                "pbc": np.array([False, False, False]),
                "pos": an_atoms.positions.reshape(-1, 3).astype(np.float32),
                "cell": an_atoms.cell.reshape(3, 3).astype(np.float32),
                "hamiltonian": ham,
                'idx': an_index,
                'basis': basis,
                'nf': 1
            }
            data_dict = pickle.dumps(data_dict)
            entries = lmdb_env.stat()["entries"]
            txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)
    lmdb_env.close()


qh9_lmdb_2_dptb_readable_size_ood(
    qh9_lmdb_path=r'/share/qh9_data/QH9Stable.lmdb',
    dptb_lmdb_out_root=r'/root/turn_qh9_into_trainable_size_ood/qh9_size_ood_4_dptb_train',
    dump_length=150000
)


