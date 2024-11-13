import os
import pickle
import shutil

import lmdb
import numpy as np
from ase.atoms import Atoms
from dftio.io.gaussian.gaussian_tools import cut_matrix, transform_matrix, generate_molecule_transform_indices
from tqdm import tqdm
import pyscf
from learn_qh9.parse_gau_logs_tools import matrix_to_image


def get_overlap_matrix(ase_atoms, basis):
    mol = pyscf.gto.Mole()
    t = [[ase_atoms.numbers[atom_idx], an_atom.position]
         for atom_idx, an_atom in enumerate(ase_atoms)]
    mol.build(verbose=0, atom=t, basis=basis, unit='ang')
    overlap = mol.intor("int1e_ovlp")
    return overlap


pyscf_def2svp_convention = {
    'atom_to_dftio_orbitals': {'C': '3s2p1d', 'H': '2s1p', 'O': '3s2p1d', 'N': '3s2p1d', 'F': '3s2p1d'},
    'atom_to_transform_indices': {
        'C': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
        'H': [0, 1, 3, 4, 2],
        'O': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
        'N': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
        'F': [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13],
    },
    'basis_name': 'def2svp'
}


def get_idx_data(txn, idx: int):
    data_dict = pickle.loads(txn.get(int(idx).to_bytes(length=4, byteorder='big')))
    num_nodes = data_dict['num_nodes']
    atoms = np.frombuffer(data_dict['atoms'], np.int32)
    pos = np.frombuffer(data_dict['pos'], np.float64).reshape(num_nodes, 3)
    Ham = np.frombuffer(data_dict['Ham'], np.float64)
    num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
    Ham = Ham.reshape(num_orbitals, num_orbitals)
    return atoms, pos, Ham


def process_data(symbols, pos, ham, an_index):
    an_atoms = Atoms(symbols=symbols, positions=pos)
    molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(
        atom_types=an_atoms.symbols,
        atom_to_transform_indices=pyscf_def2svp_convention['atom_to_transform_indices']
    )
    # print(ham.shape)
    # matrix_to_image(matrix=ham, filename='ham.png')
    ham = transform_matrix(matrix=ham, transform_indices=molecule_transform_indices)
    ham = cut_matrix(full_matrix=ham, atom_in_mo_indices=atom_in_mo_indices)
    overlap = get_overlap_matrix(ase_atoms=an_atoms, basis='def2svp')
    # print(overlap.shape)
    # matrix_to_image(matrix=overlap, filename='overlap.png')
    overlap = transform_matrix(matrix=overlap, transform_indices=molecule_transform_indices)
    overlap = cut_matrix(full_matrix=overlap, atom_in_mo_indices=atom_in_mo_indices)
    # raise RuntimeError
    basis = pyscf_def2svp_convention['atom_to_dftio_orbitals']
    return {
        "atomic_numbers": an_atoms.numbers,
        "pbc": np.array([False, False, False]),
        "pos": an_atoms.positions.reshape(-1, 3).astype(np.float32),
        "cell": an_atoms.cell.reshape(3, 3).astype(np.float32),
        "hamiltonian": ham,
        "overlap": overlap,
        'idx': an_index,
        'basis': basis,
        'nf': 1
    }


def write_to_lmdb(lmdb_path, data_generator):
    for data_dict in data_generator:
        lmdb_env = lmdb.open(lmdb_path, map_size=1048576000000, lock=True)
        with lmdb_env.begin(write=True) as txn:
            entries = lmdb_env.stat()["entries"]
            txn.put(entries.to_bytes(length=4, byteorder='big'), pickle.dumps(data_dict))
        lmdb_env.close()


def qh9_lmdb_2_dptb_readable_size_ood(qh9_lmdb_path: str, dptb_lmdb_out_root: str, dump_length: int):
    if os.path.exists(dptb_lmdb_out_root):
        shutil.rmtree(dptb_lmdb_out_root)

    train_path = os.path.join(dptb_lmdb_out_root, 'train', f"data.{os.getpid()}.lmdb")
    valid_path = os.path.join(dptb_lmdb_out_root, 'valid', f"data.{os.getpid()}.lmdb")
    test_path = os.path.join(dptb_lmdb_out_root, 'test', f"data.{os.getpid()}.lmdb")

    for path in [train_path, valid_path, test_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    qh9_env = lmdb.open(qh9_lmdb_path, readonly=True, lock=False)
    with qh9_env.begin() as qh9_txn:
        def data_generator(size_filter):
            for an_index in tqdm(range(dump_length)):
                try:
                    symbols, pos, ham = get_idx_data(qh9_txn, an_index)
                except:
                    print('Index out of range.')
                    break
                if size_filter(len(symbols)):
                    yield process_data(symbols, pos, ham, an_index)

        write_to_lmdb(train_path, data_generator(lambda x: x <= 20))
        write_to_lmdb(valid_path, data_generator(lambda x: 21 <= x <= 22))
        write_to_lmdb(test_path, data_generator(lambda x: x >= 23))

    qh9_env.close()


qh9_lmdb_2_dptb_readable_size_ood(
    qh9_lmdb_path='/share/qh9_data/QH9Stable.lmdb',
    dptb_lmdb_out_root='/root/1113_qh9_dataset/qh9_size_ood_4_dptb_train',
    dump_length=150000
)
