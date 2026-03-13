import os
import os.path as osp
import pickle
import logging
from argparse import Namespace

import lmdb
import numpy as np
import torch
from learn_qh9.tools import get_lmdb_size, get_readable_info_from_lmdb
from torch_geometric.data import InMemoryDataset, Data

# Use standard logger retrieval
logger = logging.getLogger(__name__)

BOHR2ANG = 1.8897259886

convention_dict = {
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={
            1: 'ssp',
            3: 'ssspp',
            5: 'sssppd',
            6: 'sssppd',
            7: 'sssppd',
            8: 'sssppd',
            9: 'sssppd',
            15: 'sssspppd',
            16: 'sssspppd',
            17: 'sssspppd'
        },
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            3: [0, 1, 2, 3, 4],
            5: [0, 1, 2, 3, 4, 5],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
            15: [0, 1, 2, 3, 4, 5, 6, 7],
            16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7]
        },
    ),
    'back2pyscf': Namespace(
        atom_to_orbitals_map={
            1: 'ssp',
            3: 'ssspp',
            5: 'sssppd',
            6: 'sssppd',
            7: 'sssppd',
            8: 'sssppd',
            9: 'sssppd',
            15: 'sssspppd',
            16: 'sssspppd',
            17: 'sssspppd'
        },
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            3: [0, 1, 2, 3, 4],
            5: [0, 1, 2, 3, 4, 5],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
            15: [0, 1, 2, 3, 4, 5, 6, 7],
            16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7]
        }
    ),

    'thu_cluster': Namespace(
        atom_to_orbitals_map={
            1: 'ssp', 3: 'ssspp', 5: 'sssppd', 6: 'sssppd', 7: 'sssppd',
            8: 'sssppd', 9: 'sssppd', 15: 'sssspppd', 16: 'sssspppd', 17: 'sssspppd'
        },
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 3: [0, 1, 2, 3, 4], 5: [0, 1, 2, 3, 4, 5],
            6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5], 8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5], 15: [0, 1, 2, 3, 4, 5, 6, 7], 16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7]
        },
    ),

    'back_thu_cluster': Namespace(
        atom_to_orbitals_map={
            1: 'ssp', 3: 'ssspp', 5: 'sssppd', 6: 'sssppd', 7: 'sssppd',
            8: 'sssppd', 9: 'sssppd', 15: 'sssspppd', 16: 'sssspppd', 17: 'sssspppd'
        },
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 3: [0, 1, 2, 3, 4], 5: [0, 1, 2, 3, 4, 5],
            6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5], 8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5], 15: [0, 1, 2, 3, 4, 5, 6, 7], 16: [0, 1, 2, 3, 4, 5, 6, 7],
            17: [0, 1, 2, 3, 4, 5, 6, 7]
        },
    ),
    'dptb2qhnet': Namespace(
        atom_to_orbitals_map={1: 'ssp', 3: 'ssspp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [0, 1, 2], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 3: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),


    'gau_def2svp_2_pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 3: 'ssspp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [0, 1, 2], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 3: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'pyscf_6311_plus_gdp': Namespace(
        atom_to_orbitals_map={1: 'sssp', 3: 'sssssppppd', 6: 'sssssppppd', 7: 'sssssppppd', 8: 'sssssppppd',
                              9: 'sssssppppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2, 3], 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
    ),

    'back_2_thu_pyscf': Namespace(
        atom_to_orbitals_map={1: 'sssp', 3: 'sssssppppd', 6: 'sssssppppd', 7: 'sssssppppd', 8: 'sssssppppd',
                              9: 'sssssppppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2, 3], 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
    ),



}

_ORB_DIM = {'s': 1, 'p': 3, 'd': 5}


def infer_nbasis_from_atoms(atoms: np.ndarray, convention: str) -> int:
    conv = convention_dict[convention]
    total = 0
    for a in atoms:
        z = int(a)
        if z not in conv.atom_to_orbitals_map:
            raise KeyError(f"Atom Z={z} not in convention '{convention}' atom_to_orbitals_map")
        for orb in conv.atom_to_orbitals_map[z]:
            total += _ORB_DIM[orb]
    return total


def matrix_transform(matrices, atoms, convention='pyscf_631G'):
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        a = int(a)
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    matrices_new = matrices[..., transform_indices, :]
    matrices_new = matrices_new[..., :, transform_indices]
    matrices_new = matrices_new * transform_signs[:, None]
    matrices_new = matrices_new * transform_signs[None, :]
    return matrices_new


class CustomizedQH9Stable(InMemoryDataset):
    def __init__(self, src_lmdb_folder_path: str = None, db_workbase='datasets/', split='random',
                 transform=None, pre_transform=None, pre_filter=None, convention='pyscf_def2svp',
                 is_debug=False, split_flag=None, target='density_matrix', aux_keys_list=None):
        db_workbase = os.path.abspath(db_workbase)
        self.root = db_workbase
        if split == 'pre_splitted':
            assert split_flag is not None, "Must provide split_flag for 'pre_splitted' split"
            self.split_flag = split_flag
            self.root = osp.join(db_workbase, split_flag)

        self.src_lmdb_folder_path = src_lmdb_folder_path
        self.is_debug = is_debug
        self.split = split
        self.orbital_mask = {}
        self.target = target

        # Strictly use the provided list. If None, empty list (no aux data extracted).
        self.aux_keys_list = aux_keys_list if aux_keys_list is not None else []

        if convention == 'pyscf_6311_plus_gdp':
            self.full_orbitals = 22
            orbital_mask_line1 = torch.tensor([0, 1, 2, 5, 6, 7])
            orbital_mask_line2 = torch.arange(self.full_orbitals)
            for i in range(1, 11):
                self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        elif convention == 'thu_cluster':
            self.full_orbitals = 18
            orbital_mask_h = torch.tensor([0, 1, 4, 5, 6], dtype=torch.long)
            orbital_mask_li = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9], dtype=torch.long)
            orbital_mask_3s2p1d = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17], dtype=torch.long)
            orbital_mask_full = torch.arange(self.full_orbitals, dtype=torch.long)

            self.orbital_mask[1] = orbital_mask_h
            self.orbital_mask[3] = orbital_mask_li
            for z in [5, 6, 7, 8, 9]:
                self.orbital_mask[z] = orbital_mask_3s2p1d
            for z in [15, 16, 17]:
                self.orbital_mask[z] = orbital_mask_full

        else:
            self.full_orbitals = 14
            orbital_mask_line1 = torch.tensor([0, 1, 3, 4, 5])
            orbital_mask_line_li = torch.arange(9)
            orbital_mask_line2 = torch.arange(self.full_orbitals)
            for i in range(1, 11):
                if i == 1:
                    self.orbital_mask[i] = orbital_mask_line1
                elif i == 3:
                    self.orbital_mask[i] = orbital_mask_line_li
                else:
                    self.orbital_mask[i] = orbital_mask_line2

        self.convention = convention

        super(CustomizedQH9Stable, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.train_mask, self.val_mask, self.test_mask = torch.load(self.processed_paths[0], weights_only=False)
        self.slices = {
            'id': torch.arange(self.train_mask.shape[0] + self.val_mask.shape[0] + self.test_mask.shape[0] + 1)}

    @property
    def processed_file_names(self):
        if self.split == 'random':
            return ['processed_QH9Stable_random.pt', 'QH9Stable.lmdb/data.mdb']
        elif self.split == 'size_ood':
            return ['processed_QH9Stable_size_ood.pt', 'QH9Stable.lmdb/data.mdb']
        elif self.split == 'pre_splitted':
            return ['processed_QH9Stable_pre_splitted.pt', 'QH9Stable.lmdb/data.mdb']

    def process(self):
        new_db_folder_path = os.path.join(self.processed_dir, 'QH9Stable.lmdb')
        if self.split == 'pre_splitted':
            self.sub_src_lmdb_folder_path = os.path.join(self.src_lmdb_folder_path, self.split_flag)
        else:
            self.sub_src_lmdb_folder_path = self.src_lmdb_folder_path

        if self.is_debug:
            import shutil
            shutil.copytree(src=self.sub_src_lmdb_folder_path, dst=new_db_folder_path)
        else:
            os.symlink(src=self.sub_src_lmdb_folder_path, dst=new_db_folder_path)

        if self.split == 'random':
            print('Random splitting...')
            data_ratio = [0.8, 0.1, 0.1]
            lmdb_size = get_lmdb_size(new_db_folder_path)
            data_split = [int(lmdb_size * data_ratio[0]), int(lmdb_size * data_ratio[1])]
            data_split.append(lmdb_size - sum(data_split))
            indices = np.random.RandomState(seed=43).permutation(lmdb_size)
            train_mask = indices[:data_split[0]]
            val_mask = indices[data_split[0]:data_split[0] + data_split[1]]
            test_mask = indices[data_split[0] + data_split[1]:]
            print(f'Number of train/valid/test is {len(train_mask)}/{len(val_mask)}/{len(test_mask)}')

        elif self.split == 'size_ood':
            print('Size OOD splitting...')
            num_nodes_list = []
            for idx, info in get_readable_info_from_lmdb(new_db_folder_path):
                a_num_nodes = info['num_nodes']
                num_nodes_list.append(a_num_nodes)
            num_nodes_array = np.array(num_nodes_list)
            train_indices = np.where(num_nodes_array <= 20)
            val_condition = np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22)
            val_indices = np.where(val_condition)
            test_indices = np.where(num_nodes_array >= 23)
            train_mask = train_indices[0].astype(np.int64)
            val_mask = val_indices[0].astype(np.int64)
            test_mask = test_indices[0].astype(np.int64)
            print(f'Number of train/valid/test is {len(train_mask)}/{len(val_mask)}/{len(test_mask)}')

        elif self.split == 'pre_splitted':
            print(f'Loading {self.split_flag} datasets...')

            train_lmdb_size = get_lmdb_size(os.path.join(self.src_lmdb_folder_path, 'train'))
            train_mask = np.arange(train_lmdb_size)
            valid_lmdb_size = get_lmdb_size(os.path.join(self.src_lmdb_folder_path, 'valid'))
            val_mask = np.arange(valid_lmdb_size)
            test_lmdb_size = get_lmdb_size(os.path.join(self.src_lmdb_folder_path, 'test'))
            test_mask = np.arange(test_lmdb_size)

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])
        self.train_mask, self.val_mask, self.test_mask = torch.load(self.processed_paths[0], weights_only=False)

    def cut_matrix(self, matrix, atoms):
        all_diagonal_matrix_blocks = []
        all_non_diagonal_matrix_blocks = []
        all_diagonal_matrix_block_masks = []
        all_non_diagonal_matrix_block_masks = []
        col_idx = 0
        edge_index_full = []
        for idx_i, atom_i in enumerate(atoms):  # (src)
            row_idx = 0
            atom_i = atom_i.item()
            mask_i = self.orbital_mask[atom_i]
            for idx_j, atom_j in enumerate(atoms):  # (dst)
                if idx_i != idx_j:
                    edge_index_full.append([idx_j, idx_i])
                atom_j = atom_j.item()
                mask_j = self.orbital_mask[atom_j]
                matrix_block = torch.zeros(self.full_orbitals, self.full_orbitals).type(torch.float64)
                matrix_block_mask = torch.zeros(self.full_orbitals, self.full_orbitals).type(torch.float64)
                extracted_matrix = \
                    matrix[row_idx: row_idx + len(mask_j), col_idx: col_idx + len(mask_i)]

                # for matrix_block
                tmp = matrix_block[mask_j]
                tmp[:, mask_i] = extracted_matrix
                matrix_block[mask_j] = tmp

                tmp = matrix_block_mask[mask_j]
                tmp[:, mask_i] = 1
                matrix_block_mask[mask_j] = tmp

                if idx_i == idx_j:
                    all_diagonal_matrix_blocks.append(matrix_block)
                    all_diagonal_matrix_block_masks.append(matrix_block_mask)
                else:
                    all_non_diagonal_matrix_blocks.append(matrix_block)
                    all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
                row_idx = row_idx + len(mask_j)
            col_idx = col_idx + len(mask_i)
        return torch.stack(all_diagonal_matrix_blocks, dim=0), \
            torch.stack(all_non_diagonal_matrix_blocks, dim=0), \
            torch.stack(all_diagonal_matrix_block_masks, dim=0), \
            torch.stack(all_non_diagonal_matrix_block_masks, dim=0), \
            torch.tensor(edge_index_full).transpose(-1, -2)

    def get_mol(self, atoms, pos, Ham):
        hamiltonian = torch.tensor(
            matrix_transform(Ham, atoms, convention=self.convention), dtype=torch.float64)
        diagonal_hamiltonian, non_diagonal_hamiltonian, \
            diagonal_hamiltonian_mask, non_diagonal_hamiltonian_mask, edge_index_full \
            = self.cut_matrix(hamiltonian, atoms)

        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
            edge_index_full=edge_index_full
        )
        return data

    def get(self, idx):
        db_env = lmdb.open(os.path.join(self.processed_dir, 'QH9Stable.lmdb'), readonly=True, lock=False)
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder='big'))
            data_dict = pickle.loads(data_dict)

            # --- 1. Extract Core Data ---
            num_nodes = data_dict.get('num_nodes')
            atoms = np.frombuffer(data_dict['atoms'], np.int32)
            pos = np.frombuffer(data_dict['pos'], np.float64).reshape(num_nodes, 3)
            Ham = np.frombuffer(data_dict[self.target], np.float64)

            if 'nbasis' in data_dict:
                num_orbitals = data_dict['nbasis']
            else:
                num_orbitals = infer_nbasis_from_atoms(atoms, self.convention)
            Ham = Ham.reshape(num_orbitals, num_orbitals)

            # Create base Data object
            data = self.get_mol(atoms, pos, Ham)

            # --- 2. Add ID ---
            raw_id = data_dict.get('id', idx)
            if isinstance(raw_id, (int, np.integer)):
                data.original_id = torch.tensor([int(raw_id)], dtype=torch.long)
            else:
                data.original_id = raw_id

            # --- 3. Extract Specific Aux Keys Only ---
            for key in self.aux_keys_list:
                if key in data_dict:
                    val = data_dict[key]

                    # Heuristic Type Conversion for Aux Data
                    if isinstance(val, (int, float, np.number)):
                        setattr(data, key, torch.tensor([val]))
                    elif isinstance(val, np.ndarray):
                        try:
                            setattr(data, key, torch.from_numpy(val))
                        except:
                            setattr(data, key, val)
                    elif isinstance(val, list):
                        try:
                            setattr(data, key, torch.tensor(val))
                        except:
                            setattr(data, key, val)
                    else:
                        setattr(data, key, val)

        db_env.close()
        return data