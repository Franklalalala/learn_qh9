import os
import os.path as osp
import pickle
import shutil
from argparse import Namespace

import lmdb
import numpy as np
import torch
from learn_qh9.tools import get_lmdb_size, get_readable_info_from_lmdb
from torch_geometric.data import InMemoryDataset, Data

BOHR2ANG = 1.8897259886

convention_dict = {
    'pyscf_631G': Namespace(
        atom_to_orbitals_map={1: 'ss', 6: 'ssspp', 7: 'ssspp', 8: 'ssspp', 9: 'ssspp'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd':
            [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1], 6: [0, 1, 2, 3, 4], 7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4], 9: [0, 1, 2, 3, 4]
        },
    ),
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'pyscf_6311_plus_gdp': Namespace(
        atom_to_orbitals_map={1: 'sssp', 6: 'sssssppppd', 7: 'sssssppppd', 8: 'sssssppppd', 9: 'sssssppppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2, 3], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
    ),
    'back2pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        }
    ),
    'back2pyscf_v2': Namespace(
        atom_to_orbitals_map={1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        }
    ),
    'back_2_thu_pyscf': Namespace(
        atom_to_orbitals_map={1: 'sssp', 6: 'sssssppppd', 7: 'sssssppppd', 8: 'sssssppppd', 9: 'sssssppppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2, 3], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
    ),
}

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs_tensor = torch.zeros(5, 19)
atomrefs_tensor[:, 7] = torch.tensor(atomrefs[7])
atomrefs_tensor[:, 8] = torch.tensor(atomrefs[8])
atomrefs_tensor[:, 9] = torch.tensor(atomrefs[9])
atomrefs_tensor[:, 10] = torch.tensor(atomrefs[10])


def matrix_transform(matrices, atoms, convention='pyscf_631G'):
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
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
                 is_debug=False, split_flag=None):
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

        if convention != 'pyscf_6311_plus_gdp':
            self.full_orbitals = 14
            # orbital_mask_line1 = torch.tensor([0, 1, 2, 3, 4])
            orbital_mask_line1 = torch.tensor([0, 1, 3, 4, 5])
            orbital_mask_line2 = torch.arange(self.full_orbitals)
            for i in range(1, 11):
                self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2
        else:
            self.full_orbitals = 22
            orbital_mask_line1 = torch.tensor([0, 1, 2, 5, 6, 7])
            orbital_mask_line2 = torch.arange(self.full_orbitals)
            for i in range(1, 11):
                self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2
        self.convention = convention

        super(CustomizedQH9Stable, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.slices = {'id': torch.arange(self.train_mask.shape[0] + self.val_mask.shape[0] + self.test_mask.shape[0] + 1)}

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
        self.train_mask, self.val_mask, self.test_mask = torch.load(self.processed_paths[0])

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
            _, num_nodes, atoms, pos, Ham = \
                data_dict['id'], data_dict['num_nodes'], \
                    np.frombuffer(data_dict['atoms'], np.int32), \
                    np.frombuffer(data_dict['pos'], np.float64), \
                    np.frombuffer(data_dict['Ham'], np.float64)
            pos = pos.reshape(num_nodes, 3)
            if 'nbasis' in data_dict.keys():
                num_orbitals = data_dict['nbasis']
            else:
                num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            data = self.get_mol(atoms, pos, Ham)
        db_env.close()
        return data
