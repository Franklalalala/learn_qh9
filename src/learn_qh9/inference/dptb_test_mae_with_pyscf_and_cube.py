import json
import os
import os.path
import shutil
import time
from argparse import Namespace
from pprint import pprint

import numpy as np
import py3Dmol
import pyscf
from ase.db.core import connect
from learn_qh9.parse_gau_logs_tools import transform_matrix, generate_molecule_transform_indices
from learn_qh9.trainer import Trainer
from pyscf import tools, scf
from scipy import linalg
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


loss_weights = {
    'hamiltonian': 1.0,
    'diagonal_hamiltonian': 1.0,
    'non_diagonal_hamiltonian': 1.0,
    'orbital_energies': 1.0,
    "orbital_coefficients": 1.0,
    'HOMO': 1.0, 'LUMO': 1.0, 'GAP': 1.0,
}

back_to_pyscf_convention = {
    'atom_to_simplified_orbitals': {'C': 'sspspspspd', 'H': 'sssp', 'O': 'sspspspspd'},
    'atom_to_sorted_orbitals': {'C': 'sssssppppd', 'H': 'sssp', 'O': 'sssssppppd'},
    'atom_to_transform_indices': {'C': [0, 1, 2, 3, 4, 7, 5, 6, 10, 8, 9, 13, 11, 12, 16, 14, 15, 17, 18, 19, 20, 21],
                                  'H': [0, 1, 2, 5, 3, 4],
                                  'O': [0, 1, 2, 3, 4, 7, 5, 6, 10, 8, 9, 13, 11, 12, 16, 14, 15, 17, 18, 19, 20, 21]},
    'basis_name': '6-311+g(d,p)', }


def process_cube_file(cube_path):
    # Read the cube file
    with open(cube_path, 'r') as f:
        cube_data = f.read()

    # Create visualization
    view = py3Dmol.view()
    view.addModel(cube_data, 'cube')
    view.addVolumetricData(cube_data, "cube", {'isoval': -0.03, 'color': "red", 'opacity': 0.75})
    view.addVolumetricData(cube_data, "cube", {'isoval': 0.03, 'color': "blue", 'opacity': 0.75})
    view.setStyle({'stick': {}})
    view.zoomTo()

    # Generate HTML content
    html_content = view._make_html()

    # Create HTML file with the same name as the cube file
    html_path = os.path.splitext(cube_path)[0] + '.html'
    with open(html_path, "w") as out:
        out.write(html_content)

    print(f"Generated HTML for: {cube_path}")


def process_batch_cube_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.cube'):
                cube_path = os.path.join(dirpath, filename)
                process_cube_file(cube_path)


def vec_cosine_similarity(a, b):
    # Calculate the dot product and norms
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.abs(dot_product / (norm_a * norm_b))


def criterion(outputs, target, names):
    error_dict = {}
    for key in names:
        if key == 'orbital_coefficients':
        # if key.endswith('orbital_coefficients'):
            "The shape is [batch, total_orb, num_occ_orb]."
            # print(outputs[key])
            # print(target[key])
            error_dict[key] = np.mean(np.abs(cosine_similarity(outputs[key], target[key])))
        elif key == 'HOMO_orbital_coefficients':
            error_dict[key] = vec_cosine_similarity(outputs[key], target[key])
        else:
            diff = np.array(outputs[key] - target[key])
            mae = np.mean(np.abs(diff))
            error_dict[key] = mae
    print(error_dict)
    return error_dict


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)

    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    Fs = np.matmul(np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = np.linalg.eigh(Fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients
    return orbital_energies[0], orbital_coefficients[0]


def post_processing(batch, default_type=np.float32):
    for key in batch.keys():
        if isinstance(batch[key], np.ndarray) and np.issubdtype(batch[key].dtype, np.floating):
            batch[key] = batch[key].astype(default_type)
    return batch


def test_with_npy(abs_ase_path, npy_folder_path, n_grid):
    cwd_ = os.getcwd()
    total_error_dict = {'total_items': 0}
    start_time = time.time()
    with connect(abs_ase_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            atom_nums = a_row.numbers
            an_atoms = a_row.toatoms()
            total_error_dict['total_items'] += 1
            os.chdir(npy_folder_path)
            os.chdir(f'{str(idx)}')
            mol_transform_indices, _ = generate_molecule_transform_indices(atom_types=an_atoms.symbols,
                                                                        atom_to_transform_indices=back_to_pyscf_convention['atom_to_transform_indices'])
            predicted_overlap = np.load(r'predicted_overlap.npy')
            # print(mol_transform_indices)
            predicted_overlap = transform_matrix(predicted_overlap, mol_transform_indices)
            # print(predicted_overlap.shape)
            # break
            original_overlap = np.load(r'original_overlap.npy')
            original_overlap = transform_matrix(original_overlap, mol_transform_indices)
            predicted_ham = np.load(r'predicted_ham.npy')
            predicted_ham = transform_matrix(predicted_ham, mol_transform_indices)
            original_ham = np.load(r'original_ham.npy')
            original_ham = transform_matrix(original_ham, mol_transform_indices)

            predicted_orbital_energies, predicted_orbital_coefficients = cal_orbital_and_energies(
                overlap_matrix=predicted_overlap, full_hamiltonian=predicted_ham)
            original_orbital_energies, original_orbital_coefficients = cal_orbital_and_energies(
                overlap_matrix=original_overlap, full_hamiltonian=original_ham)

            mol = pyscf.gto.Mole()
            t = [[atom_nums[atom_idx], an_atom.position]
                 for atom_idx, an_atom in enumerate(an_atoms)]
            mol.build(verbose=0, atom=t, basis='6-311+g(d,p)', unit='ang')
            homo_idx = int(sum(atom_nums) / 2) - 1

            pred_HOMO = predicted_orbital_energies[homo_idx]
            tgt_HOMO = original_orbital_energies[homo_idx]
            pred_LUMO = predicted_orbital_energies[homo_idx + 1]
            tgt_LUMO = original_orbital_energies[homo_idx + 1]
            outputs = {}
            outputs['HOMO'], outputs['LUMO'], outputs['GAP'], outputs['hamiltonian'], outputs['overlap'], outputs['orbital_coefficients'], outputs['HOMO_orbital_coefficients'] = \
                pred_HOMO, pred_LUMO, pred_LUMO - pred_HOMO, predicted_ham, predicted_overlap, predicted_orbital_coefficients, predicted_orbital_coefficients[homo_idx]

            tgt_info = {}
            tgt_info['HOMO'], tgt_info['LUMO'], tgt_info['GAP'], tgt_info['hamiltonian'], tgt_info['overlap'], tgt_info['orbital_coefficients'], tgt_info['HOMO_orbital_coefficients'] = \
                tgt_HOMO, tgt_LUMO, tgt_LUMO - tgt_HOMO, original_ham, original_overlap, original_orbital_coefficients, original_orbital_coefficients[homo_idx]
            error_dict = criterion(outputs, tgt_info, outputs.keys())

            tools.cubegen.orbital(mol, 'predicted_HOMO.cube', outputs['HOMO_orbital_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'real_HOMO.cube', tgt_info['HOMO_orbital_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)


            diff = np.abs(tgt_info['HOMO_orbital_coefficients']) - np.abs(outputs['HOMO_orbital_coefficients'])
            tools.cubegen.orbital(mol, 'diff_HOMO.cube', diff, nx=n_grid, ny=n_grid, nz=n_grid)

            a_sim = error_dict['HOMO_orbital_coefficients']
            a_sim = np.round(a_sim, decimals=2)
            a_sim_str = f"{a_sim:.2f}"
            folder_name = f'idx_{idx}_sim_{a_sim_str}'
            os.chdir('./..')
            os.rename(src=f'{str(idx)}', dst=folder_name)
            os.chdir(cwd_)

            for key in error_dict.keys():
                if key in total_error_dict.keys():
                    total_error_dict[key] += error_dict[key]
                else:
                    total_error_dict[key] = error_dict[key]

    for key in total_error_dict.keys():
        if key != 'total_items':
            total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']
    end_time = time.time()
    total_error_dict['second_per_item'] = (end_time - start_time) / total_error_dict['total_items']

    return total_error_dict


if __name__ == '__main__':
    abs_ase_path = r'/root/local_test_ham_overlap/0923/dump.db'
    npy_folder_path = r'/root/local_test_ham_overlap/0923/output'
    total_error_dict = test_with_npy(abs_ase_path=abs_ase_path, npy_folder_path=npy_folder_path, n_grid=75)
    pprint(total_error_dict)
    with open('test_results.json', 'w') as f:
        json.dump(total_error_dict, f, indent=2)

    process_batch_cube_file(npy_folder_path)
