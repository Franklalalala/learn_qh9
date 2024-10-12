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
from learn_qh9.datasets import matrix_transform
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
            error_dict[key] = np.mean(np.abs(cosine_similarity(outputs[key], target[key])))
        elif key == 'HOMO_orbital_coefficients':
            error_dict[key] = vec_cosine_similarity(outputs[key], target[key])
        else:
            diff = np.array(outputs[key] - target[key])
            mae = np.mean(np.abs(diff))
            error_dict[key] = mae
    # print(error_dict)
    return error_dict


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian, atom_symbols, transform_ham_flag=False, transform_overlap_flag=False):
    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)
    if transform_ham_flag:
        full_hamiltonian = matrix_transform(full_hamiltonian, atom_symbols, convention='back_2_thu_pyscf')
    if transform_overlap_flag:
        overlap_matrix = matrix_transform(overlap_matrix, atom_symbols, convention='back_2_thu_pyscf')

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


def load_gaussian_data(idx, gau_npy_folder_path):
    gau_path = os.path.join(gau_npy_folder_path, f'{idx}')
    gau_ham = np.load(os.path.join(gau_path, 'fock.npy'))
    # gau_ham = np.load(os.path.join(gau_path, 'original_ham.npy'))[0]
    gau_overlap = np.load(os.path.join(gau_path, 'overlap.npy'))
    return gau_ham, gau_overlap


def test_with_npy(abs_ase_path, npy_folder_path, gau_npy_folder_path, temp_data_file):
    total_error_dict = {'total_items': 0, 'dptb_label_vs_gau': {}, 'dptb_pred_vs_gau': {}}
    start_time = time.time()

    temp_data = []

    with connect(abs_ase_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            atom_nums = a_row.numbers
            an_atoms = a_row.toatoms()
            total_error_dict['total_items'] += 1

            # Change directory to npy_folder_path
            predicted_overlap = np.load(os.path.join(npy_folder_path, f'{idx}/predicted_overlap.npy'))
            original_overlap = np.load(os.path.join(npy_folder_path, f'{idx}/original_overlap.npy'))
            predicted_ham = np.load(os.path.join(npy_folder_path, f'{idx}/predicted_ham.npy'))
            original_ham = np.load(os.path.join(npy_folder_path, f'{idx}/original_ham.npy'))

            # Load Gaussian data
            gau_ham, gau_overlap = load_gaussian_data(idx, gau_npy_folder_path)

            mol = pyscf.gto.Mole()
            t = [[atom_nums[atom_idx], an_atom.position]
                 for atom_idx, an_atom in enumerate(an_atoms)]
            mol.build(verbose=0, atom=t, basis='6-311+g(d,p)', unit='ang')
            target_overlap = mol.intor("int1e_ovlp")
            homo_idx = int(sum(atom_nums) / 2) - 1

            predicted_orbital_energies, predicted_orbital_coefficients = cal_orbital_and_energies(atom_symbols=atom_nums,
                overlap_matrix=predicted_overlap, full_hamiltonian=predicted_ham, transform_ham_flag=True, transform_overlap_flag=True)
            original_orbital_energies, original_orbital_coefficients = cal_orbital_and_energies(atom_symbols=atom_nums,
                overlap_matrix=original_overlap, full_hamiltonian=original_ham, transform_ham_flag=True, transform_overlap_flag=True)
            gau_orbital_energies, gau_orbital_coefficients = cal_orbital_and_energies(atom_symbols=atom_nums,
                overlap_matrix=gau_overlap, full_hamiltonian=gau_ham)

            pred_HOMO, pred_LUMO = predicted_orbital_energies[homo_idx], predicted_orbital_energies[homo_idx + 1]
            tgt_HOMO, tgt_LUMO = original_orbital_energies[homo_idx], original_orbital_energies[homo_idx + 1]
            gau_HOMO, gau_LUMO = gau_orbital_energies[homo_idx], gau_orbital_energies[homo_idx + 1]

            outputs = {
                'HOMO': pred_HOMO, 'LUMO': pred_LUMO, 'GAP': pred_LUMO - pred_HOMO,
                'hamiltonian': predicted_ham, 'overlap': predicted_overlap,
                'orbital_coefficients': predicted_orbital_coefficients[:, :homo_idx + 1],
                'HOMO_orbital_coefficients': predicted_orbital_coefficients[:, homo_idx]
            }

            tgt_info = {
                'HOMO': tgt_HOMO, 'LUMO': tgt_LUMO, 'GAP': tgt_LUMO - tgt_HOMO,
                'hamiltonian': original_ham, 'overlap': original_overlap,
                'orbital_coefficients': original_orbital_coefficients[:, :homo_idx + 1],
                'HOMO_orbital_coefficients': original_orbital_coefficients[:, homo_idx]
            }

            gau_info = {
                'HOMO': gau_HOMO, 'LUMO': gau_LUMO, 'GAP': gau_LUMO - gau_HOMO,
                'hamiltonian': gau_ham, 'overlap': gau_overlap,
                'orbital_coefficients': gau_orbital_coefficients[:, :homo_idx + 1],
                'HOMO_orbital_coefficients': gau_orbital_coefficients[:, homo_idx]
            }

            # print(os.getcwd())
            tools.cubegen.orbital(mol, 'gau_HOMO.cube', gau_orbital_coefficients[:, homo_idx], nx=n_grid, ny=n_grid, nz=n_grid)

            error_dict = criterion(outputs, tgt_info, outputs.keys())
            dptb_label_vs_gau = criterion(tgt_info, gau_info, tgt_info.keys())
            dptb_pred_vs_gau = criterion(outputs, gau_info, outputs.keys())

            # Store temporary data for cube file generation
            temp_data.append({
                'mol': mol,
                'outputs': outputs,
                'tgt_info': tgt_info,
                'gau_info': gau_info,
                'idx': idx,
                'dptb_pred_vs_gau_HOMO_sim': dptb_pred_vs_gau['HOMO_orbital_coefficients']
            })

            for key in error_dict.keys():
                if key in total_error_dict.keys():
                    total_error_dict[key] += error_dict[key]
                else:
                    total_error_dict[key] = error_dict[key]

            for key in dptb_label_vs_gau.keys():
                if key in total_error_dict['dptb_label_vs_gau'].keys():
                    total_error_dict['dptb_label_vs_gau'][key] += dptb_label_vs_gau[key]
                else:
                    total_error_dict['dptb_label_vs_gau'][key] = dptb_label_vs_gau[key]

            for key in dptb_pred_vs_gau.keys():
                if key in total_error_dict['dptb_pred_vs_gau'].keys():
                    total_error_dict['dptb_pred_vs_gau'][key] += dptb_pred_vs_gau[key]
                else:
                    total_error_dict['dptb_pred_vs_gau'][key] = dptb_pred_vs_gau[key]

            # if idx == 0:
            #     continue
            # break

    for key in total_error_dict.keys():
        if key not in ['total_items', 'dptb_label_vs_gau', 'dptb_pred_vs_gau']:
            total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']

    for comparison in ['dptb_label_vs_gau', 'dptb_pred_vs_gau']:
        for key in total_error_dict[comparison].keys():
            total_error_dict[comparison][key] = total_error_dict[comparison][key] / total_error_dict['total_items']

    end_time = time.time()
    total_error_dict['second_per_item'] = (end_time - start_time) / total_error_dict['total_items']

    # Save all temporary data in a single npz file
    np.savez(temp_data_file, temp_data=temp_data)

    return total_error_dict


def generate_cube_files(temp_data_file, n_grid, cube_dump_place):
    """Generate cube files for HOMO orbitals and save them in sub-folders named by idx."""

    cwd_ = os.getcwd()
    # Load the saved temporary data
    data = np.load(temp_data_file, allow_pickle=True)
    temp_data = data['temp_data']

    for item in temp_data:
        mol = item['mol']
        outputs = item['outputs']
        tgt_info = item['tgt_info']
        gau_info = item['gau_info']
        idx = item['idx']
        dptb_pred_vs_gau_HOMO_sim = item['dptb_pred_vs_gau_HOMO_sim']

        # Create a sub-folder for each idx inside the cube_dump_place
        sub_folder = os.path.join(cube_dump_place, f'idx_{idx}_sim_{dptb_pred_vs_gau_HOMO_sim:.2g}')
        os.makedirs(sub_folder, exist_ok=True)
        os.chdir(sub_folder)

        tools.cubegen.orbital(mol, 'dptb_predicted_HOMO.cube', outputs['HOMO_orbital_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)
        tools.cubegen.orbital(mol, 'dptb_label_HOMO.cube', tgt_info['HOMO_orbital_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)
        tools.cubegen.orbital(mol, 'gau_HOMO.cube', gau_info['HOMO_orbital_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)
        diff_HOMO = gau_info['HOMO_orbital_coefficients'] - outputs['HOMO_orbital_coefficients']
        tools.cubegen.orbital(mol, 'gau_prediction_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)
        diff_HOMO = gau_info['HOMO_orbital_coefficients'] - tgt_info['HOMO_orbital_coefficients']
        tools.cubegen.orbital(mol, 'gau_label_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)
        os.chdir(cwd_)

if __name__ == '__main__':
    import os
    import json
    from pprint import pprint
    import shutil

    abs_ase_path = r'/root/local_test_ham_overlap/1010_test/gaussian_systems.db'
    npy_folder_path = r'/root/local_test_ham_overlap/1010_test/output'
    gau_npy_folder_path = r'/root/local_test_ham_overlap/1010_test/real_gau_mat_v4/test_matrices_dump_place'
    n_grid = 75

    # Create a temporary folder for storing intermediate results
    os.makedirs(npy_folder_path, exist_ok=True)
    temp_data_file = os.path.abspath('temp_data.npz')
    cube_dump_place = os.path.abspath('cubes')
    if os.path.exists(cube_dump_place):
        shutil.rmtree(cube_dump_place)
        os.remove(temp_data_file)
    os.makedirs(cube_dump_place)

    # Call the test_with_npy function (assuming this function is defined elsewhere)
    total_error_dict = test_with_npy(abs_ase_path=abs_ase_path, npy_folder_path=npy_folder_path,
                                     temp_data_file=temp_data_file, gau_npy_folder_path=gau_npy_folder_path)

    # Generate cube files after test_with_npy
    generate_cube_files(temp_data_file, n_grid, cube_dump_place)

    # Print results
    pprint(total_error_dict)
    with open('test_results.json', 'w') as f:
        json.dump(total_error_dict, f, indent=2)

    # Assuming process_batch_cube_file is defined elsewhere
    process_batch_cube_file(cube_dump_place)
