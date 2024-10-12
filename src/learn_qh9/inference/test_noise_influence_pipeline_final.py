import os
import re
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from ase.db import connect
from dftio.io.gaussian.gaussian_tools import read_fock_from_gau_log, read_int1e_from_gau_log, get_basic_info
from dftio.io.gaussian.gaussian_conventionns import gau_6311_plus_gdp_convention, gau_6311_plus_gdp_to_pyscf_convention
from learn_qh9.parse_gau_logs_tools import matrix_to_image, transform_matrix, generate_molecule_transform_indices
from tqdm import tqdm
import shutil
import pickle
from ase.units import Hartree


def get_gau_logs(valid_file_path: str):
    raw_datas = []
    with open(valid_file_path, 'r') as file:
        for line in file.readlines():
            raw_datas.append(line.strip())
    return raw_datas


def cal_orbital_and_energies_np(overlap_matrix, full_hamiltonian):
    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)

    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    Fs = np.matmul(np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = np.linalg.eigh(Fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients
    return orbital_energies, orbital_coefficients


def process_gaussian_logs(log_files, db_path, dump_folder):
    if os.path.exists(dump_folder):
        shutil.rmtree(dump_folder)
    if os.path.exists(db_path):
        os.remove(db_path)

    os.makedirs(dump_folder)

    with connect(db_path) as db:
        for idx, log_path in tqdm(enumerate(log_files), desc="Processing Gaussian logs"):
            ep_number = re.search(r'ep-(\d+)', log_path).group(1)
            save_dir = os.path.join(dump_folder, f'{idx}')
            os.makedirs(save_dir, exist_ok=True)

            nbasis, atoms = get_basic_info(file_path=log_path)
            molecule_transform_indices, _ = generate_molecule_transform_indices(atom_types=atoms.symbols, atom_to_transform_indices=gau_6311_plus_gdp_to_pyscf_convention['atom_to_transform_indices'])
            fock_matrix = read_fock_from_gau_log(logname=log_path, nbf=nbasis)
            overlap_matrix = read_int1e_from_gau_log(logname=log_path, nbf=nbasis, matrix_type=0)
            fock_matrix = transform_matrix(fock_matrix, molecule_transform_indices)
            overlap_matrix = transform_matrix(overlap_matrix, molecule_transform_indices)
            np.save(os.path.join(save_dir, 'fock.npy'), fock_matrix)
            np.save(os.path.join(save_dir, 'overlap.npy'), overlap_matrix)
            real_data = {
                'log_path': log_path,
                'matrix_path': save_dir,
                'num_electrons': sum(atoms.numbers),
                'ep_number': int(ep_number),
                'index_number': idx
            }
            db.write(atoms, key_value_pairs=real_data)


def test_noise_influence(fock, overlap, num_electrons, noise_lower, noise_upper, grid):
    def add_noise(matrix, noise_level):
        noise = np.random.laplace(loc=0, scale=noise_level, size=matrix.shape)
        return matrix + noise

    def compare_results(true_energies, true_coeffs, noisy_energies, noisy_coeffs, num_occ):
        energy_mae = np.mean(np.abs(true_energies[:num_occ] - noisy_energies[:num_occ]))
        coeff_similarity = np.mean([np.abs(pearsonr(true_col, noisy_col)[0])
                                    for true_col, noisy_col in zip(true_coeffs.T, noisy_coeffs.T)])

        homo_energy_diff = np.abs(true_energies[num_occ - 1] - noisy_energies[num_occ - 1])
        homo_coeff_similarity = np.abs(pearsonr(true_coeffs[:, num_occ - 1], noisy_coeffs[:, num_occ - 1])[0])

        lumo_energy_diff = np.abs(true_energies[num_occ] - noisy_energies[num_occ])
        gap_energy_diff = np.abs((true_energies[num_occ] - true_energies[num_occ - 1]) -
                                 (noisy_energies[num_occ] - noisy_energies[num_occ - 1]))

        return energy_mae, coeff_similarity, homo_energy_diff, homo_coeff_similarity, lumo_energy_diff, gap_energy_diff

    noise_levels = np.logspace(np.log10(noise_lower), np.log10(noise_upper), grid)
    results = []

    true_energies, true_coeffs = cal_orbital_and_energies_np(overlap, fock)
    true_energies = true_energies[0]
    true_coeffs = true_coeffs[0]
    num_occ = num_electrons // 2

    for noise in noise_levels:
        noisy_overlap = add_noise(overlap, noise)
        noisy_fock = add_noise(fock, noise)

        noisy_energies, noisy_coeffs = cal_orbital_and_energies_np(noisy_overlap, noisy_fock)
        noisy_energies = noisy_energies[0]
        noisy_coeffs = noisy_coeffs[0]

        energy_mae, coeff_similarity, homo_energy_diff, homo_coeff_similarity, lumo_energy_diff, gap_energy_diff = \
            compare_results(true_energies, true_coeffs, noisy_energies, noisy_coeffs, num_occ)

        results.append({
            'noise_level': noise,
            'energy_mae': energy_mae,
            'coeff_similarity': coeff_similarity,
            'homo_energy_diff': homo_energy_diff,
            'homo_coeff_similarity': homo_coeff_similarity,
            'lumo_energy_diff': lumo_energy_diff,
            'gap_energy_diff': gap_energy_diff
        })

    return results


def analyze_all_systems(db_path, noise_lower, noise_upper, grid):
    results_file = 'analysis_results.pkl'

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            return pickle.load(f)

    all_results = []

    with connect(db_path) as db:
        for row in tqdm(db.select(), desc="Processing systems"):
            fock = np.load(os.path.join(row.matrix_path, 'fock.npy'))
            overlap = np.load(os.path.join(row.matrix_path, 'overlap.npy'))
            num_electrons = row.num_electrons
            results = test_noise_influence(fock, overlap, num_electrons, noise_lower, noise_upper, grid)
            all_results.append(results)

    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)

    return all_results


def calculate_mean_results(all_results):
    grid = len(all_results[0])
    mean_results = []
    for i in range(grid):
        noise_level = all_results[0][i]['noise_level']
        mean_energy_mae = np.mean([results[i]['energy_mae'] for results in all_results])
        mean_coeff_similarity = np.mean([results[i]['coeff_similarity'] for results in all_results])
        mean_homo_energy_diff = np.mean([results[i]['homo_energy_diff'] for results in all_results])
        mean_homo_coeff_similarity = np.mean([results[i]['homo_coeff_similarity'] for results in all_results])
        mean_lumo_energy_diff = np.mean([results[i]['lumo_energy_diff'] for results in all_results])
        mean_gap_energy_diff = np.mean([results[i]['gap_energy_diff'] for results in all_results])

        mean_results.append({
            'noise_level': noise_level,
            'energy_mae': mean_energy_mae,
            'coeff_similarity': mean_coeff_similarity,
            'homo_energy_diff': mean_homo_energy_diff,
            'homo_coeff_similarity': mean_homo_coeff_similarity,
            'lumo_energy_diff': mean_lumo_energy_diff,
            'gap_energy_diff': mean_gap_energy_diff
        })
    return mean_results


def plot_coeff_similarity(noise_levels, coeff_similarities, homo_coeff_similarities, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
    noise_80_list = []
    for ax, similarities, title in zip([ax1, ax2],
                                       [coeff_similarities, homo_coeff_similarities],
                                       ['Mean Coefficient Similarity', 'Mean HOMO Coefficient Similarity']):
        ax.semilogx(noise_levels, similarities)
        ax.set_xlabel('Noise level')
        ax.set_ylabel('Coefficient similarity')
        ax.set_title(f'{title} vs Noise level')
        ax.grid(True)

        diff_80 = np.abs(np.array(similarities) - 0.8)
        idx_80_closest = np.argmin(diff_80)
        idx_80_second_closest = np.argsort(diff_80)[1]

        if idx_80_closest < idx_80_second_closest:
            idx_80_lower, idx_80_upper = idx_80_closest, idx_80_second_closest
        else:
            idx_80_lower, idx_80_upper = idx_80_second_closest, idx_80_closest

        noise_80 = np.sqrt(noise_levels[idx_80_lower] * noise_levels[idx_80_upper])
        similarity_80 = np.interp(noise_80, noise_levels, similarities)

        ax.plot(noise_80, similarity_80, 'kx', markersize=10, label='80% similarity')
        ax.axvline(x=noise_80, color='r', linestyle='--')
        ax.axhline(y=similarity_80, color='r', linestyle='--')
        ax.legend()
        ax.text(0.05, 0.05, f'Noise level at 80% similarity: {noise_80:.2e}',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        noise_80_list.append(noise_80)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return noise_80_list[0], noise_80_list[1]

def plot_energy_differences(noise_levels, energy_maes, homo_energy_diffs, lumo_energy_diffs, gap_energy_diffs,
                            output_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

    axes = [ax1, ax2, ax3, ax4]
    data = [energy_maes, homo_energy_diffs, lumo_energy_diffs, gap_energy_diffs]
    titles = ['Mean Energy MAE', 'Mean HOMO Energy Difference',
              'Mean LUMO Energy Difference', 'Mean Gap Energy Difference']

    for ax, d, title in zip(axes, data, titles):
        # Convert from Hartree to eV
        d_ev = np.array(d) * Hartree

        ax.loglog(noise_levels, d_ev)
        ax.set_xlabel('Noise level')
        ax.set_ylabel('Energy (eV)')
        ax.set_title(f'{title} vs Noise level')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    log_file_list = '/personal/ham_data/0924/test_gaussian_logs.txt'
    db_path = '/root/local_test_ham_overlap/1010_test/real_gau_mat_v5/gaussian_systems.db'
    dump_folder = '/root/local_test_ham_overlap/1010_test/real_gau_mat_v5/test_matrices_dump_place'

    noise_lower = 1e-8
    noise_upper = 1e-4
    grid = 50

    # Step 1: Process Gaussian logs and create ASE database
    log_files = get_gau_logs(log_file_list)
    process_gaussian_logs(log_files, db_path, dump_folder)

    # Step 2: Add noise and analyze
    all_results = analyze_all_systems(db_path, noise_lower, noise_upper, grid)

    # Step 3: Calculate mean results
    mean_results = calculate_mean_results(all_results)

    # Step 4: Extract data for plotting
    noise_levels = [r['noise_level'] for r in mean_results]
    energy_maes = [r['energy_mae'] for r in mean_results]
    coeff_similarities = [r['coeff_similarity'] for r in mean_results]
    homo_coeff_similarities = [r['homo_coeff_similarity'] for r in mean_results]
    homo_energy_diffs = [r['homo_energy_diff'] for r in mean_results]
    lumo_energy_diffs = [r['lumo_energy_diff'] for r in mean_results]
    gap_energy_diffs = [r['gap_energy_diff'] for r in mean_results]

    # Step 6: Plot and save figures
    all_sim_80, homo_sim_80 = plot_coeff_similarity(noise_levels, coeff_similarities, homo_coeff_similarities,
                                     '/root/local_test_ham_overlap/1010_test/real_gau_mat_v5/mean_coeff_similarity.png')

    plot_energy_differences(noise_levels, energy_maes, homo_energy_diffs, lumo_energy_diffs, gap_energy_diffs,
                            '/root/local_test_ham_overlap/1010_test/real_gau_mat_v5/energy_differences.png')

    print(f"Estimated noise level for 80% coefficient similarity: {all_sim_80:.2e}")
    print(f"Estimated noise level for 80% HOMO coefficient similarity: {homo_sim_80:.2e}")


if __name__ == "__main__":
    main()