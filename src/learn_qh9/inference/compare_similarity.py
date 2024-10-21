import os
import shutil

import numpy as np
import torch
from tqdm import trange
import json
import pyscf
from pyscf import gto, tools
from ase.db import connect
from ase.units import Hartree
from scipy.linalg import sqrtm
import time
from pprint import pprint


def load_matrices(idx, folder_path):
    path = os.path.join(folder_path, f'{idx}')
    ham = np.load(os.path.join(path, 'fock.npy'))
    overlap = np.load(os.path.join(path, 'overlap.npy'))
    return ham, overlap


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)

    overlap_matrix = torch.tensor(overlap_matrix)
    full_hamiltonian = torch.tensor(full_hamiltonian)

    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = 1e-8 * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)

    return orbital_energies, orbital_coefficients


def orbital_similarity(coeff1, basis1, coeff2, basis2, overlap_12):
    coeff1, coeff2 = coeff1.numpy(), coeff2.numpy()

    coeff1 = np.atleast_2d(coeff1).T if coeff1.ndim == 1 else coeff1
    coeff2 = np.atleast_2d(coeff2).T if coeff2.ndim == 1 else coeff2

    metric1 = sqrtm(np.linalg.inv(basis1))
    metric2 = sqrtm(np.linalg.inv(basis2))

    coeff1_norm = metric1.dot(coeff1)
    coeff2_norm = metric2.dot(coeff2)

    overlap = np.dot(coeff1_norm.T, overlap_12).dot(coeff2_norm)
    similarity = np.abs(overlap)

    return similarity[0, 0]  # Return as a scalar


def process_molecule(idx, def2svp_folder, g631dp_folder, ase_db_path, cube_path, n_grid=75, dump_threshold=0.8):
    cwd_ = os.getcwd()
    def2svp_ham, def2svp_overlap = load_matrices(idx, def2svp_folder)
    g631dp_ham, g631dp_overlap = load_matrices(idx, g631dp_folder)

    # Calculate orbital energies and coefficients
    def2svp_energies, def2svp_coeffs = cal_orbital_and_energies(def2svp_overlap, def2svp_ham)
    g631dp_energies, g631dp_coeffs = cal_orbital_and_energies(g631dp_overlap, g631dp_ham)

    # Load atomic coordinates (assuming they are stored in a file named 'coordinates.npy' in the def2svp folder)
    with connect(ase_db_path) as db:
        for a_row in db.select():
            if idx == a_row.index_number:
                an_atoms = a_row.toatoms()
                break

    # Get the number of occupied orbitals (assuming closed-shell)
    num_occ = int(sum(a_row.numbers) / 2)

    # Get HOMO coefficients
    def2svp_homo = def2svp_coeffs[0][:, num_occ - 1]
    g631dp_homo = g631dp_coeffs[0][:, num_occ - 1]

    t = [[an_atoms.numbers[atom_idx], an_atoms.positions[atom_idx]]
         for atom_idx in range(len(an_atoms))]

    # Create PySCF mol objects for both basis sets
    mol_def2svp = gto.Mole()
    mol_def2svp.build(verbose=0, atom=t, basis='def2svp', unit='ang')

    mol_g631dp = gto.Mole()
    mol_g631dp.build(verbose=0, atom=t, basis='6-311+g(d,p)', unit='ang')

    # Calculate overlap between the two basis sets
    overlap_12 = gto.intor_cross('int1e_ovlp', mol_def2svp, mol_g631dp)

    # Calculate orbital similarity
    similarity = orbital_similarity(def2svp_homo, def2svp_overlap, g631dp_homo, g631dp_overlap, overlap_12)
    os.chdir(cwd_)
    os.chdir(cube_path)
    filename = f'idx_{idx}_sim_{round(similarity, 2)}'
    os.makedirs(filename)
    if similarity > dump_threshold:
        os.chdir(filename)
        tools.cubegen.orbital(mol_g631dp, 'g631dp_HOMO.cube', g631dp_homo, nx=n_grid, ny=n_grid, nz=n_grid)
        tools.cubegen.orbital(mol_def2svp, 'def2svp_HOMO.cube', def2svp_homo, nx=n_grid, ny=n_grid, nz=n_grid)

    def2svp_HOMO, g631dp_HOMO = def2svp_energies[0, num_occ - 1], g631dp_energies[0, num_occ - 1]
    HOMO_diff = def2svp_HOMO - g631dp_HOMO
    homo_energy_mae = np.abs(HOMO_diff.numpy()) * Hartree
    return {
        'idx': idx,
        'similarity': similarity,
        'homo_energy_mae': homo_energy_mae,
    }


def safe_json_dump(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    def2svp_folder = '/root/matrix_dump/matrices_dump_def2svp'
    g631dp_folder = '/root/matrix_dump/matrices_dump_6311gdp'
    ase_db_path = r'/root/matrix_dump/gaussian_systems_6311gdp.db'
    cube_path = r'/root/fchk2gau/cubes'

    if os.path.exists(cube_path):
        shutil.rmtree(cube_path)
    os.makedirs(cube_path)
    with connect(ase_db_path) as db:
        n_structures = db.count()
    results = []
    start_time = time.time()
    for idx in trange(n_structures):
        result = process_molecule(idx, def2svp_folder, g631dp_folder, ase_db_path, cube_path, dump_threshold=0)
        results.append(result)
        # if idx == 5:
        #     break

    total_time = time.time() - start_time

    # Calculate average values
    total_items = len(results)
    avg_similarity = np.mean([r['similarity'] for r in results])
    avg_homo_mae = np.mean([r['homo_energy_mae'] for r in results])
    avg_processing_time = total_time / total_items

    # Prepare final results dictionary
    final_results = {
        'total_items': total_items,
        'avg_similarity': avg_similarity,
        'avg_homo_mae': avg_homo_mae,
        'avg_processing_time': avg_processing_time,
    }

    # Print and save results
    pprint(final_results)
    safe_json_dump(data=final_results, filepath='final_results.json')


if __name__ == '__main__':
    main()