import os
import numpy as np
from ase.db import connect
from dftio.io.gaussian.gaussian_tools import read_fock_from_gau_log, read_int1e_from_gau_log, get_basic_info, get_convention
from dftio.io.gaussian.gaussian_conventionns import gau_6311_plus_gdp_to_pyscf_convention, orbital_idx_map_2_pyscf
from learn_qh9.parse_gau_logs_tools import transform_matrix, generate_molecule_transform_indices
from tqdm import tqdm
import shutil
from pprint import pprint


def process_gaussian_logs(base_folder, db_path, dump_folder, convention=None):
    if os.path.exists(dump_folder):
        shutil.rmtree(dump_folder)
    if os.path.exists(db_path):
        os.remove(db_path)

    os.makedirs(dump_folder)
    if not convention:
        convention = gau_6311_plus_gdp_to_pyscf_convention

    with connect(db_path) as db:
        for idx, subfolder in enumerate(tqdm(os.listdir(base_folder), desc=f"Processing {base_folder}")):
            log_path = os.path.join(base_folder, subfolder, 'gau.log')
            if not os.path.exists(log_path):
                continue

            save_dir = os.path.join(dump_folder, f'{idx}')
            os.makedirs(save_dir, exist_ok=True)

            nbasis, atoms = get_basic_info(file_path=log_path)
            molecule_transform_indices, _ = generate_molecule_transform_indices(
                atom_types=atoms.symbols,
                atom_to_transform_indices=convention['atom_to_transform_indices']
            )
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
                'basis_set': os.path.basename(base_folder),
                'index_number': idx
            }
            db.write(atoms, key_value_pairs=real_data)


def main():
    # Process 6311gdp set

    # base_folder_6311gdp = '/root/fchk2gau/6311gdp/cooking'
    # db_path_6311gdp = '/root/matrix_dump/gaussian_systems_6311gdp.db'
    # dump_folder_6311gdp = '/root/matrix_dump/matrices_dump_6311gdp'
    #
    # process_gaussian_logs(base_folder_6311gdp, db_path_6311gdp, dump_folder_6311gdp)

    # Process def2svp set
    convention_filepath = r'/root/fchk2gau/def2svp_v2/cooking/ep-12574_vum/gau.log'
    convention = get_convention(filename=convention_filepath, orbital_idx_map=orbital_idx_map_2_pyscf)
    # print(convention)

    base_folder_def2svp = '/root/fchk2gau/def2svp/cooking'
    db_path_def2svp = '/root/matrix_dump/gaussian_systems_def2svp.db'
    dump_folder_def2svp = '/root/matrix_dump/matrices_dump_def2svp'

    process_gaussian_logs(base_folder_def2svp, db_path_def2svp, dump_folder_def2svp, convention=convention)


if __name__ == "__main__":
    main()