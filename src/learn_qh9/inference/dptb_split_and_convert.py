from dftio.io.gaussian.gaussian_tools import split_files_by_atoms
import os

from dftio.io.gaussian.gaussian_parser import GaussianParser
from dftio.io.gaussian.gaussian_tools import get_convention, chk_valid_gau_logs
from tqdm import tqdm
import numpy as np
from ase.db.core import connect
from ase.io.gaussian import read_gaussian_out


def convert_unit(valid_gau_info_path):
    convention = get_convention(filename=r'/share/ham_data/test_dftio/get_gau_lmdb/gau_with_pop.log',
                                dump_file='convention.txt')
    a_gau_parser = GaussianParser(root=r'/personal/thu_ham_data/rescue_0917/14536526/*/cooking', prefix='ep*/gau.log',
                                  convention_file='convention.txt', valid_gau_info_path=valid_gau_info_path,
                                  add_phase_transfer=False)
    for i in tqdm(range(len(a_gau_parser)), desc="Parsing the DFT files: "):
        a_gau_parser.write(idx=i, format='lmdb', hamiltonian=True, overlap=True,
                           outroot=r'./', eigenvalue=False, density_matrix=False)

    def get_gau_logs(valid_file_path: str):
        raw_datas = []
        with open(valid_file_path, 'r') as file:
            for line in file.readlines():
                raw_datas.append(line.strip())
        return raw_datas

    gau_logs = get_gau_logs(valid_gau_info_path)
    with connect('dump.db') as db:
        for a_gau_log in gau_logs:
            with open(a_gau_log, 'r') as f:
                an_atoms = read_gaussian_out(f)
            db.write(an_atoms)


# chk_valid_gau_logs(root=r'/personal/thu_ham_data/rescue_0917/14536526/*/cooking', prefix='ep*/gau.log',
#                    hamiltonian=True, overlap=True, is_fixed_convention=True)

# split_files_by_atoms(
#     file_list_path=r'./valid_gaussian_logs.txt',
#     output_train_path=r'train_gaussian_logs.txt',
#     output_valid_path=r'valid_gaussian_logs.txt',
#     output_test_path=r'test_gaussian_logs.txt',
# )
abs_train_gaussian_logs_path = os.path.abspath(r'train_gaussian_logs.txt')
abs_valid_gaussian_logs_path = os.path.abspath(r'valid_gaussian_logs.txt')
abs_test_gaussian_logs_path = os.path.abspath(r'test_gaussian_logs.txt')


cwd_ = os.getcwd()
os.makedirs('train')
os.chdir('train')
convert_unit(abs_train_gaussian_logs_path)
os.chdir(cwd_)

cwd_ = os.getcwd()
os.makedirs('valid')
os.chdir('valid')
convert_unit(abs_valid_gaussian_logs_path)
os.chdir(cwd_)

cwd_ = os.getcwd()
os.makedirs('test')
os.chdir('test')
convert_unit(abs_test_gaussian_logs_path)
os.chdir(cwd_)
