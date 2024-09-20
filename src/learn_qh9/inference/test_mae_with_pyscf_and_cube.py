import json
import os.path
import os
import py3Dmol
from argparse import Namespace
from pprint import pprint
import shutil

import torch
import pyscf
from pyscf import tools, scf

from learn_qh9.trainer import Trainer
from learn_qh9.datasets import matrix_transform
from tqdm import tqdm
import time

from torch_scatter import scatter_sum


loss_weights = {
    'hamiltonian': 1.0,
    'diagonal_hamiltonian': 1.0,
    'non_diagonal_hamiltonian': 1.0,
    'orbital_energies': 1.0,
    "orbital_coefficients": 1.0,
    'HOMO': 1.0, 'LUMO': 1.0, 'GAP': 1.0,
}

def process_cube_file(cube_path):
    # Read the cube file
    with open(cube_path, 'r') as f:
        cube_data = f.read()

    # Create visualization
    view = py3Dmol.view()
    view.addModel(cube_data, 'cube')
    view.addVolumetricData(cube_data, "cube", {'isoval': -0.03, 'color': "red", 'opacity': 0.75})
    view.addVolumetricData(cube_data, "cube", {'isoval': 0.03, 'color': "blue", 'opacity': 0.75})
    view.setStyle({'stick':{}})
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

def criterion(outputs, target, names):
    error_dict = {}
    for key in names:
        if key == 'orbital_coefficients':
            "The shape if [batch, total_orb, num_occ_orb]."
            error_dict[key] = torch.cosine_similarity(outputs[key], target[key], dim=1).abs().mean()
        elif key in ['diagonal_hamiltonian', 'non_diagonal_hamiltonian']:
            diff_blocks = outputs[key].cpu() - target[key].cpu()
            mae_blocks = torch.sum(torch.abs(diff_blocks) * target[f"{key}_mask"], dim=[1, 2])
            count_sum_blocks = torch.sum(target[f"{key}_mask"], dim=[1, 2])
            if key == 'non_diagonal_hamiltonian':
                row = target.edge_index_full[0]
                batch = target.batch[row]
            else:
                batch = target.batch
            mae_blocks = scatter_sum(mae_blocks, batch)
            count_sum_blocks = scatter_sum(count_sum_blocks, batch)
            error_dict[key + '_mae'] = (mae_blocks / count_sum_blocks).mean()
        else:
            diff = torch.tensor(outputs[key] - target[key])
            mae = torch.mean(torch.abs(diff))
            error_dict[key] = mae
    return error_dict


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
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


def post_processing(batch, default_type=torch.float32):
    for key in batch.keys():
        if torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
            batch[key] = batch[key].type(default_type)
    return batch


@torch.no_grad()
def test_with_pyscf(model, data_loader, device, cubes_abs_folder, n_grid):
    cwd_ = os.getcwd()
    model.eval()
    total_error_dict = {'total_items': 0}
    start_time = time.time()
    for valid_batch_idx, batch in tqdm(enumerate(data_loader)):
        batch = post_processing(batch)
        batch = batch.to(device)
        outputs = model(batch)

        batch = batch.cpu()

        outputs['hamiltonian'] = model.build_final_matrix(
            batch, outputs['hamiltonian_diagonal_blocks'].cpu(), outputs['hamiltonian_non_diagonal_blocks'].cpu())

        batch.hamiltonian = model.build_final_matrix(
            batch, batch[0].diagonal_hamiltonian, batch[0].non_diagonal_hamiltonian)

        outputs['hamiltonian'] = outputs['hamiltonian'].type(torch.float64)
        outputs['hamiltonian'] = matrix_transform(outputs['hamiltonian'].cpu().numpy(), batch.atoms.squeeze().numpy(), convention='back2pyscf')
        batch.hamiltonian = batch.hamiltonian.type(torch.float64)
        batch.hamiltonian = matrix_transform(batch.hamiltonian.numpy(), batch.atoms.squeeze().numpy(), convention='back2pyscf')

        total_error_dict['total_items'] += (batch.ptr.shape[0] - 1)
        mol = pyscf.gto.Mole()
        t = [[batch.atoms[atom_idx].cpu().item(), batch.pos[atom_idx].cpu().numpy()]
             for atom_idx in range(batch.num_nodes)]
        mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')

        overlap = torch.tensor(mol.intor("int1e_ovlp")).unsqueeze(0)
        # overlap = overlap.to(device)

        outputs['orbital_energies'], outputs['orbital_coefficients'] = \
            cal_orbital_and_energies(overlap, outputs['hamiltonian'])
        batch.orbital_energies, batch.orbital_coefficients = \
            cal_orbital_and_energies(overlap, batch['hamiltonian'])

        # here it only considers the occupied orbitals
        # pay attention that the last dimension here corresponds to the different orbitals.
        num_orb = int(batch.atoms[batch.ptr[0]: batch.ptr[1]].sum() / 2)
        pred_HOMO = outputs['orbital_energies'][:, num_orb-1]
        gt_HOMO = batch.orbital_energies[:, num_orb-1]
        pred_LUMO = outputs['orbital_energies'][:, num_orb]
        gt_LUMO = batch.orbital_energies[:, num_orb]
        outputs['HOMO'], outputs['LUMO'], outputs['GAP'] = pred_HOMO, pred_LUMO, pred_LUMO - pred_HOMO
        batch.HOMO, batch.LUMO, batch.GAP = gt_HOMO, gt_LUMO, gt_LUMO - gt_HOMO

        outputs['orbital_energies'], outputs['orbital_coefficients'], \
        batch.orbital_energies, batch.orbital_coefficients = \
            outputs['orbital_energies'][:, :num_orb], outputs['orbital_coefficients'][:, :, :num_orb], \
            batch.orbital_energies[:, :num_orb], batch.orbital_coefficients[:, :, :num_orb]

        outputs['diagonal_hamiltonian'], outputs['non_diagonal_hamiltonian'] = \
            outputs['hamiltonian_diagonal_blocks'], outputs['hamiltonian_non_diagonal_blocks']
        error_dict = criterion(outputs, batch, loss_weights)

        os.chdir(cubes_abs_folder)
        a_sim = error_dict['orbital_coefficients']
        a_sim = torch.round(a_sim, decimals=2)
        folder_name = f'idx_{valid_batch_idx}_sim_{a_sim}'
        os.makedirs(folder_name)
        os.chdir(folder_name)

        # mf = scf.RHF(mol)
        # homo_idx = mf.mo_occ.argmax()
        tools.cubegen.orbital(mol, 'predicted_HOMO.cube', outputs['orbital_coefficients'][0][:, -1], nx=n_grid, ny=n_grid, nz=n_grid)
        tools.cubegen.orbital(mol, 'real_HOMO.cube', batch.orbital_coefficients[0][:, -1], nx=n_grid, ny=n_grid, nz=n_grid)
        diff = (batch.orbital_coefficients[0][:, -1] - outputs['orbital_coefficients'][0][:, -1])
        tools.cubegen.orbital(mol, 'diff_HOMO.cube', diff, nx=n_grid, ny=n_grid, nz=n_grid)

        os.chdir(cwd_)

        for key in error_dict.keys():
            if key in total_error_dict.keys():
                total_error_dict[key] += error_dict[key].item() * batch.hamiltonian.shape[0]
            else:
                total_error_dict[key] = error_dict[key].item() * batch.hamiltonian.shape[0]

    for key in total_error_dict.keys():
        if key != 'total_items':
            total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']
    end_time = time.time()
    total_error_dict['second_per_item'] = (end_time - start_time) / total_error_dict['total_items']

    return total_error_dict


if __name__ == '__main__':
    if os.path.exists('output'):
        shutil.rmtree('output')

    # new_dataset_path = r'/personal/qh9_data/qh9_all'
    new_dataset_path = r'/personal/qh9_data/dummy_2500'
    best_ckpt_path = r'/personal/qh9_data/remote_all_0903/cooked/ckpt/32.pt'
    train_para_path = r'/personal/qh9_data/remote_all_0903/raw/32/input.json'

    cubes_abs_folder = os.path.abspath(os.path.join('output', 'cubes'))
    os.makedirs(cubes_abs_folder)

    with open(train_para_path, 'r') as json_file:
        params = json.load(json_file)
    params["dataset"]["src_lmdb_folder_path"] = new_dataset_path
    params["testing"]["test_batch_size"] = 1
    params['dataset']['convention'] = 'pyscf_def2svp'
    a_trainer = Trainer(params)

    a_trainer.model.load_state_dict(torch.load(best_ckpt_path)['state_dict'])
    print(a_trainer.model.convention)
    total_error_dict = test_with_pyscf(model=a_trainer.model, data_loader=a_trainer.test_data_loader,
                                       device=a_trainer.device, cubes_abs_folder=cubes_abs_folder, n_grid=75)

    with open('test_results.json', 'w') as f:
        json.dump(total_error_dict, f, indent=2)

    process_batch_cube_file(cubes_abs_folder)
    pprint(total_error_dict)
