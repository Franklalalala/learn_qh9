import os
import json
import time
import argparse
import numpy as np
import torch
import logging
from tqdm import tqdm
from scipy.linalg import sqrtm
from torch_geometric.loader import DataLoader
from pyscf import gto, scf
from ase.units import Hartree
from ase import Atoms
from ase.db import connect

# Reuse modules
from learn_qh9.datasets import CustomizedQH9Stable, convention_dict, matrix_transform
from learn_qh9.models.ori_QHNet_with_bias_norm import QHNet
from learn_qh9.tools import set_logger

logger = set_logger()

# Mapping from dataset convention string to PySCF basis string
CONVENTION_TO_BASIS = {
    'thu_cluster': 'def2-svp',
    'pyscf_def2svp': 'def2-svp',
    'gau_def2svp_2_pyscf': 'def2-svp',
    'pyscf_6311_plus_gdp': '6-311+g(d,p)',
    'back2pyscf': 'def2-svp',
    'back_thu_cluster': 'def2-svp'
}


class Tester:
    def __init__(self, config_path, checkpoint_path, mode='infer', output_folder=None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.mode = mode

        with open(config_path, 'r') as f:
            self.params = json.load(f)

        # Update output directory for results
        self.base_output_dir = self.params['general']['output_dir']
        if output_folder:
            self.base_output_dir = output_folder

        # Directory for NPYs
        self.npy_save_dir = os.path.join(self.base_output_dir, 'predictions_npy')
        os.makedirs(self.npy_save_dir, exist_ok=True)

        # Path for ASE Database
        self.db_path = os.path.join(self.base_output_dir, 'structures.db')

        self.default_type = torch.float32
        torch.set_default_dtype(self.default_type)

        # --- Config Settings ---
        self.max_items = self.params.get('testing', {}).get('max_items', 3)
        if self.max_items != -1:
            logger.info(f"Test limited to {self.max_items} items.")
        else:
            logger.info("Test will run on ALL items.")

        # Get Aux keys from config to pass to dataset
        self.aux_keys = self.params['dataset'].get('aux_keys', [])

        self.setup_device()
        self.setup_datasets()
        self.setup_model()
        self.load_checkpoint(checkpoint_path)
        self.setup_convention_helpers()

    def setup_device(self):
        input_device = self.params['general']['device'].lower()
        if torch.cuda.is_available() and 'cuda' in input_device:
            self.device = torch.device(input_device)
        else:
            self.device = torch.device('cpu')
        logger.info(f"Testing on device: {self.device}")

    def setup_datasets(self):
        src_lmdb_folder_path = os.path.abspath(self.params['dataset']['src_lmdb_folder_path'])

        # Pass aux_keys strictly to dataset
        dataset_test = CustomizedQH9Stable(
            src_lmdb_folder_path=src_lmdb_folder_path,
            is_debug=False,
            split='pre_splitted',
            target=self.params['dataset']['target'],
            split_flag='test',
            convention=self.params['dataset']['convention'],
            db_workbase=os.path.join(self.base_output_dir, 'data_cache'),
            aux_keys_list=self.aux_keys
        )

        self.test_dataset = dataset_test[list(dataset_test.test_mask)]

        # Batch size MUST be 1 for complex PySCF analysis and precise reconstruction/saving
        batch_size = 1 if self.mode == 'analysis' else self.params['testing'].get('test_batch_size', 1)

        # IMPORTANT: Shuffle MUST be False to align IDs with iteration index
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.params['dataset']['num_workers'],
            pin_memory=self.params['dataset']['pin_memory']
        )
        logger.info(f"Loaded test dataset: {len(self.test_dataset)} samples. Batch size: {batch_size}")

    def setup_model(self):
        self.model = QHNet(
            in_node_features=self.params['model']['in_node_features'],
            sh_lmax=self.params['model']['sh_lmax'],
            hidden_size=self.params['model']['hidden_size'],
            bottle_hidden_size=self.params['model']['bottle_hidden_size'],
            num_gnn_layers=self.params['model']['num_gnn_layers'],
            max_radius=self.params['model']['max_radius'],
            num_nodes=self.params['model']['num_nodes'],
            radius_embed_dim=self.params['model']['radius_embed_dim'],
            convention=self.params['dataset']['convention']
        )
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, path):
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

    def setup_convention_helpers(self):
        conv_name = self.params['dataset']['convention']
        self.convention_name = conv_name

        if conv_name == 'thu_cluster':
            self.back_convention_name = 'back_thu_cluster'
        elif conv_name == 'pyscf_def2svp':
            self.back_convention_name = 'back2pyscf'
        else:
            self.back_convention_name = conv_name

        if hasattr(self.test_dataset, 'dataset'):
            self.orbital_mask = self.test_dataset.dataset.orbital_mask
        else:
            self.orbital_mask = self.test_dataset.orbital_mask

        self.pyscf_basis = CONVENTION_TO_BASIS.get(conv_name, 'def2-svp')
        logger.info(f"Convention: {conv_name} -> PySCF Basis: {self.pyscf_basis}")
        logger.info(f"Backward Convention: {self.back_convention_name}")

    def reconstruct_full_matrix(self, batch, diag_blocks, non_diag_blocks):
        atoms = batch.atoms.flatten().cpu().numpy()
        num_atoms = len(atoms)

        atom_basis_sizes = [len(self.orbital_mask[int(a)]) for a in atoms]
        total_basis = sum(atom_basis_sizes)
        full_matrix = torch.zeros((total_basis, total_basis), dtype=torch.float64, device=diag_blocks.device)
        start_indices = np.cumsum([0] + atom_basis_sizes[:-1])

        # 1. 填充对角块 (Diagonal Blocks)
        for i in range(num_atoms):
            atom_type = int(atoms[i])
            mask = self.orbital_mask[atom_type].to(diag_blocks.device)
            # block[mask][:, mask] 是正确的
            valid_block = diag_blocks[i][mask][:, mask]

            start, end = start_indices[i], start_indices[i] + atom_basis_sizes[i]
            full_matrix[start:end, start:end] = valid_block

        # 2. 填充非对角块 (Non-Diagonal Blocks)
        edge_index = getattr(batch, 'edge_index_full', batch.edge_index)
        if edge_index.shape[0] != 2: edge_index = edge_index.t()

        # 根据 cut_matrix 的逻辑，edge_index 是 [Row_Atom, Col_Atom]
        row_atoms_list = edge_index[0]
        col_atoms_list = edge_index[1]

        for k in range(len(row_atoms_list)):
            idx_row = row_atoms_list[k].item()  # 行原子索引 (j)
            idx_col = col_atoms_list[k].item()  # 列原子索引 (i)

            if idx_row == idx_col: continue

            atom_row = int(atoms[idx_row])
            atom_col = int(atoms[idx_col])

            mask_row = self.orbital_mask[atom_row].to(non_diag_blocks.device)
            mask_col = self.orbital_mask[atom_col].to(non_diag_blocks.device)

            # 提取 Block: 行取 mask_row, 列取 mask_col
            valid_block = non_diag_blocks[k][mask_row][:, mask_col]

            # 计算在大矩阵中的位置
            # Row Start/End 来自 idx_row
            r_s = start_indices[idx_row]
            r_e = r_s + atom_basis_sizes[idx_row]

            # Col Start/End 来自 idx_col
            c_s = start_indices[idx_col]
            c_e = c_s + atom_basis_sizes[idx_col]

            # 赋值
            full_matrix[r_s:r_e, c_s:c_e] = valid_block

        return full_matrix

    def inverse_transform(self, matrix_tensor, atoms):
        mat_np = matrix_tensor.detach().cpu().numpy()
        atoms_np = atoms.flatten().cpu().numpy()
        mat_pyscf = matrix_transform(mat_np, atoms_np, convention=self.back_convention_name)
        return torch.tensor(mat_pyscf)

    def build_pyscf_mol(self, atoms_Z, pos_Ang):
        atom_spec = []
        for z, r in zip(atoms_Z, pos_Ang):
            atom_spec.append([int(z), (float(r[0]), float(r[1]), float(r[2]))])

        mol = gto.Mole()
        mol.atom = atom_spec
        mol.basis = self.pyscf_basis
        mol.unit = 'Angstrom'
        mol.verbose = 0
        mol.spin = 0
        mol.build()
        return mol

    def get_ortho_matrix(self, S_tensor):
        eigvals, eigvecs = torch.linalg.eigh(S_tensor)
        eigvals = torch.where(eigvals > 1e-10, eigvals, torch.tensor(1e-10, device=S_tensor.device))
        inv_sqrt_eigvals = torch.diag(1.0 / torch.sqrt(eigvals))
        X = eigvecs @ inv_sqrt_eigvals @ eigvecs.T
        return X

    def solve_eigenvalues(self, H, S):
        device = H.device
        X = self.get_ortho_matrix(S)
        H_prime = X.T @ H @ X
        energies, C_prime = torch.linalg.eigh(H_prime)
        C = X @ C_prime
        return energies, C

    def run_analysis_heavy(self, idx, pred_matrix_pyscf, atoms_Z, pos_Ang):
        try:
            mol = self.build_pyscf_mol(atoms_Z, pos_Ang)
            S_np = mol.intor('int1e_ovlp')
            S = torch.tensor(S_np, dtype=torch.float64)

            mf = scf.RHF(mol)
            mf.kernel()
            gt_fock = torch.tensor(mf.get_fock(), dtype=torch.float64)

            target_type = self.params['dataset']['target']

            if target_type == 'density_matrix':
                pred_dm_np = pred_matrix_pyscf.cpu().numpy()
                pred_fock_np = mf.get_fock(dm=pred_dm_np)
                pred_target_H = torch.tensor(pred_fock_np, dtype=torch.float64)
            else:
                pred_target_H = pred_matrix_pyscf.cpu()

            gt_eps, gt_C = self.solve_eigenvalues(gt_fock, S)
            pred_eps, pred_C = self.solve_eigenvalues(pred_target_H, S)

            n_elec = mol.nelectron
            n_occ = n_elec // 2

            gt_homo_E = gt_eps[n_occ - 1].item()
            pred_homo_E = pred_eps[n_occ - 1].item()
            homo_diff_ev = abs(gt_homo_E - pred_homo_E) * Hartree

            gt_homo_coeff = gt_C[:, n_occ - 1].numpy()
            pred_homo_coeff = pred_C[:, n_occ - 1].numpy()
            sim = abs(gt_homo_coeff.T @ S_np @ pred_homo_coeff)

            return {
                'id': idx,
                'similarity': sim,
                'homo_mae_ev': homo_diff_ev,
                'gt_homo_h': gt_homo_E,
                'pred_homo_h': pred_homo_E,
            }

        except Exception as e:
            logger.error(f"Analysis failed for molecule {idx}: {e}")
            return None

    def post_processing(self, batch):
        for key in batch.keys():
            if torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
                batch[key] = batch[key].type(self.default_type)
        return batch

    def test(self):
        logger.info(f"Starting Testing in MODE: {self.mode}")

        results_agg = {'similarity': [], 'homo_mae': []}

        if self.mode == 'infer':
            logger.info(f"Saving structures to {self.db_path}")

        # Use enumerate to guarantee index alignment
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_data_loader, desc=f"Test ({self.mode})")):

                # Check Limit
                if self.max_items != -1 and batch_idx >= self.max_items:
                    logger.info(f"Reached max_items limit ({self.max_items}). Stopping.")
                    break

                batch = self.post_processing(batch)
                batch = batch.to(self.device)
                outputs = self.model(batch)

                # 1. Reconstruct Prediction
                pred_full_dataset_conv = self.reconstruct_full_matrix(
                    batch,
                    outputs['hamiltonian_diagonal_blocks'],
                    outputs['hamiltonian_non_diagonal_blocks']
                )
                pred_full_pyscf = self.inverse_transform(pred_full_dataset_conv, batch.atoms)

                # 2. Reconstruct Label
                label_full_dataset_conv = self.reconstruct_full_matrix(
                    batch,
                    batch.diagonal_hamiltonian,
                    batch.non_diagonal_hamiltonian
                )
                label_full_pyscf = self.inverse_transform(label_full_dataset_conv, batch.atoms)

                # Use STRICT batch_idx for folder and ID alignment
                sample_id = str(batch_idx)

                atoms_Z = batch.atoms.flatten().cpu().tolist()
                pos_Ang = batch.pos.cpu().tolist()

                if self.mode == 'infer':
                    # A. Save NPYs to folder named by batch_idx
                    sample_folder = os.path.join(self.npy_save_dir, sample_id)
                    os.makedirs(sample_folder, exist_ok=True)

                    np.save(os.path.join(sample_folder, 'pred.npy'), pred_full_pyscf.cpu().numpy())
                    np.save(os.path.join(sample_folder, 'label.npy'), label_full_pyscf.cpu().numpy())

                    # B. Save Structure to ASE DB
                    atoms_obj = Atoms(numbers=atoms_Z, positions=pos_Ang)

                    # Data dict for DB: Original ID + Aux Keys
                    data_to_store = {}

                    # 1. Original ID from Dataset
                    if hasattr(batch, 'original_id'):
                        oid = batch.original_id
                        if torch.is_tensor(oid):
                            data_to_store['original_id'] = int(oid[0].item())
                        elif isinstance(oid, list):
                            data_to_store['original_id'] = oid[0]
                        else:
                            data_to_store['original_id'] = oid

                    # 2. Add ONLY the requested Aux keys
                    for key in self.aux_keys:
                        if hasattr(batch, key):
                            val = getattr(batch, key)
                            if torch.is_tensor(val):
                                if val.numel() == 1:
                                    data_to_store[key] = val.item()
                                else:
                                    data_to_store[key] = val.tolist()
                            elif isinstance(val, list) and len(val) > 0:
                                data_to_store[key] = val[0]
                            else:
                                data_to_store[key] = val

                    # C. Write with test_idx as queryable key
                    with connect(self.db_path) as db:
                        # 'test_idx' allows retrieving this row for folder '0', '1', etc.
                        db.write(atoms_obj, key_value_pairs={'test_idx': batch_idx}, data=data_to_store)

                elif self.mode == 'analysis':
                    # Heavy Analysis Mode
                    res = self.run_analysis_heavy(sample_id, pred_full_pyscf, atoms_Z, pos_Ang)
                    if res:
                        results_agg['similarity'].append(res['similarity'])
                        results_agg['homo_mae'].append(res['homo_mae_ev'])

                        if batch_idx % 100 == 0:
                            logger.info(
                                f"ID {sample_id}: Sim={res['similarity']:.4f}, HOMO_Diff={res['homo_mae_ev']:.4f} eV")

        # Final Reporting
        if self.mode == 'analysis' and results_agg['similarity']:
            avg_sim = np.mean(results_agg['similarity'])
            avg_homo = np.mean(results_agg['homo_mae'])
            final_res = {
                'total_samples': len(results_agg['similarity']),
                'avg_similarity': avg_sim,
                'avg_homo_mae_ev': avg_homo
            }
            logger.info("========== Analysis Results ==========")
            logger.info(json.dumps(final_res, indent=4))
            with open(os.path.join(self.base_output_dir, 'analysis_results.json'), 'w') as f:
                json.dump(final_res, f, indent=4)

        elif self.mode == 'infer':
            logger.info(f"Inference complete. NPYs saved to {self.npy_save_dir}, DB saved to {self.db_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config json')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pt')
    parser.add_argument('--mode', type=str, default='infer', choices=['infer', 'analysis'],
                        help="Mode: 'infer' (save npy & db) or 'analysis' (run pyscf similarity)")
    parser.add_argument('--output', type=str, default=None, help='Override output folder')

    args = parser.parse_args()

    tester = Tester(args.config, args.checkpoint, mode=args.mode, output_folder=args.output)
    tester.test()


if __name__ == '__main__':
    main()