import os
import shutil
from pprint import pprint

import numpy as np
import torch
from dptb.data import AtomicDataset, DataLoader, AtomicData, AtomicDataDict
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.nn.hr2hk import HR2HK
from dptb.utils.argcheck import normalize, collect_cutoffs
from dptb.utils.tools import j_loader
from ase.db.core import connect
from ase.atoms import Atoms
import time


def batch_info_2_npy(batch_info, model, device, has_overlap, filename_flag):
    batch_info['kpoint'] = torch.tensor([0.0, 0.0, 0.0], device=device)
    a_ham_hr2hk = HR2HK(
        idp=model.idp,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        overlap=True,
        device=device
    )
    ham_out_data = a_ham_hr2hk.forward(batch_info)
    a_ham = ham_out_data[AtomicDataDict.HAMILTONIAN_KEY]
    ham_ndarray = a_ham.real.cpu().numpy()
    np.save(f'{filename_flag}_ham.npy', ham_ndarray[0])

    if has_overlap:
        an_overlap_hr2hk = HR2HK(
            idp=model.idp,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            overlap=True,
            device=device
        )

        overlap_out_data = an_overlap_hr2hk.forward(batch_info)
        an_overlap = overlap_out_data[AtomicDataDict.OVERLAP_KEY]
        overlap_ndarray = an_overlap.real.cpu().numpy()
        np.save(f'{filename_flag}_overlap.npy', overlap_ndarray[0])


def save_info(folder_path, idx, original_data, predicted_data, model, device, has_overlap):
    cwd_ = os.getcwd()
    os.chdir(folder_path)
    os.makedirs(f'{idx}')
    os.chdir(f'{idx}')
    batch_info_2_npy(batch_info=original_data, model=model, device=device, has_overlap=has_overlap, filename_flag='original')
    batch_info_2_npy(batch_info=predicted_data, model=model, device=device, has_overlap=has_overlap, filename_flag='predicted')
    os.chdir(cwd_)


checkpoint_path = r'/root/local_test_ham_overlap/0923/5.pth'
input_path = r'/root/local_test_ham_overlap/0923/input.json'
out_info_path = r'/root/local_test_ham_overlap/0923/output'
ase_db_path = r'dump.db'
abs_ase_db_path = os.path.abspath(ase_db_path)
if os.path.exists(out_info_path):
    shutil.rmtree(out_info_path)
    os.remove(abs_ase_db_path)
os.makedirs(out_info_path)

reference_info = {
    "root": r"/personal/ham_data/0911_inference_db/14536521_new",
    "prefix": "data",
    "type": "LMDBDataset",
    "get_Hamiltonian": True,
    "get_overlap": True
}

device = 'cuda'
device = torch.device(device)
model = build_model(checkpoint=checkpoint_path)
model.to(device)
jdata = j_loader(input_path)
cutoff_options = collect_cutoffs(jdata)

reference_datasets = build_dataset(**cutoff_options, **reference_info, **jdata["common_options"])
reference_loader = DataLoader(dataset=reference_datasets, batch_size=1, shuffle=False)
start = time.time()
for idx, a_ref_batch in enumerate(reference_loader):
    batch = a_ref_batch.to(device)
    batch = AtomicData.to_AtomicDataDict(batch)
    original_data = batch.copy()
    with torch.no_grad():
        predicted_data = model(batch)
    save_info(folder_path=out_info_path, idx=idx,
              original_data=original_data, predicted_data=predicted_data,
              model=model, device=device, has_overlap=True)
    type_mapper = reference_datasets.type_mapper
    atomic_nums = original_data['atom_types'].cpu().reshape(-1)
    atomic_nums = type_mapper.untransform(atomic_nums).numpy()
    pos = original_data['pos'].cpu().numpy()
    an_atoms = Atoms(symbols=atomic_nums, positions=pos)
    with connect(abs_ase_db_path) as db:
        db.write(an_atoms)
    if idx == 3:
        break

end = time.time()
time_per_item = (end - start) / (idx + 1)
print(f'time_per_item: {time_per_item}')
