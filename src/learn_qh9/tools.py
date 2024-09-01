import random
import pickle
import lmdb
import numpy as np
from ase.db.core import connect
from ase.atoms import Atoms


def parse_lmdb_info_2_readable(value):
    data_dict = pickle.loads(value)
    atoms, pos, Ham = np.frombuffer(data_dict['atoms'], np.int32), \
        np.frombuffer(data_dict['pos'], np.float64), \
        np.frombuffer(data_dict['Ham'], np.float64)
    data_dict['atoms'], data_dict['pos'], data_dict['Ham'] = atoms, pos, Ham
    return data_dict

def parse_lmdb_info_2_ase_readable(value):
    data_dict = pickle.loads(value)
    atoms, pos, Ham = np.frombuffer(data_dict['atoms'], np.int32), \
        np.frombuffer(data_dict['pos'], np.float64), \
        np.frombuffer(data_dict['Ham'], np.float64)
    num_nodes = len(atoms)
    pos = pos.reshape(num_nodes, 3)
    num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
    Ham = Ham.reshape(num_orbitals, num_orbitals)
    data_dict['atoms'], data_dict['pos'], data_dict['Ham'] = atoms, pos, Ham
    return data_dict


def lmdb_2_small_lmdb(old_lmdb_path: str, new_lmdb_path: str, n_small: int, is_random: bool):
    old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
    new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)
    with old_db_env.begin() as txn, new_db_env.begin(write=True) as new_txn:
        num_entries = old_db_env.stat()["entries"]
        old_id_list = list(range(num_entries))
        if is_random:
            random.shuffle(old_id_list)
        for new_idx, an_id in enumerate(old_id_list[:n_small]):
            value = txn.get(an_id.to_bytes(length=4, byteorder='big'))
            data_dict = parse_lmdb_info_2_readable(value)
            data_dict = pickle.dumps(data_dict)
            new_txn.put(new_idx.to_bytes(length=4, byteorder='big'), data_dict)
    old_db_env.close()
    new_db_env.close()


def get_readable_info_from_lmdb(lmdb_path: str):
    db_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with db_env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            idx = int.from_bytes(key, byteorder='big')
            data_dict = parse_lmdb_info_2_ase_readable(value)
            yield idx, data_dict
    db_env.close()


def lmdb_2_ase_db(lmdb_path: str, ase_db_path: str):
    with connect(ase_db_path) as db:
        for idx, info in get_readable_info_from_lmdb(lmdb_path):
            an_atoms = Atoms(symbols=info['atoms'], positions=info['pos'])
            db.write(an_atoms, data=info)


def get_lmdb_size(lmdb_path: str):
    with lmdb.open(lmdb_path, readonly=True, lock=False) as db:
        size = db.stat()["entries"]
    return size



