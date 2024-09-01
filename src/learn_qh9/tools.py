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


def lmdb_2_small_lmdb(old_lmdb_path: str, new_lmdb_path: str, n_small: int, is_random: bool):
    old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
    new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)
    with old_db_env.begin() as txn:
        num_entries = old_db_env.stat()["entries"]
        old_id_list = list(range(num_entries))
        if is_random:
            random.shuffle(old_id_list)
        for an_id in old_id_list[:n_small]:
            value = txn.get(an_id.to_bytes(length=4, byteorder='big'))
            data_dict = parse_lmdb_info_2_readable(value)
            data_dict = pickle.dumps(data_dict)
            with new_db_env.begin(write=True) as new_txn:
                entries = new_db_env.stat()["entries"]
                new_txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)

    pass

