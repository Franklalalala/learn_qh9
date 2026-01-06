from learn_qh9_e3x.datasets import CustomizedQH9StableJax, create_data_iterator
import jax

params = {
    'general': {
        'seed': 0,
        'device': 'cuda',
        'output_dir': r'./output',
        'is_debug': True
    },
    'dataset': {
        'src_lmdb_folder_path': r'./dummy_10',
        'split': 'random',
        'num_workers': 0,
        'pin_memory': False,
        'convention': 'pyscf_def2svp'
    },
    'training': {
        'train_batch_size': 4,
        'learning_rate': 1e-3,
        'warmup_steps': 1000,
        'total_steps': 100000,
        'lr_end': 1e-5,
        'ema_start_epoch': 5,
        'use_gradient_clipping': True,
        'clip_norm': 1.0,
        'log_interval': 1
    },
    'validation': {
        'valid_batch_size': 1,
        'valid_interval': 2,
        'save_interval': 30000
    },
    'testing': {
        'test_batch_size': 1
    },
    'model': {
        'in_node_features': 1,
        'sh_lmax': 4,
        'hidden_size': 128,
        'bottle_hidden_size': 32,
        'num_gnn_layers': 5,
        'max_radius': 15,
        'num_nodes': 10,
        'radius_embed_dim': 16
    }
}
dataset_random = CustomizedQH9StableJax(
    src_lmdb_folder_path=params['dataset']['src_lmdb_folder_path'],  # Point to the 'train' LMDB as the source for random split
    db_workbase='temp_jax_datasets_processed/qh9_random',
    split_type='random',
    convention=params['dataset']['convention'],
    is_debug=params['general']['is_debug']
)
print(f"Random split dataset initialized. Total items in source LMDB: {len(dataset_random)}")
print(f"Train mask length: {len(dataset_random.train_mask)}")
print(f"Val mask length: {len(dataset_random.val_mask)}")
print(f"Test mask length: {len(dataset_random.test_mask)}")

if len(dataset_random.train_mask) > 0:
    rng_key_rand = jax.random.PRNGKey(123)
    rand_loader_rng, rand_shuffle_rng = jax.random.split(rng_key_rand)
    random_train_iterator = create_data_iterator(
        dataset_random,
        dataset_random.train_mask,
        batch_size=params['training']['train_batch_size'],
        shuffle_rng=rand_shuffle_rng
    )
    print("Iterating one batch from random_train_iterator:")
    try:
        batch_rand = next(random_train_iterator)
        print(f"Batch type: {type(batch_rand)}")
    except StopIteration:
        print("Random train iterator is empty.")
    except Exception as e:
        print(f"Error getting batch from random split: {e}")
dataset_random.close_lmdb()