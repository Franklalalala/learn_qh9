from learn_qh9.trainer import Trainer

params = {
    'general': {
        'seed': 0,
        'device': 'cuda',
        'output_dir': r'./output'
    },
    'dataset': {
        'src_lmdb_folder_path': r'./dummy_10',
        'split': 'random',
        'num_workers': 4,
        'pin_memory': True,
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

a_trainer = Trainer(params)
a_trainer.train()
