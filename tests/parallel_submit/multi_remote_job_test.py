import sys

sys.path.append('./')

import json

from qhnet_dpdispatcher import QHNetDpdispatcher

private_para = {
    "training": {
        "train_batch_size": [32, 16]
    },
}

public_para = {
    'general': {
        'seed': 0,
        'device': 'cuda',
        'output_dir': r'./output'
    },
    'dataset': {
        'src_lmdb_folder_path': r'/bohr/qhnet-2500-kk7e/v1/dummy_2500',
        'split': 'size_ood',
        'num_workers': 4,
        'pin_memory': True,
    },
    'training': {
        'train_batch_size': 32,
        'learning_rate': 1e-3,
        'warmup_steps': 1000,
        'total_steps': 100000,
        'lr_end': 1e-5,
        'ema_start_epoch': 5,
        'use_gradient_clipping': True,
        'clip_norm': 1.0,
        'log_interval': 100
    },
    'validation': {
        'valid_batch_size': 32,
        'valid_interval': 1000,
        'save_interval': 30000
    },
    'testing': {
        'test_batch_size': 32
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
# -----------------------------------------------------------------------------------------------------------------
machine_info = {
    "batch_type": "Bohrium",
    "context_type": "Bohrium",
    'local_root': "./",
    'remote_root': './test_dpdispatcher',
    'remote_profile': {
        "email": '1660810667@qq.com',
        "password": 'frank685231_',
        "project_id": 14480,
        "input_data": {
            "job_type": "container",
            "log_file": "log",
            "job_name": "dptb_test",
            "disk_size": 200,
            "scass_type": "1 * NVIDIA V100_32g",
            "platform": 'ali',
            "image_name": "registry.dp.tech/dptech/prod-11729/qhnet:0903",
            "dataset_path": ['/bohr/qhnet-2500-kk7e/v1']
        },
    }
}

resrc_info = {
    'number_node': 1,
    'cpu_per_node': 12,
    'gpu_per_node': 1,
    'group_size': 1,
    'queue_name': "LBG_GPU",
    'envs': {
        "PYTHONUNBUFFERED": "1"
    }
}
a_dpdispatcher_eval = QHNetDpdispatcher(private_para_dict=private_para,
                                        public_para_dict=public_para,
                                        machine_info=machine_info,
                                        resrc_info=resrc_info,
                                        handler_file=r'qhnet_unit.py')
a_dpdispatcher_eval.run_with_dpdispatcher()
