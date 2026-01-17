import os
import time
from contextlib import nullcontext
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

from learn_qh9.datasets import CustomizedQH9Stable
from learn_qh9.models.ori_QHNet_with_bias_norm import QHNet
from learn_qh9.loss import criterion
from learn_qh9.tools import set_logger

logger = set_logger()


class Trainer:
    def __init__(self, params):
        self.params = params
        self.default_type = torch.float32
        torch.set_default_dtype(self.default_type)
        logger.info(self.params)

        # Seed and Debug
        torch.manual_seed(self.params['general']['seed'])
        self.is_debug = self.params['general'].get('is_debug', False)
        self.ckpt_queue = []
        self.max_ckpt_maintain = self.params['training'].get('max_ckpt_maintain', 5)
        self.setup_device()
        self.setup_output_directories()
        self.setup_datasets()
        self.setup_model()
        self.setup_optimizer()
        self.setup_tensorboard()

    def setup_device(self):
        input_device = self.params['general']['device'].lower()
        if input_device == 'cpu':
            self.device = torch.device('cpu')
        elif 'cuda' in input_device:
            if torch.cuda.is_available():
                try:
                    device = torch.device(input_device)
                    if device.index is not None and device.index >= torch.cuda.device_count():
                        logger.warning(f"'{input_device}' is out of bounds. Defaulting to 'cuda:0'.")
                        self.device = torch.device('cuda:0')
                    else:
                        self.device = device
                    torch.cuda.manual_seed_all(self.params['general']['seed'])
                except Exception as e:
                    logger.warning(f"{e}. Defaulting to 'cpu'.")
                    self.device = torch.device('cpu')
            else:
                logger.warning("CUDA is not available. Defaulting to 'cpu'.")
                self.device = torch.device('cpu')
        else:
            logger.warning(f"Unrecognized device '{input_device}'. Defaulting to 'cpu'.")
            self.device = torch.device('cpu')

    def setup_output_directories(self):
        self.output_dir = os.path.abspath(self.params['general']['output_dir'])
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        self.data_dir = os.path.join(self.output_dir, 'data')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_datasets(self):
        src_lmdb_folder_path = os.path.abspath(self.params['dataset']['src_lmdb_folder_path'])
        if 'convention' not in self.params['dataset'].keys():
            self.params["dataset"].update({'convention': 'pyscf_def2svp'})
            logger.info('Convention not found, set default to pyscf_def2svp.')

        # Logic for pre_splitted dataset
        if self.params['dataset']['split'] == 'pre_splitted':
            dataset_train = CustomizedQH9Stable(src_lmdb_folder_path=src_lmdb_folder_path, is_debug=self.is_debug,
                                                split='pre_splitted', target=self.params['dataset']['target'],
                                                split_flag='train', convention=self.params['dataset']['convention'],
                                                db_workbase=self.data_dir)
            train_dataset = dataset_train[list(dataset_train.train_mask)]

            dataset_val = CustomizedQH9Stable(src_lmdb_folder_path=src_lmdb_folder_path, is_debug=self.is_debug,
                                              split='pre_splitted', target=self.params['dataset']['target'],
                                              split_flag='valid', convention=self.params['dataset']['convention'],
                                              db_workbase=self.data_dir)
            val_dataset = dataset_val[list(dataset_val.val_mask)]

            dataset_test = CustomizedQH9Stable(src_lmdb_folder_path=src_lmdb_folder_path, is_debug=self.is_debug,
                                               split='pre_splitted', target=self.params['dataset']['target'],
                                               split_flag='test', convention=self.params['dataset']['convention'],
                                               db_workbase=self.data_dir)
            test_dataset = dataset_test[list(dataset_test.test_mask)]

            self.train_data_loader = DataLoader(train_dataset, batch_size=self.params['training']['train_batch_size'],
                                                shuffle=True,
                                                num_workers=self.params['dataset']['num_workers'],
                                                pin_memory=self.params['dataset']['pin_memory'])
            self.val_data_loader = DataLoader(val_dataset, batch_size=self.params['validation']['valid_batch_size'],
                                              shuffle=False,
                                              num_workers=self.params['dataset']['num_workers'],
                                              pin_memory=self.params['dataset']['pin_memory'])
            self.test_data_loader = DataLoader(test_dataset, batch_size=self.params['testing']['test_batch_size'],
                                               shuffle=False,
                                               num_workers=self.params['dataset']['num_workers'],
                                               pin_memory=self.params['dataset']['pin_memory'])
            print(
                f'Number of train/valid/test is {len(dataset_train.train_mask)}/{len(dataset_val.val_mask)}/{len(dataset_test.test_mask)}')

        else:
            # Logic for standard split
            logger.info(f"loading source lmdb dataset from {src_lmdb_folder_path}...")
            dataset = CustomizedQH9Stable(src_lmdb_folder_path=src_lmdb_folder_path,
                                          db_workbase=self.data_dir, target=self.params['dataset']['target'],
                                          split=self.params['dataset']['split'],
                                          convention=self.params['dataset']['convention'],
                                          is_debug=self.is_debug)

            if self.is_debug:
                train_dataset = dataset[dataset.train_mask.tolist()]
                valid_dataset = dataset[dataset.val_mask.tolist()]
                test_dataset = dataset[dataset.test_mask.tolist()]
            else:
                train_dataset = dataset[dataset.train_mask]
                valid_dataset = dataset[dataset.val_mask]
                test_dataset = dataset[dataset.test_mask]

            g = torch.Generator()
            g.manual_seed(self.params['general']['seed'])
            self.train_data_loader = DataLoader(train_dataset, batch_size=self.params['training']['train_batch_size'],
                                                shuffle=True,
                                                num_workers=self.params['dataset']['num_workers'],
                                                pin_memory=self.params['dataset']['pin_memory'], generator=g)
            self.val_data_loader = DataLoader(valid_dataset, batch_size=self.params['validation']['valid_batch_size'],
                                              shuffle=False,
                                              num_workers=self.params['dataset']['num_workers'],
                                              pin_memory=self.params['dataset']['pin_memory'])
            self.test_data_loader = DataLoader(test_dataset, batch_size=self.params['testing']['test_batch_size'],
                                               shuffle=False,
                                               num_workers=self.params['dataset']['num_workers'],
                                               pin_memory=self.params['dataset']['pin_memory'])

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
        logger.info(self.model)
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"the number of parameters in this model is {num_params}.")

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params['training']['learning_rate'],
            betas=(0.99, 0.999),
            amsgrad=False)

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.99)
        self.total_steps = self.params['training']['total_steps']
        self.setup_scheduler()

    def setup_scheduler(self):
        train_params = self.params['training']
        scheduler_config = train_params.get('lr_scheduler', {})
        scheduler_type = scheduler_config.get('type', 'poly')

        # Identify if using ROP which requires a metric
        self.is_rop = (scheduler_type == 'rop')

        if self.is_rop:
            # ROP usually defaults to per-validation update (per_iter=False)
            # but we respect the config if explicitly set to True
            self.scheduler_step_on_iter = scheduler_config.get('update_lr_per_iter', False)

            mode = scheduler_config.get('mode', 'min')
            factor = scheduler_config.get('factor', 0.5)
            # If step_on_iter=True, patience is # of iterations.
            # If step_on_iter=False, patience is # of validation checks.
            patience = scheduler_config.get('patience', 10)
            threshold = scheduler_config.get('threshold', 1e-4)
            min_lr = scheduler_config.get('min_lr', 1e-6)

            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                min_lr=min_lr,
            )
            logger.info(f"Using ReduceLROnPlateau. Step on iter: {self.scheduler_step_on_iter}, Patience: {patience}")

        else:
            # Poly/Cosine decay usually steps per iteration
            self.scheduler_step_on_iter = scheduler_config.get('update_lr_per_iter', True)

            if 'cool_down_steps' in train_params.keys():
                self.total_steps_wo_cool_down = self.total_steps - train_params['cool_down_steps']
            else:
                self.total_steps_wo_cool_down = self.total_steps

            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer, num_warmup_steps=train_params['warmup_steps'],
                num_training_steps=self.total_steps_wo_cool_down,
                lr_end=train_params['lr_end'], power=1.0, last_epoch=-1)

            logger.info("Using Polynomial Decay Scheduler")

    def step_scheduler(self, metric=None):
        """Unified step function for different schedulers"""
        if self.is_rop:
            # ROP requires a metric (loss) to step
            if metric is not None:
                self.scheduler.step(metric)
        else:
            # Other schedulers (Poly, etc.) do not use metrics
            self.scheduler.step()

    def setup_tensorboard(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):
        self.model.train()
        epoch = 0
        self.best_val_result = float('inf')
        train_iterator = iter(self.train_data_loader)

        for batch_idx in range(self.total_steps):
            try:
                batch = next(train_iterator)
                batch = self.post_processing(batch)
            except StopIteration:
                epoch += 1
                train_iterator = iter(self.train_data_loader)
                # Retry fetching batch to handle end of epoch edge case
                try:
                    batch = next(train_iterator)
                    batch = self.post_processing(batch)
                except StopIteration:
                    break

            batch = batch.to(self.device)
            errors = self.train_one_batch(batch)

            # --- Scheduler Update (Per Iteration) ---
            if self.scheduler_step_on_iter:
                if self.is_rop:
                    # If ROP is used per-iteration, we must pass the training loss.
                    # Note: Train loss is noisy, high patience recommended.
                    self.step_scheduler(metric=errors['hamiltonian_mae'])
                else:
                    self.step_scheduler()

            if self.params['training']['ema_start_epoch'] > -1 and epoch > self.params['training']['ema_start_epoch']:
                self.ema.update()

            if batch_idx % self.params['training']['log_interval'] == 0:
                self.log_training_progress(epoch, batch_idx, errors)

            if batch_idx % self.params['validation']['valid_interval'] == 0:
                self.validate_and_save(epoch, batch_idx, errors)

        self.writer.close()

    def train_one_batch(self, batch):
        loss_weights = {'hamiltonian': 1.0}
        outputs = self.model(batch)
        errors = criterion(outputs, batch, loss_weights=loss_weights)
        self.optimizer.zero_grad()
        errors['loss'].backward()
        if self.params['training']['use_gradient_clipping']:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['training']['clip_norm'])
        self.optimizer.step()
        return errors

    def validate_and_save(self, epoch, batch_idx, errors):
        logger.info(f"Evaluating on epoch {epoch}")
        use_ema = self.params['training']['ema_start_epoch'] > -1 and epoch > self.params['training']['ema_start_epoch']

        if use_ema:
            logger.info("with ema")
            context_manager = self.ema.average_parameters()
        else:
            context_manager = nullcontext()

        with context_manager:
            val_errors = self.do_valid(self.val_data_loader)

            # --- Scheduler Update (Per Validation) ---
            # Only step here if NOT configured to step per iteration
            if not self.scheduler_step_on_iter:
                # Use Validation Loss (more stable than train loss)
                self.step_scheduler(metric=val_errors['hamiltonian_mae'])

            if val_errors['hamiltonian_mae'] < self.best_val_result:
                self.best_val_result = val_errors['hamiltonian_mae']
                test_errors = self.do_valid(self.test_data_loader)
                self.save_model("results_best.pt", errors, batch_idx)
            else:
                test_errors = None

        if batch_idx % self.params['validation']['save_interval'] == 0:
            self.save_model(f"results_{batch_idx}.pt", errors, batch_idx)

        self.log_validation_results(epoch, batch_idx, errors, val_errors, test_errors)

    @torch.no_grad()
    def do_valid(self, data_loader):
        self.model.eval()
        total_error_dict = {'total_items': 0}
        loss_weights = {'hamiltonian': 1.0}
        start_time = time.time()
        for batch in data_loader:
            batch = self.post_processing(batch)
            batch = batch.to(self.device)
            outputs = self.model(batch)
            error_dict = criterion(outputs, batch, loss_weights)

            for key in error_dict.keys():
                if key not in ['total_items', 'loss']:
                    if key in total_error_dict.keys():
                        total_error_dict[key] += error_dict[key].item() * (batch.ptr.shape[0] - 1)
                    else:
                        total_error_dict[key] = error_dict[key].item() * (batch.ptr.shape[0] - 1)
            total_error_dict['total_items'] += (batch.ptr.shape[0] - 1)

        # Prevent division by zero if dataset is empty
        if total_error_dict['total_items'] == 0:
            total_error_dict['total_items'] = 1

        for key in total_error_dict.keys():
            if key != 'total_items':
                total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']
        end_time = time.time()
        total_error_dict['second_per_item'] = (end_time - start_time) / total_error_dict['total_items']
        return total_error_dict

    def save_model(self, filename, errors, batch_idx):
        save_path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            "state_dict": self.model.state_dict(),
            "eval": errors,
            "batch_idx": batch_idx,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, save_path)

        # --- [新增] Checkpoint 轮替逻辑 ---
        # 如果不是 best 模型，则加入队列管理
        if "best" not in filename:
            self.ckpt_queue.append(save_path)
            if len(self.ckpt_queue) > self.max_ckpt_maintain:
                oldest_ckpt = self.ckpt_queue.pop(0)
                if os.path.exists(oldest_ckpt):
                    try:
                        os.remove(oldest_ckpt)
                        logger.info(f"Removed old checkpoint: {oldest_ckpt}")
                    except OSError as e:
                        logger.warning(f"Failed to remove {oldest_ckpt}: {e}")

    def post_processing(self, batch):
        for key in batch.keys():
            if torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
                batch[key] = batch[key].type(self.default_type)
        return batch

    def log_training_progress(self, epoch, batch_idx, errors):
        logger.info(f"Train: Epoch {epoch} {batch_idx} hamiltonian: {errors['hamiltonian_mae']:.8f}.")
        lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"hamiltonian: diagonal/non diagonal :{errors['hamiltonian_diagonal_mae']:.8f}, "
                    f"{errors['hamiltonian_non_diagonal_mae']:.8f}, lr: {lr}.")

        self.writer.add_scalar('Train/Hamiltonian_MAE', errors['hamiltonian_mae'], batch_idx)
        self.writer.add_scalar('Train/Hamiltonian_Diagonal_MAE', errors['hamiltonian_diagonal_mae'], batch_idx)
        self.writer.add_scalar('Train/Hamiltonian_Non_Diagonal_MAE', errors['hamiltonian_non_diagonal_mae'], batch_idx)
        self.writer.add_scalar('Train/Learning_Rate', lr, batch_idx)

    def log_validation_results(self, epoch, batch_idx, train_errors, val_errors, test_errors):
        log_messages = [
            f"Epoch {epoch} batch_idx {batch_idx} with hamiltonian {train_errors['hamiltonian_mae']:.8f}.",
            f"hamiltonian: diagonal/non diagonal :{train_errors['hamiltonian_diagonal_mae']:.8f}, {train_errors['hamiltonian_non_diagonal_mae']:.8f}.",
            "-------------------------",
            f"best val hamiltonian so far: {self.best_val_result:.8f}.",
            f"current val hamiltonian: {val_errors['hamiltonian_mae']:.8f}",
        ]

        if test_errors:
            log_messages.extend([
                f"test hamiltonian: {test_errors['hamiltonian_mae']:.8f},",
                f"test hamiltonian: diagonal/non diagonal :{test_errors['hamiltonian_diagonal_mae']:.8f}, {test_errors['hamiltonian_non_diagonal_mae']:.8f}.",
            ])

        log_messages.append("=========================")

        for message in log_messages:
            logger.info(message)

        self.writer.add_scalar('Validation/Best_Hamiltonian_MAE', self.best_val_result, batch_idx)
        self.writer.add_scalar('Validation/Current_Hamiltonian_MAE', val_errors['hamiltonian_mae'], batch_idx)

        if test_errors:
            tensorboard_logs = {
                'Test/Hamiltonian_MAE': test_errors['hamiltonian_mae'],
                'Test/Hamiltonian_Diagonal_MAE': test_errors['hamiltonian_diagonal_mae'],
                'Test/Hamiltonian_Non_Diagonal_MAE': test_errors['hamiltonian_non_diagonal_mae']
            }

            for name, value in tensorboard_logs.items():
                self.writer.add_scalar(name, value, batch_idx)