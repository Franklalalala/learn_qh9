import logging
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from learn_qh9.datasets import CustomizedQH9Stable
from learn_qh9.models.QHNet import QHNet
from learn_qh9.loss import criterion
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from transformers import get_polynomial_decay_schedule_with_warmup

logger = logging.getLogger()

class Trainer:
    def __init__(self, params):
        self.params = params
        self.default_type = torch.float32
        torch.set_default_dtype(self.default_type)
        logger.info(self.params)
        torch.manual_seed(self.params['general']['seed'])

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
                except:
                    logger.warning(f"{e}. Defaulting to 'cpu'.")
                    self.device = torch.device('cpu')
            else:
                logger.warning("CUDA is not available. Defaulting to 'cpu'.")
                self.device = torch.device('cpu')
        else:
            logger.warning(f"Unrecognized device '{input_device}'. Defaulting to 'cpu'.")
            self.device = torch.device('cpu')

    def setup_output_directories(self):
        self.output_dir = self.params['general']['output_dir']
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        self.data_dir = os.path.join(self.output_dir, 'data')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_datasets(self):
        src_lmdb_folder_path = self.params['dataset']['src_lmdb_folder_path']
        logger.info(f"loading source lmdb dataset from {src_lmdb_folder_path}...")
        dataset = CustomizedQH9Stable(src_lmdb_folder_path=src_lmdb_folder_path,
                                      db_workbase=self.data_dir,
                                      split=self.params['dataset']['split'])
        train_dataset = dataset[dataset.train_mask]
        valid_dataset = dataset[dataset.val_mask]
        test_dataset = dataset[dataset.test_mask]

        g = torch.Generator()
        g.manual_seed(self.params['general']['seed'])
        self.train_data_loader = DataLoader(
            train_dataset, batch_size=self.params['training']['train_batch_size'], shuffle=True,
            num_workers=self.params['dataset']['num_workers'], pin_memory=self.params['dataset']['pin_memory'], generator=g)
        self.val_data_loader = DataLoader(
            valid_dataset, batch_size=self.params['validation']['valid_batch_size'], shuffle=False,
            num_workers=self.params['dataset']['num_workers'], pin_memory=self.params['dataset']['pin_memory'])
        self.test_data_loader = DataLoader(
            test_dataset, batch_size=self.params['testing']['test_batch_size'], shuffle=False,
            num_workers=self.params['dataset']['num_workers'], pin_memory=self.params['dataset']['pin_memory'])

    def setup_model(self):
        self.model = QHNet(
            in_node_features=self.params['model']['in_node_features'],
            sh_lmax=self.params['model']['sh_lmax'],
            hidden_size=self.params['model']['hidden_size'],
            bottle_hidden_size=self.params['model']['bottle_hidden_size'],
            num_gnn_layers=self.params['model']['num_gnn_layers'],
            max_radius=self.params['model']['max_radius'],
            num_nodes=self.params['model']['num_nodes'],
            radius_embed_dim=self.params['model']['radius_embed_dim']
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
        self.scheduler = get_polynomial_decay_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.params['training']['warmup_steps'],
            num_training_steps=self.params['training']['total_steps'],
            lr_end=self.params['training']['lr_end'], power=1.0, last_epoch=-1)

    def setup_tensorboard(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):
        self.model.train()
        epoch = 0
        self.best_val_result = float('inf')
        train_iterator = iter(self.train_data_loader)

        for batch_idx in range(self.params['training']['total_steps'] + 10000):
            try:
                batch = next(train_iterator)
                batch = self.post_processing(batch)
            except StopIteration:
                epoch += 1
                train_iterator = iter(self.train_data_loader)
                continue

            batch = batch.to(self.device)
            errors = self.train_one_batch(batch)
            self.scheduler.step()
            if self.params['training']['ema_start_epoch'] > -1 and epoch > self.params['training']['ema_start_epoch']:
                self.ema.update()

            if batch_idx % self.params['training']['train_batch_interval'] == 0:
                self.log_training_progress(epoch, batch_idx, errors)

            if batch_idx % self.params['validation']['validation_batch_interval'] == 0:
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
            val_errors = self.validation_dataset(self.val_data_loader)
            if val_errors['hamiltonian_mae'] < self.best_val_result:
                self.best_val_result = val_errors['hamiltonian_mae']
                test_errors = self.validation_dataset(self.test_data_loader)
                self.save_model("results_best.pt", errors, batch_idx)
            else:
                test_errors = None

        if batch_idx % self.params['validation']['save_interval'] == 0:
            self.save_model(f"results_{batch_idx}.pt", errors, batch_idx)

        self.log_validation_results(epoch, batch_idx, errors, val_errors, test_errors)

    @torch.no_grad()
    def validation_dataset(self, data_loader):
        self.model.eval()
        total_error_dict = {'total_items': 0}
        loss_weights = {'hamiltonian': 1.0}
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
        for key in total_error_dict.keys():
            if key != 'total_items':
                total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']

        return total_error_dict

    def save_model(self, filename, errors, batch_idx):
        save_path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            "state_dict": self.model.state_dict(),
            "eval": errors,
            "batch_idx": batch_idx
        }, save_path)

    def post_processing(self, batch):
        for key in batch.keys:
            if torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
                batch[key] = batch[key].type(self.default_type)
        return batch

    def log_training_progress(self, epoch, batch_idx, errors):
        logger.info(f"Train: Epoch {epoch} {batch_idx} hamiltonian: {errors['hamiltonian_mae']:.8f}.")
        logger.info(f"hamiltonian: diagonal/non diagonal :{errors['hamiltonian_diagonal_mae']:.8f}, "
                    f"{errors['hamiltonian_non_diagonal_mae']:.8f}, lr: {self.optimizer.param_groups[0]['lr']}.")

        # Log to TensorBoard
        self.writer.add_scalar('Train/Hamiltonian_MAE', errors['hamiltonian_mae'], batch_idx)
        self.writer.add_scalar('Train/Hamiltonian_Diagonal_MAE', errors['hamiltonian_diagonal_mae'], batch_idx)
        self.writer.add_scalar('Train/Hamiltonian_Non_Diagonal_MAE', errors['hamiltonian_non_diagonal_mae'], batch_idx)
        self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], batch_idx)

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

        # Log to TensorBoard
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