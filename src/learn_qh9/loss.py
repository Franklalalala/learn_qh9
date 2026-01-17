# learn_qh9/loss.py (或者代码块中的 criterion 部分)

from torch_scatter import scatter_sum
import torch
import logging

# 获取 logger 实例，如果没有定义则使用默认的
logger = logging.getLogger()


def criterion(outputs, target, loss_weights):
    error_dict = {}
    keys = loss_weights.keys()

    # ---------------- DEBUG INFO START ----------------
    # 检查网络输出本身是否含有 nan/inf
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                logger.error(f"[CRITICAL] Model output '{k}' contains NaN or Inf!")
                # 可以选择在这里 raise ValueError 或者打印更多信息
    # ---------------- DEBUG INFO END ------------------

    for key in keys:
        # --- diagonal: per node ---
        diff_diagonal = outputs[f'{key}_diagonal_blocks'] - target[f'diagonal_{key}']

        # 计算平方误差和绝对误差
        mse_diagonal_elem = diff_diagonal ** 2 * target[f"diagonal_{key}_mask"]
        mae_diagonal_elem = torch.abs(diff_diagonal) * target[f"diagonal_{key}_mask"]

        mse_diagonal = torch.sum(mse_diagonal_elem, dim=[1, 2])
        mae_diagonal = torch.sum(mae_diagonal_elem, dim=[1, 2])
        count_sum_diagonal = torch.sum(target[f"diagonal_{key}_mask"], dim=[1, 2])

        mse_diagonal = scatter_sum(mse_diagonal, target.batch)
        mae_diagonal = scatter_sum(mae_diagonal, target.batch)
        count_sum_diagonal = scatter_sum(count_sum_diagonal, target.batch)

        # --- non-diagonal: per full edge ---
        diff_non_diagonal = outputs[f'{key}_non_diagonal_blocks'] - target[f'non_diagonal_{key}']

        mse_non_diagonal_elem = diff_non_diagonal ** 2 * target[f"non_diagonal_{key}_mask"]
        mae_non_diagonal_elem = torch.abs(diff_non_diagonal) * target[f"non_diagonal_{key}_mask"]

        mse_non_diagonal = torch.sum(mse_non_diagonal_elem, dim=[1, 2])
        mae_non_diagonal = torch.sum(mae_non_diagonal_elem, dim=[1, 2])
        count_sum_non_diagonal = torch.sum(target[f"non_diagonal_{key}_mask"], dim=[1, 2])

        # 处理 Edge Batch
        if hasattr(target, "full_edge_index") and target.full_edge_index is not None:
            row_full = target.full_edge_index[0]
        elif hasattr(target, "edge_index_full") and target.edge_index_full is not None:
            row_full = target.edge_index_full[0]
        else:
            row_full = target.edge_index[0]

        edge_batch = target.batch[row_full]

        mse_non_diagonal = scatter_sum(mse_non_diagonal, edge_batch)
        mae_non_diagonal = scatter_sum(mae_non_diagonal, edge_batch)
        count_sum_non_diagonal = scatter_sum(count_sum_non_diagonal, edge_batch)

        # ---------------- FIX & DEBUG START ----------------
        total_count = count_sum_diagonal + count_sum_non_diagonal

        # 检查分母是否为 0
        if (total_count == 0).any():
            logger.error(f"[ERROR] Found graph with 0 valid matrix entries in batch! This causes Inf.")
            logger.error(f"Diag counts: {count_sum_diagonal}")
            logger.error(f"Non-Diag counts: {count_sum_non_diagonal}")
            # 强制将 0 变为 1 防止报错，虽然这代表数据有问题
            total_count = torch.clamp(total_count, min=1.0)

        epsilon = 1e-8  # 防止除以极小值

        mae = ((mae_diagonal + mae_non_diagonal) / (total_count + epsilon)).mean()
        mse = ((mse_diagonal + mse_non_diagonal) / (total_count + epsilon)).mean()

        # 分别计算时也要加 epsilon
        diag_mae_val = (mae_diagonal / (count_sum_diagonal + epsilon)).mean()
        non_diag_mae_val = (mae_non_diagonal / (count_sum_non_diagonal + epsilon)).mean()
        # ---------------- FIX & DEBUG END ------------------

        error_dict[key + '_mae'] = mae
        error_dict[key + '_rmse'] = torch.sqrt(mse)
        error_dict[key + '_diagonal_mae'] = diag_mae_val
        error_dict[key + '_non_diagonal_mae'] = non_diag_mae_val

        loss = mse + mae
        error_dict[key] = loss
        error_dict['loss'] = error_dict.get('loss', 0.0) + loss_weights[key] * loss

        # 检查 Loss 是否炸了
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"[CRITICAL] Loss is {loss}! MAE: {mae}, MSE: {mse}")

    return error_dict