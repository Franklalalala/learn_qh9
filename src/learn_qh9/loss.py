import torch

from torch_scatter import scatter_sum


def criterion(outputs, target, loss_weights):
    error_dict = {}
    keys = loss_weights.keys()
    try:
        for key in keys:
            row = target.edge_index[0]
            edge_batch = target.batch[row]
            diff_diagonal = outputs[f'{key}_diagonal_blocks'] - target[f'diagonal_{key}']
            mse_diagonal = torch.sum(diff_diagonal ** 2 * target[f"diagonal_{key}_mask"], dim=[1, 2])
            mae_diagonal = torch.sum(torch.abs(diff_diagonal) * target[f"diagonal_{key}_mask"], dim=[1, 2])
            count_sum_diagonal = torch.sum(target[f"diagonal_{key}_mask"], dim=[1, 2])
            mse_diagonal = scatter_sum(mse_diagonal, target.batch)
            mae_diagonal = scatter_sum(mae_diagonal, target.batch)
            count_sum_diagonal = scatter_sum(count_sum_diagonal, target.batch)

            diff_non_diagonal = outputs[f'{key}_non_diagonal_blocks'] - target[f'non_diagonal_{key}']
            mse_non_diagonal = torch.sum(diff_non_diagonal ** 2 * target[f"non_diagonal_{key}_mask"], dim=[1, 2])
            mae_non_diagonal = torch.sum(torch.abs(diff_non_diagonal) * target[f"non_diagonal_{key}_mask"], dim=[1, 2])
            count_sum_non_diagonal = torch.sum(target[f"non_diagonal_{key}_mask"], dim=[1, 2])
            mse_non_diagonal = scatter_sum(mse_non_diagonal, edge_batch)
            mae_non_diagonal = scatter_sum(mae_non_diagonal, edge_batch)
            count_sum_non_diagonal = scatter_sum(count_sum_non_diagonal, edge_batch)

            mae = ((mae_diagonal + mae_non_diagonal) / (count_sum_diagonal + count_sum_non_diagonal)).mean()
            mse = ((mse_diagonal + mse_non_diagonal) / (count_sum_diagonal + count_sum_non_diagonal)).mean()

            error_dict[key + '_mae'] = mae
            error_dict[key + '_rmse'] = torch.sqrt(mse)
            error_dict[key + '_diagonal_mae'] = (mae_diagonal / count_sum_diagonal).mean()
            error_dict[key + '_non_diagonal_mae'] = (mae_non_diagonal / count_sum_non_diagonal).mean()
            loss = mse + mae
            error_dict[key] = loss
            if 'loss' in error_dict.keys():
                error_dict['loss'] = error_dict['loss'] + loss_weights[key] * loss
            else:
                error_dict['loss'] = loss_weights[key] * loss
    except Exception as exc:
        raise exc
    return error_dict

