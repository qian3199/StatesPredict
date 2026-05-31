import torch
import numpy as np
import glob
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import DyTR_LSTM
from dataset import TimeSeriesDataset
from trainer import Trainer
from visualizer import VisualUtils
from inference import InferModel, SimpleBicycleModel

def run_inference(model_path=None, data_dir='./data/left_turn_dataset', output_dir='./inference_results'):
    if model_path is None:
        model_path = 'left_turn_model.pth'

    print(f"Loading model from {model_path}...")

    data_dir = './data/left_turn_dataset'
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))

    print(f"Found {len(pkl_files)} files")

    n_files = min(50, len(pkl_files))
    test_files = pkl_files[n_train:n_files] if 'n_train' in dir() else pkl_files[n_files//2:n_files]

    if len(test_files) == 0:
        test_files = pkl_files[:min(10, len(pkl_files))]

    print(f"Using {len(test_files)} files for inference")

    hist_len = 15
    pred_len = 1
    step = 1

    train_files = pkl_files[:n_files//2] if len(pkl_files) >= n_files else pkl_files[:int(len(pkl_files)*0.8)]
    train_dataset = TimeSeriesDataset(train_files, hist_len=hist_len, pred_len=pred_len, step=step, train=True)

    test_dataset = TimeSeriesDataset(test_files, hist_len=hist_len, pred_len=pred_len, step=step, train=False)
    test_dataset.state_scaler = train_dataset.state_scaler
    test_dataset.control_scaler = train_dataset.control_scaler
    test_dataset.base_scaler = train_dataset.base_scaler
    test_dataset.diff_scaler = train_dataset.diff_scaler

    print(f"Test windows: {len(test_dataset.windows)}")

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DyTR_LSTM().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using untrained model")

    os.makedirs(output_dir, exist_ok=True)

    visualizer = VisualUtils(
        state_names=['vlon', 'vlat', 'yaw', 'omega'],
        save_dir=output_dir
    )

    phy_model = SimpleBicycleModel()

    infer_model = InferModel(
        model,
        train_dataset.state_scaler,
        train_dataset.diff_scaler,
        train_dataset.control_scaler,
        train_dataset.base_scaler,
        phy_model,
        visualizer=visualizer,
        log_dir=output_dir,
        device=device
    )

    print("\nRunning inference on test data...")
    print("="*80)

    all_predictions = []
    all_base_predictions = []
    all_ground_truth = []

    def denormalize_diff(diff_norm, diff_scaler, device):
        mean = torch.FloatTensor(diff_scaler.mean_).to(device)
        scale = torch.FloatTensor(diff_scaler.scale_).to(device)

        if diff_norm.ndim == 3:
            return diff_norm * scale.unsqueeze(0).unsqueeze(0) + mean.unsqueeze(0).unsqueeze(0)
        else:
            return diff_norm * scale + mean

    model.eval()
    with torch.no_grad():
        for batch_idx, (hist_state, hist_control, base_norm, cfg_tensor, gt_norm, diff_norm, future_pos) in enumerate(test_loader):
            hist_state = hist_state.to(device)
            hist_control = hist_control.to(device)
            base_norm = base_norm.to(device)
            cfg_tensor = cfg_tensor.to(device)
            gt_norm = gt_norm.to(device)

            delta_norm, corrected_norm = model(hist_state, hist_control, base_norm, cfg_tensor)

            base_real = infer_model.denormalize_state(base_norm)
            gt_real = infer_model.denormalize_state(gt_norm)
            net_pred = base_real + denormalize_diff(delta_norm, train_dataset.diff_scaler, device)

            all_predictions.append(net_pred.cpu().numpy())
            all_base_predictions.append(base_real.cpu().numpy())
            all_ground_truth.append(gt_real.cpu().numpy())

            base_rmse = np.sqrt(np.mean((base_real.cpu().numpy() - gt_real.cpu().numpy())**2))
            net_rmse = np.sqrt(np.mean((net_pred.cpu().numpy() - gt_real.cpu().numpy())**2))

            print(f"Batch {batch_idx}: Base RMSE: {base_rmse:.8f}, Net RMSE: {net_rmse:.8f}, Improvement: {base_rmse - net_rmse:.8f}")

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_base_predictions = np.concatenate(all_base_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)

    overall_base_rmse = np.sqrt(np.mean((all_base_predictions - all_ground_truth)**2))
    overall_net_rmse = np.sqrt(np.mean((all_predictions - all_ground_truth)**2))

    print("\n" + "="*80)
    print("OVERALL INFERENCE RESULTS:")
    print("="*80)
    print(f"Base Predictions - Mean RMSE: {overall_base_rmse:.8f}")
    print(f"Net Predictions  - Mean RMSE: {overall_net_rmse:.8f}")
    print(f"Improvement: {overall_base_rmse - overall_net_rmse:.8f} ({(overall_base_rmse - overall_net_rmse)/overall_base_rmse*100:.4f}%)")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    num_samples = min(100, len(all_predictions))

    ax1 = axes[0, 0]
    sample_indices = np.arange(num_samples)
    ax1.plot(sample_indices, overall_base_rmse * np.ones(num_samples), 'b--', alpha=0.5, label=f'Base RMSE: {overall_base_rmse:.6f}')
    ax1.plot(sample_indices, overall_net_rmse * np.ones(num_samples), 'r--', alpha=0.5, label=f'Net RMSE: {overall_net_rmse:.6f}')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Overall RMSE Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(all_base_predictions.flatten(), bins=50, alpha=0.5, label='Base Predictions', color='blue')
    ax2.hist(all_predictions.flatten(), bins=50, alpha=0.5, label='Net Predictions', color='red')
    ax2.hist(all_ground_truth.flatten(), bins=50, alpha=0.5, label='Ground Truth', color='green')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    state_names = ['vlon', 'vlat', 'yaw', 'omega']
    for i, state_name in enumerate(state_names):
        ax = axes[1, i] if i < 2 else None
        if ax is not None:
            ax.plot(all_ground_truth[:num_samples, 0, i], 'g-', alpha=0.7, label='Ground Truth', linewidth=2)
            ax.plot(all_base_predictions[:num_samples, 0, i], 'b--', alpha=0.5, label='Base', linewidth=1)
            ax.plot(all_predictions[:num_samples, 0, i], 'r--', alpha=0.5, label='Net', linewidth=1)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(state_name)
            ax.set_title(f'{state_name.upper()} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'inference_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nInference comparison plot saved to: {save_path}")
    print(f"\nAll results saved to: {output_dir}")

    results_file = os.path.join(output_dir, 'inference_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("INFERENCE RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test files: {len(test_files)}\n")
        f.write(f"Test windows: {len(test_dataset.windows)}\n\n")
        f.write(f"Base Predictions - Mean RMSE: {overall_base_rmse:.8f}\n")
        f.write(f"Net Predictions  - Mean RMSE: {overall_net_rmse:.8f}\n")
        f.write(f"Improvement: {overall_base_rmse - overall_net_rmse:.8f} ({(overall_base_rmse - overall_net_rmse)/overall_base_rmse*100:.4f}%)\n")
        f.write("="*80 + "\n")

    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DyTR-LSTM 推理脚本')
    parser.add_argument('--model', type=str, default='left_turn_model.pth', help='模型路径 (默认: left_turn_model.pth)')
    parser.add_argument('--data_dir', type=str, default='./data/left_turn_dataset', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='输出目录')
    args = parser.parse_args()

    run_inference(model_path=args.model, data_dir=args.data_dir, output_dir=args.output_dir)