import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os


class TimeSeriesDataset(Dataset):
    def __init__(self, pkl_paths, hist_len=15, pred_len=10, step=16, train=True):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.step = step
        self.train = train

        self.state_indices = [2, 3, 4, 5]
        self.control_indices = [0, 1]
        self.base_indices = [2, 3, 4, 5]
        self.pos_indices = [0, 1]

        self.state_scaler = StandardScaler()
        self.control_scaler = StandardScaler()
        self.base_scaler = StandardScaler()
        self.diff_scaler = StandardScaler()
        self.pos_scaler = StandardScaler()

        all_data = []
        for pkl_path in pkl_paths:
            with open(pkl_path, 'rb') as f:
                trip_id, [real_states, base_states, controls] = pickle.load(f)
            all_data.append({
                'trip_id': trip_id,
                'real_states': real_states,
                'base_states': base_states,
                'controls': controls
            })

        self.trips = all_data
        self.windows = self._create_windows()

        if train:
            self._fit_scalers()

    def _create_windows(self):
        windows = []
        for trip_idx, trip_data in enumerate(self.trips):
            n_samples = len(trip_data['real_states'])
            i = 0
            while i + self.hist_len + self.pred_len <= n_samples:
                windows.append((trip_idx, i))
                i += self.step
        return windows

    def _fit_scalers(self):
        all_states = []
        all_controls = []
        all_bases = []
        all_diffs = []
        all_pos = []

        for trip_data in self.trips:
            states = trip_data['real_states'][:, self.state_indices]
            controls = trip_data['controls'][:, self.control_indices]
            positions = trip_data['real_states'][:, self.pos_indices]
            last_states = states[1:]
            bases = states[:-1]
            diffs = last_states - bases

            all_states.append(states)
            all_controls.append(controls)
            all_bases.append(bases)
            all_diffs.append(diffs)
            all_pos.append(positions)

        all_states = np.vstack(all_states)
        all_controls = np.vstack(all_controls)
        all_bases = np.vstack(all_bases)
        all_diffs = np.vstack(all_diffs)
        all_pos = np.vstack(all_pos)

        self.state_scaler.fit(all_states)
        self.control_scaler.fit(all_controls)
        self.base_scaler.fit(all_bases)
        self.diff_scaler.fit(all_diffs)
        self.pos_scaler.fit(all_pos)

    def _get_raw_data(self, idx):
        trip_idx, window_start = self.windows[idx]
        trip_data = self.trips[trip_idx]
        hist_end = window_start + self.hist_len
        pred_end = hist_end + self.pred_len

        real_states = trip_data['real_states'][window_start:hist_end, :]
        future_states = trip_data['real_states'][hist_end:pred_end, :]
        controls = trip_data['controls'][window_start:hist_end, :]
        
        last_state = real_states[-1, self.state_indices]  # [vlon, vlat, yaw, omega]
        base_states = np.tile(last_state, (self.pred_len, 1))  # 匀速假设：保持最后状态
        
        future_pos = future_states[:, self.pos_indices]  # [x, y] 未来位置

        return real_states, future_states, controls, base_states, future_pos

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        real_states, future_states, controls, base_states, future_pos = self._get_raw_data(idx)

        states = real_states[:, self.state_indices]
        future_states_4 = future_states[:, self.state_indices]
        diffs = future_states_4 - base_states
        bases = base_states
        ctls = controls[:, self.control_indices]

        state_norm = self.state_scaler.transform(states)
        control_norm = self.control_scaler.transform(ctls)
        base_norm = self.base_scaler.transform(bases)
        diff_norm = self.diff_scaler.transform(diffs)
        
        future_pos_tensor = torch.FloatTensor(future_pos)

        hist_state = torch.FloatTensor(state_norm)
        hist_control = torch.FloatTensor(control_norm)
        base_pred = torch.FloatTensor(base_norm)
        gt_pred = torch.FloatTensor(self.state_scaler.transform(future_states_4))
        diff_pred = torch.FloatTensor(diff_norm)

        # 确保base_pred和gt_pred有正确的形状 (pred_len, state_dim)
        if base_pred.dim() == 1:
            base_pred = base_pred.unsqueeze(0)
        if gt_pred.dim() == 1:
            gt_pred = gt_pred.unsqueeze(0)
        if diff_pred.dim() == 1:
            diff_pred = diff_pred.unsqueeze(0)

        cfg_tensor = torch.FloatTensor([self.hist_len, self.pred_len, self.train])

        return hist_state, hist_control, base_pred, cfg_tensor, gt_pred, diff_pred, future_pos_tensor


def load_dataset_from_directory(data_dir, train_ratio=0.8, hist_len=15, pred_len=10, step=16):
    pkl_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')])

    if len(pkl_files) == 0:
        raise ValueError(f"No pkl files found in {data_dir}")

    np.random.seed(42)
    perm = np.random.permutation(len(pkl_files))
    train_size = int(len(pkl_files) * train_ratio)

    train_files = [pkl_files[i] for i in perm[:train_size]]
    val_files = [pkl_files[i] for i in perm[train_size:]]

    train_dataset = TimeSeriesDataset(train_files, hist_len=hist_len, pred_len=pred_len, step=step, train=True)
    val_dataset = TimeSeriesDataset(val_files, hist_len=hist_len, pred_len=pred_len, step=step, train=False)

    val_dataset.state_scaler = train_dataset.state_scaler
    val_dataset.control_scaler = train_dataset.control_scaler
    val_dataset.base_scaler = train_dataset.base_scaler
    val_dataset.diff_scaler = train_dataset.diff_scaler

    return train_dataset, val_dataset, train_files, val_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test TimeSeriesDataset')
    parser.add_argument('--data_dir', type=str, default='./data/diverse_left_turn',
                        help='Directory containing pkl files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/val split ratio')

    args = parser.parse_args()

    train_dataset, val_dataset, train_files, val_files = load_dataset_from_directory(
        args.data_dir, train_ratio=args.train_ratio
    )

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    hist_state, hist_control, base_pred, cfg_tensor, gt_pred, diff_pred = train_dataset[0]
    print(f"\nTrain sample shapes:")
    print(f"  hist_state: {hist_state.shape}")
    print(f"  hist_control: {hist_control.shape}")
    print(f"  base_pred: {base_pred.shape}")
    print(f"  gt_pred: {gt_pred.shape}")
    print(f"  diff_pred: {diff_pred.shape}")
