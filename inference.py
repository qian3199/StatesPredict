import torch
import numpy as np
import logging
import os


class InferModel:
    def __init__(self, model, state_scaler, diff_scaler, control_scaler, base_scaler, phy_model,
                 visualizer=None, log_dir='./logs', device=None):
        self.model = model
        self.state_scaler = state_scaler
        self.diff_scaler = diff_scaler
        self.control_scaler = control_scaler
        self.base_scaler = base_scaler
        self.phy_model = phy_model
        self.visualizer = visualizer
        self.model.eval()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'inference.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def normalize_state(self, state):
        if isinstance(state, torch.Tensor):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = state

        if state_np.ndim == 2:
            state_norm = self.state_scaler.transform(state_np)
        elif state_np.ndim == 3:
            batch_size, seq_len, dim = state_np.shape
            state_norm_flat = state_np.reshape(-1, dim)
            state_norm_flat = self.state_scaler.transform(state_norm_flat)
            state_norm = state_norm_flat.reshape(batch_size, seq_len, dim)
        else:
            state_norm = self.state_scaler.transform(state_np.reshape(1, -1)).flatten()

        if isinstance(state, torch.Tensor):
            return torch.FloatTensor(state_norm).to(state.device)
        return state_norm

    def denormalize_state(self, state_norm):
        if isinstance(state_norm, torch.Tensor):
            state_norm_np = state_norm.detach().cpu().numpy()
        else:
            state_norm_np = state_norm

        if state_norm_np.ndim == 2:
            state_real = self.state_scaler.inverse_transform(state_norm_np)
        elif state_norm_np.ndim == 3:
            batch_size, seq_len, dim = state_norm_np.shape
            state_norm_flat = state_norm_np.reshape(-1, dim)
            state_real_flat = self.state_scaler.inverse_transform(state_norm_flat)
            state_real = state_real_flat.reshape(batch_size, seq_len, dim)
        else:
            state_real = self.state_scaler.inverse_transform(state_norm_np.reshape(1, -1)).flatten()

        if isinstance(state_norm, torch.Tensor):
            return torch.FloatTensor(state_real).to(state_norm.device)
        return state_real

    def normalize_control(self, control):
        if isinstance(control, torch.Tensor):
            control_np = control.detach().cpu().numpy()
        else:
            control_np = control

        if control_np.ndim == 2:
            control_norm = self.control_scaler.transform(control_np)
        elif control_np.ndim == 3:
            batch_size, seq_len, dim = control_np.shape
            control_norm_flat = control_np.reshape(-1, dim)
            control_norm_flat = self.control_scaler.transform(control_norm_flat)
            control_norm = control_norm_flat.reshape(batch_size, seq_len, dim)
        else:
            control_norm = self.control_scaler.transform(control_np.reshape(1, -1)).flatten()

        if isinstance(control, torch.Tensor):
            return torch.FloatTensor(control_norm).to(control.device)
        return control_norm

    def predict_net_single_step(self, hist_state, hist_control, base_state, cfg_tensor):
        hist_state_norm = self.normalize_state(hist_state)
        hist_control_norm = self.normalize_control(hist_control)
        base_state_norm = self.normalize_state(base_state)

        if isinstance(hist_state_norm, np.ndarray):
            hist_state_norm = torch.FloatTensor(hist_state_norm)
        if isinstance(hist_control_norm, np.ndarray):
            hist_control_norm = torch.FloatTensor(hist_control_norm)
        if isinstance(base_state_norm, np.ndarray):
            base_state_norm = torch.FloatTensor(base_state_norm)

        if hist_state_norm.ndim == 2:
            hist_state_norm = hist_state_norm.unsqueeze(0)
            hist_control_norm = hist_control_norm.unsqueeze(0)
            base_state_norm = base_state_norm.unsqueeze(0)
            cfg_tensor = cfg_tensor.unsqueeze(0)

        with torch.no_grad():
            delta_norm, corrected_norm = self.model(
                hist_state_norm, hist_control_norm, base_state_norm, cfg_tensor
            )

        delta_real = self.denormalize_state(delta_norm)
        corrected_real = self.denormalize_state(corrected_norm)
        base_real = self.denormalize_state(base_state_norm)

        corrected_real = corrected_real[0] if corrected_real.ndim >= 2 else corrected_real
        delta_real = delta_real[0] if delta_real.ndim >= 2 else delta_real
        base_real = base_real[0] if base_real.ndim >= 2 else base_real

        return corrected_real, delta_real, base_real

    def autoregressive_predict(self, initial_state, controls, pred_len=100, dt=0.01, inference_mask=[0, 1, 0, 0]):
        current_state = initial_state.copy()
        hybrid_states = []
        physics_states = []

        mask = np.array(inference_mask)

        for t in range(pred_len):
            control = controls[t] if t < len(controls) else controls[-1]

            physics_next = self.phy_model.simulate(current_state, control)
            physics_states.append(physics_next.copy())

            hist_state = np.array([current_state])
            hist_control = np.array([control])
            base_state = np.array([physics_next])
            cfg_tensor = torch.FloatTensor([1, 1, 0])

            corrected, delta, base = self.predict_net_single_step(hist_state, hist_control, base_state, cfg_tensor)
            corrected = corrected.numpy() if isinstance(corrected, torch.Tensor) else corrected
            if corrected.ndim == 2:
                corrected = corrected[0] if corrected.shape[0] == 1 else corrected[-1]

            hybrid_pred = mask * corrected + (1 - mask) * physics_next

            hybrid_states.append(hybrid_pred.copy())
            current_state = hybrid_pred

        hybrid_states = np.array(hybrid_states)
        physics_states = np.array(physics_states)

        hybrid_x, hybrid_y = self.integrate_trajectory(hybrid_states, dt)
        physics_x, physics_y = self.integrate_trajectory(physics_states, dt)

        return {
            'hybrid_states': hybrid_states,
            'physics_states': physics_states,
            'hybrid_x': hybrid_x,
            'hybrid_y': hybrid_y,
            'physics_x': physics_x,
            'physics_y': physics_y
        }

    def autoregressive_predict_ind(self, initial_state, controls, pred_len=100, dt=0.01, inference_mask=[0, 1, 0, 0]):
        current_state = initial_state.copy()
        physics_full = []

        for t in range(pred_len):
            control = controls[t] if t < len(controls) else controls[-1]
            current_state = self.phy_model.simulate(current_state, control)
            physics_full.append(current_state.copy())

        physics_full = np.array(physics_full)

        hybrid_states = []
        mask = np.array(inference_mask)

        for t in range(pred_len):
            hist_state = np.array([physics_full[t-1] if t > 0 else initial_state])
            control = controls[t] if t < len(controls) else controls[-1]
            hist_control = np.array([control])
            base_state = np.array([physics_full[t]])
            cfg_tensor = torch.FloatTensor([1, 1, 0])

            corrected, delta, base = self.predict_net_single_step(hist_state, hist_control, base_state, cfg_tensor)
            corrected = corrected.numpy() if isinstance(corrected, torch.Tensor) else corrected
            delta = delta.numpy() if isinstance(delta, torch.Tensor) else delta
            if corrected.ndim == 2:
                corrected = corrected[0] if corrected.shape[0] == 1 else corrected[-1]
            if delta.ndim == 2:
                delta = delta[0] if delta.shape[0] == 1 else delta[-1]

            hybrid_pred = physics_full[t] + mask * delta

            hybrid_states.append(hybrid_pred.copy())

        hybrid_states = np.array(hybrid_states)

        hybrid_x, hybrid_y = self.integrate_trajectory(hybrid_states, dt)
        physics_x, physics_y = self.integrate_trajectory(physics_full, dt)

        return {
            'hybrid_states': hybrid_states,
            'physics_states': physics_full,
            'hybrid_x': hybrid_x,
            'hybrid_y': hybrid_y,
            'physics_x': physics_x,
            'physics_y': physics_y
        }

    def integrate_trajectory(self, states, dt=0.01):
        n = len(states)
        x = np.zeros(n)
        y = np.zeros(n)

        for i in range(1, n):
            vlon = states[i-1, 0]
            vlat = states[i-1, 1]
            yaw = states[i-1, 2]

            dx = (vlon * np.cos(yaw) - vlat * np.sin(yaw)) * dt
            dy = (vlon * np.sin(yaw) + vlat * np.cos(yaw)) * dt

            x[i] = x[i-1] + dx
            y[i] = y[i-1] + dy

        return x, y

    def compute_lateral_error(self, pred_x, pred_y, gt_x, gt_y, gt_yaw):
        n_x = -np.sin(gt_yaw)
        n_y = np.cos(gt_yaw)

        dx = pred_x - gt_x
        dy = pred_y - gt_y

        lateral_error = dx * n_x + dy * n_y
        return lateral_error

    def evaluate(self, pred_results, gt_xy, gt_yaw=None):
        physics_x = pred_results['physics_x']
        physics_y = pred_results['physics_y']
        hybrid_x = pred_results['hybrid_x']
        hybrid_y = pred_results['hybrid_y']

        gt_x = gt_xy[:, 0]
        gt_y = gt_xy[:, 1]

        physics_lateral = self.compute_lateral_error(physics_x, physics_y, gt_x, gt_y, gt_yaw)
        hybrid_lateral = self.compute_lateral_error(hybrid_x, hybrid_y, gt_x, gt_y, gt_yaw)

        physics_rmse = np.sqrt(np.mean((physics_x - gt_x)**2 + (physics_y - gt_y)**2))
        hybrid_rmse = np.sqrt(np.mean((hybrid_x - gt_x)**2 + (hybrid_y - gt_y)**2))

        physics_mae = np.mean(np.sqrt((physics_x - gt_x)**2 + (physics_y - gt_y)**2))
        hybrid_mae = np.mean(np.sqrt((hybrid_x - gt_x)**2 + (hybrid_y - gt_y)**2))

        physics_max = np.max(np.sqrt((physics_x - gt_x)**2 + (physics_y - gt_y)**2))
        hybrid_max = np.max(np.sqrt((hybrid_x - gt_x)**2 + (hybrid_y - gt_y)**2))

        physics_lateral_rmse = np.sqrt(np.mean(physics_lateral**2))
        hybrid_lateral_rmse = np.sqrt(np.mean(hybrid_lateral**2))

        physics_lateral_mae = np.mean(np.abs(physics_lateral))
        hybrid_lateral_mae = np.mean(np.abs(hybrid_lateral))

        physics_lateral_max = np.max(np.abs(physics_lateral))
        hybrid_lateral_max = np.max(np.abs(hybrid_lateral))

        return {
            'physics_rmse': physics_rmse,
            'hybrid_rmse': hybrid_rmse,
            'physics_mae': physics_mae,
            'hybrid_mae': hybrid_mae,
            'physics_max': physics_max,
            'hybrid_max': hybrid_max,
            'physics_lateral_rmse': physics_lateral_rmse,
            'hybrid_lateral_rmse': hybrid_lateral_rmse,
            'physics_lateral_mae': physics_lateral_mae,
            'hybrid_lateral_mae': hybrid_lateral_mae,
            'physics_lateral_max': physics_lateral_max,
            'hybrid_lateral_max': hybrid_lateral_max,
        }

    def print_evaluation_table(self, eval_results, mode_name="Interactive"):
        header = f"\n{'='*70}"
        header += f"\n{mode_name} Mode - Lateral Error Evaluation Results"
        header += f"\n{'='*70}"

        table_header = f"\n{'Metric':<25} {'Physics Baseline':>18} {'Hybrid Model':>18} {'Improvement %':>15}"
        table_divider = f"\n{'-'*70}"

        rows = []
        metrics = [
            ('Lateral RMSE (m)', 'physics_lateral_rmse', 'hybrid_lateral_rmse'),
            ('Lateral MAE (m)', 'physics_lateral_mae', 'hybrid_lateral_mae'),
            ('Lateral Max Error (m)', 'physics_lateral_max', 'hybrid_lateral_max'),
            ('XY RMSE (m)', 'physics_rmse', 'hybrid_rmse'),
            ('XY MAE (m)', 'physics_mae', 'hybrid_mae'),
            ('XY Max Error (m)', 'physics_max', 'hybrid_max'),
        ]

        for metric_name, phys_key, hybr_key in metrics:
            phys_val = eval_results[phys_key]
            hybr_val = eval_results[hybr_key]
            improvement = ((phys_val - hybr_val) / phys_val) * 100 if phys_val > 0 else 0

            if improvement > 0:
                improvement_str = f"+{improvement:.2f}%"
            else:
                improvement_str = f"{improvement:.2f}%"

            rows.append(f"{metric_name:<25} {phys_val:>18.4f} {hybr_val:>18.4f} {improvement_str:>15}")

        footer = f"\n{'='*70}"
        footer += "\n* Positive improvement (%) indicates hybrid model outperforms physics baseline"
        footer += f"\n{'='*70}\n"

        self.logger.info(header)
        self.logger.info(table_header)
        self.logger.info(table_divider)
        for row in rows:
            self.logger.info(row)
        self.logger.info(footer)

    def run_full_evaluation(self, initial_state, controls, gt_xy, gt_yaw, pred_len=100,
                             dt=0.01, save_prefix='evaluation'):
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"# Running Full Evaluation - pred_len={pred_len}")
        self.logger.info(f"{'#'*70}\n")

        result_interactive = self.autoregressive_predict(initial_state, controls, pred_len, dt)
        result_additive = self.autoregressive_predict_ind(initial_state, controls, pred_len, dt)

        if self.visualizer is not None:
            gt_xy_np = gt_xy if isinstance(gt_xy, np.ndarray) else gt_xy.cpu().numpy()
            physics_xy = np.column_stack([result_interactive['physics_x'], result_interactive['physics_y']])
            hybrid_interactive_xy = np.column_stack([result_interactive['hybrid_x'], result_interactive['hybrid_y']])
            hybrid_additive_xy = np.column_stack([result_additive['hybrid_x'], result_additive['hybrid_y']])

            save_path = f'{save_prefix}_4trajectories.png'
            self.visualizer.plot_4trajectories(gt_xy_np, physics_xy, hybrid_interactive_xy, hybrid_additive_xy, save_path)
            self.logger.info(f"Trajectory comparison plot saved to {os.path.join(self.visualizer.save_dir, save_path)}")

        self.logger.info("\n--- Interactive (Closed-Loop) Mode Evaluation ---")
        eval_interactive = self.evaluate(result_interactive, gt_xy, gt_yaw)
        self.print_evaluation_table(eval_interactive, "Interactive (Closed-Loop)")

        self.logger.info("\n--- Additive (Open-Loop) Mode Evaluation ---")
        eval_additive = self.evaluate(result_additive, gt_xy, gt_yaw)
        self.print_evaluation_table(eval_additive, "Additive (Open-Loop)")

        return {
            'interactive': eval_interactive,
            'additive': eval_additive,
            'interactive_result': result_interactive,
            'additive_result': result_additive
        }


class SimpleBicycleModel:
    def __init__(self, m=2273.9, Iz=3057.6, lf=1.3535, lr=1.4015, Caf=54920, Car=62190, ratio=15.6):
        self.m = m
        self.Iz = Iz
        self.lf = lf
        self.lr = lr
        self.Caf = Caf
        self.Car = Car
        self.ratio = ratio

    def simulate(self, state, control, dt=0.01):
        vlon, vlat, yaw, omega = state
        acc, steering_angle = control

        delta = steering_angle / self.ratio

        alpha_f = delta - np.arctan2(vlat + self.lf * omega, vlon) if vlon != 0 else delta
        alpha_r = -np.arctan2(vlat - self.lr * omega, vlon) if vlon != 0 else 0

        Fyf = -self.Caf * alpha_f
        Fyr = -self.Car * alpha_r

        dvx_dt = acc + (vlat * omega)
        dvy_dt = (-vlon * omega) + (Fyf + Fyr) / self.m
        domega_dt = (self.lf * Fyf - self.lr * Fyr) / self.Iz

        vlon_new = vlon + dvx_dt * dt
        vlat_new = vlat + dvy_dt * dt
        yaw_new = yaw + omega * dt
        omega_new = omega + domega_dt * dt

        return np.array([vlon_new, vlat_new, yaw_new, omega_new])


if __name__ == "__main__":
    from model import DyTR_LSTM
    from dataset import TimeSeriesDataset
    from visualizer import VisualUtils

    model = DyTR_LSTM()
    dataset = TimeSeriesDataset(['processed_trip.pkl'])
    visualizer = VisualUtils(state_names=['vlon', 'vlat', 'yaw', 'omega'], save_dir='./plots')

    phy_model = SimpleBicycleModel()

    infer_model = InferModel(
        model,
        dataset.state_scaler,
        dataset.diff_scaler,
        dataset.control_scaler,
        dataset.base_scaler,
        phy_model,
        visualizer=visualizer
    )

    np.random.seed(42)
    initial_state = np.array([20.0, 0.5, 0.1, 0.05])
    controls = np.random.randn(50, 2)

    gt_yaw = np.linspace(0.1, 0.5, 50)
    gt_x = np.cumsum(np.ones(50) * 0.2)
    gt_y = np.cumsum(np.sin(np.linspace(0, 3, 50)) * 0.1)
    gt_xy = np.column_stack([gt_x, gt_y])

    results = infer_model.run_full_evaluation(initial_state, controls, gt_xy, gt_yaw, pred_len=50)
