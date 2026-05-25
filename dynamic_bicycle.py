import numpy as np

class State(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

class ControlInput(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

class LateralDynamicBicycle:
    """
    运动学自行车模型 (Kinematic Bicycle Model)
    状态: [x, y, vx, vy, psi, r]
    控制: [a, delta]   (a: 纵向加速度, delta: 前轮转角)
    """
    States = ['x', 'y', 'vx', 'vy', 'psi', 'r']
    ControlInputs = ['a', 'delta']

    def __init__(self, parameters: dict):
        self.wheelbase = parameters['wheelbase']
        # 其余参数为兼容性保留，运动学模型不使用
        self.l_f = parameters.get('l_f', self.wheelbase/2)
        self.l_r = parameters.get('l_r', self.wheelbase/2)
        self.steering_gear_ratio = parameters.get('steering_gear_ratio', 1.0)

    def State(self, data):
        return State(data)

    def ControlInput(self, data):
        return ControlInput(data)

    def _dynamics(self, state, control):
        """运动学模型导数，用于 RK4 积分"""
        x, y, vx, vy, psi, r = state
        a, delta = control
        # 运动学关系
        if abs(vx) < 0.01:
            vx = 0.01
        # 横摆角速度由前轮转角和速度决定
        r_desired = vx * np.tan(delta) / self.wheelbase
        # 令角加速度与期望值成正比（快速响应），避免直接强制等于期望值导致不稳定
        # 采用简单的比例控制，使 r 快速跟踪 r_desired，时间常数 0.05s
        tau = 0.05
        r_dot = (r_desired - r) / tau
        # 纵向加速度
        vx_dot = a
        # 侧向速度变化（运动学假设无侧滑，vy 保持很小）
        vy_dot = 0.0
        # 位置更新
        x_dot = vx * np.cos(psi) - vy * np.sin(psi)
        y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = r
        return np.array([x_dot, y_dot, vx_dot, vy_dot, psi_dot, r_dot])

    def simulate(self, initial_state, control_input, n=1, dt=0.01):
        """
        使用 4 阶 Runge-Kutta 积分 n 步
        initial_state: (6,) 或 (6,1)
        control_input: (2,) 或 (2,1)
        return: traj (6, n+1), success
        """
        if initial_state.ndim == 1:
            init = initial_state.reshape(-1, 1)
        else:
            init = initial_state
        if control_input.ndim == 1:
            ctrl = control_input.reshape(-1, 1)
        else:
            ctrl = control_input

        traj = np.zeros((6, n+1))
        traj[:, 0] = init.flatten()

        for step in range(n):
            cur = traj[:, step]
            # 控制量在仿真期间保持不变（假设零阶保持）
            a = ctrl[0, 0] if ctrl.shape[1] == 1 else ctrl[0, step]
            delta = ctrl[1, 0] if ctrl.shape[1] == 1 else ctrl[1, step]
            control = np.array([a, delta])

            k1 = self._dynamics(cur, control)
            k2 = self._dynamics(cur + 0.5*dt*k1, control)
            k3 = self._dynamics(cur + 0.5*dt*k2, control)
            k4 = self._dynamics(cur + dt*k3, control)
            next_state = cur + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            traj[:, step+1] = next_state

        return traj, True