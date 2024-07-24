import torch 
import torch.nn as nn 
import dgl
import dgl.function as fn
from einops import repeat

from ddefunc import OdeintAdjointMethod, AdjointFunc
from ddeutil import _check_inputs, _flat_to_shape


# dx/dt for forward process 
class DDEFunc(nn.Module):
    def __init__(self, in_dim, hid_dim, step_size):
        super(DDEFunc, self).__init__()
        self.d = nn.Parameter(torch.ones(hid_dim))
        self.w = nn.Parameter(torch.eye(hid_dim))
        self.t = None 
        self.step_size = step_size
        self.auto_regress_length = 5
        self.wy = nn.Parameter(torch.FloatTensor(hid_dim, hid_dim))
        self.wc = nn.Parameter(torch.FloatTensor(in_dim, hid_dim))
        self.trans_y = nn.Sequential(nn.Linear(hid_dim, hid_dim), 
                                    #  nn.ReLU(), 
                                    #  nn.Linear(hid_dim, hid_dim)
                                     )
        self.trans_control = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                    #  nn.ReLU(),
                                    #  nn.Linear(hid_dim, hid_dim)
                                    )
        self.auto_regress = nn.Parameter(torch.FloatTensor(self.auto_regress_length))
        self.gate = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())
        self.gate_out = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())
        self.norm = nn.BatchNorm1d(hid_dim)
        self.reset_param()
    
    def reset_param(self):
        nn.init.kaiming_normal_(self.wc)
        nn.init.kaiming_normal_(self.wy)
        nn.init.normal_(self.auto_regress)
    
    def message(self, edges):

        delay = edges.data['delay']
        catch = (self.t + delay)/self.step_size
        catch[catch < 0] = 0    # if history state doesn't exist, use the initial state
        hist = edges.src['state']
        catch = catch.to(torch.long)
        choose_state = hist[range(hist.shape[0]), catch]
        weight = edges.data['w'].reshape(-1, 1, 1)
        return {'m': weight * choose_state}

    def forward(self, g, x, funcx, t):
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))

        # use the former state as a temporary 
        g.ndata['state'][:, int(t/self.step_size)] = x 
        self.t = t 
        g.update_all(self.message, fn.sum('m', 's'))
        y = g.ndata['s']
        # 后做线性变换可以大大提升计算效率
        y = torch.einsum('ijk, kl->ijl', y, w)

        # # autoregression with history
        # hist = g.ndata['state'][:, int(t/self.step_size)-1].clone()

        gate = self.gate(y)
        y = (1 - gate) * (y - x)

        # batchnorm
        # y = self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)

        if funcx is not None:
            dx_dt = funcx(t).permute(1, 0, 2)
            dx_dt = self.trans_control(dx_dt)
            # dx_dt = torch.einsum('ijk, kl->ijl', dx_dt, self.wc)
            y = y * dx_dt
        
        y = self.trans_y(torch.relu(y))
        y = self.gate_out(y) * y 
        # y = torch.relu(y)
        # y = torch.einsum('ijk, kl->ijl', y, self.wy)
        
        return y
    

# integrate with euler method, and return solutions corresponding to the input time sequence
class Euler():
    def __init__(self, func, funcx, y0, step_size):
        self.func = func
        self.funcx = funcx
        self.y0 = y0
        self.step_size = step_size 
        self.grid_constructor = self._grid_constructor_from_step_size(step_size)

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(t):
            # t is a sample time sequence 
            t_reverse = None
            if t[0] > t[1]:
                t = t.flip(0)
                t_reverse = True 
            
            start_time = t[0]
            end_time = t[-1]
            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]
            if t_reverse:
                t_infer = t_infer.flip(0)
            return t_infer
        return _grid_constructor

    def _step_func(self, dt, g, y0, t0):
        f0 = self.func(g, y0, self.funcx, t0)
        return dt * f0

    def integrate(self, g, t):
        time_grid = self.grid_constructor(t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            dy = self._step_func(dt, g, y0, t0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1
            y0 = y1
        return solution
    
    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


# integrate
def ddeint(func, g, y0, funcx, t, step_size):
    shapes, func, y0 = _check_inputs(func, y0)
    solver = Euler(func, funcx, y0, step_size=step_size)
    solution = solver.integrate(g, t)
    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution


class DDEBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, step_size):
        super(DDEBlock, self).__init__()
        self.ddefunc = DDEFunc(in_dim, hid_dim, step_size)
        self.step_size = step_size

    def forward(self, g, y0, funcx, t):
        ans = ddeint(self.ddefunc, g, y0, funcx, t, self.step_size)
        
        return ans
