import torch 
import torch.nn as nn  


from ddeutil import _check_inputs, _flat_to_shape


# dx/dt for backward process
class AdjointFunc(torch.nn.Module):
    def __init__(self, func, adjoint_params):
        super(AdjointFunc, self).__init__()
        self.func = func
        self.adjoint_params = adjoint_params
 
    def forward(self, g, y_aug, t, for_process=False):
        func = self.func
        adjoint_params = self.adjoint_params

        y = y_aug[0]
        adj_y = y_aug[1]
        with torch.enable_grad():
            y = y.detach().requires_grad_(True)
            func_eval = func(g, y, t, for_process)

            vjp_y, *vjp_params = torch.autograd.grad(
                func_eval, (y,) + adjoint_params, -adj_y,
                allow_unused=True, retain_graph=True
            )
            # autograd.grad returns None if no gradient, set to zero.
        vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
        vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                        for param, vjp_param in zip(adjoint_params, vjp_params)]
        return (func_eval, vjp_y, *vjp_params)


# integrate with euler method, and return solutions corresponding to the input time sequence
class Euler():
    def __init__(self, func, y0, step_size, for_process=True):
        self.func = func
        self.y0 = y0
        self.step_size = step_size 
        self.for_process = for_process
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

    def _step_func(self, dt, g, y0, t0, for_process=True):
        f0 = self.func(g, y0, t0, for_process)
        return dt * f0

    def integrate(self, g, t, for_process=True):
        time_grid = self.grid_constructor(t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            # print(t0, t1)
            dt = t1 - t0
            dy = self._step_func(dt, g, y0, t0, for_process)
            # self.hist[int(t0/self.step_size)] = y0
            # dt = t1 - t0
            # dy = 0
            # # use y from past \tau time
            # for tau in taus:
            #     if (tau > 0 and t0 < tau) or (tau < 0 and int((t0 - tau)/ self.step_size) >= len(self.hist)):
            #         y_tau = self.y0
            #     else:
            #         y_tau = self.hist[int((t0 - tau)/ self.step_size)]
            #     dy += self._step_func(dt, y_tau)
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
def ddeint(func, y0, g, t, step_size, for_process=True):
    shapes, func, y0 = _check_inputs(func, y0)
    solver = Euler(func, y0, step_size=step_size)
    solution = solver.integrate(g, t, for_process)
    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, forward_func, adjoint_func, y0, graph, t, step_size, *forward_params):
        ctx.adjoint_func = adjoint_func
        ctx.graph = graph
        ctx.step_size = step_size
        with torch.no_grad():
            ans = ddeint(forward_func, y0, graph, t, step_size, for_process=True)
            y = ans 
        ctx.save_for_backward(t, y, *forward_params)
        return ans

    @staticmethod
    def backward(ctx, *grad_y):
        with torch.no_grad():
            adjoint_func, graph, step_size = ctx.adjoint_func, ctx.graph, ctx.step_size
            t, y, *forward_params = ctx.saved_tensors
            forward_params = tuple(forward_params)
            grad_y = grad_y[0]  

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            aug_state = [y[-1], grad_y[-1]]  # y, vjp_y
            aug_state.extend([torch.zeros_like(param) for param in forward_params])  # vjp_params

            for i in range(len(t) - 1, 0, -1):
                aug_state = ddeint(adjoint_func, tuple(aug_state), graph, t[i - 1:i + 1].flip(0), step_size, for_process=False)
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[0] = y[i - 1]  # update to use our forward-pass estimate of the state
                aug_state[1] += grad_y[i - 1]  # update any gradients wrt state at this time point

            adj_y = aug_state[1]
            adj_params = aug_state[2:]

        return (None, None, adj_y, None, None, None, *adj_params)


