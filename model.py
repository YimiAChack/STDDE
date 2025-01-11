import torch
import torch.nn as nn
import torch.nn.functional as F 
import dgl 
from scipy import sparse

from dde import DDEBlock
from interpolate import NaturalCubicSpline
from dataset import get_normalized_adj_tensor
from get_delay import check_delay


# project input data to the history hidden states of DDE, the length of history 
# state is set as 1/step_size
class InputToHidden(nn.Module):
    def __init__(self, in_dim, hid_dim, step_size, back):
        super(InputToHidden, self).__init__()
        self.hid_dim = hid_dim
        self.hist_length = int(back / step_size)
        self.tohidden = nn.Sequential(
                            nn.Linear(in_dim, hid_dim), 
                            nn.ReLU(),
                            nn.Linear(hid_dim, hid_dim * self.hist_length)
                        )
        self.norm = nn.BatchNorm2d(hid_dim)

    def forward(self, input):
        # input: batch * node * in_dim
        batch_size, num_node, _ = input.shape
        input = input.reshape(batch_size, num_node, -1)
        hidden = self.tohidden(input).reshape(batch_size, num_node, self.hist_length, self.hid_dim) # hidden: batch * node * hist_length * hid_dim

        hidden = hidden.permute(1, 2, 0, 3)     # hidden: node * hist_length * batch * hid_dim
        # hidden = self.norm(hidden.permute(0, 3, 1, 2)).permute(2, 3, 0, 1)
        return hidden


class STDDE(nn.Module):
    def __init__(self, adj, num_node, in_dim, hidden_dim, out_dim, step_size, back=2, thres=0.1, extra=None):
        super(STDDE, self).__init__()
        self.in_dim = in_dim
        device = torch.device('cuda')
        self.adj = torch.FloatTensor(adj).to(device)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.step_size = step_size
        self.back = back 
        self.thres = thres
        self.extra = extra

        st, ed = torch.nonzero(self.adj).T
        self.graph = dgl.graph((st, ed), device=device)
        self.graph.edata['w'] = self.adj[self.adj > 0] 
        self.graph.edata['delay'] = torch.zeros(self.graph.num_edges(), device=device)

        self.node_embeddings = nn.Parameter(torch.rand(num_node, hidden_dim))
        self.semantic_weight = nn.Sequential(nn.ReLU(), nn.Softmax(dim=1))

        self.input_to_hidden = InputToHidden(in_dim=in_dim, hid_dim=hidden_dim, step_size=step_size, back=self.back)
        self.dde = DDEBlock(in_dim=in_dim, hid_dim=hidden_dim, step_size=step_size)
        self.dde2 = DDEBlock(in_dim=in_dim, hid_dim=hidden_dim, step_size=step_size)

        self.regress = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim//2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim//2, 1)
                    )
        self.pred = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim//2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim//2, 1)
                    )
        self.pred_one_step = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim//2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim//2, out_dim)
                    )
        
        self.get_control = nn.Linear((out_dim - 1) * in_dim * 4, (out_dim - 1) * in_dim * 4)
        self.print = True 

        
    def get_graph(self, A, all_delay, device):
        spatial_A = torch.from_numpy(A).to(torch.float32).to(device)
        # node_embeddings = self.node_embeddings(torch.arange(spatial_A.shape[0]).to(device))
        node_embeddings = self.node_embeddings
        semantic_A = self.semantic_weight(torch.mm(node_embeddings, node_embeddings.T))

        spatial_A = get_normalized_adj_tensor(spatial_A)
        A = semantic_A + spatial_A
        # A = spatial_A
        A_tensor, delay = check_delay(A, all_delay, back=self.back, threshold=self.thres)
        st, ed = torch.nonzero(A_tensor).T 
        graph = dgl.graph((st, ed))
        # A_numpy = A_tensor.detach().cpu().numpy()
        
        # graph = dgl.from_scipy(sparse.coo_matrix(A_numpy)).to(device)
        if self.training:
            if self.print:
                print('num of nodes:', graph.num_nodes(), 'num of edges:', graph.num_edges())
                self.print = False 
        if not self.training:
            self.print = True 

        graph.edata['delay'] = delay[delay < 0]
        # graph.edata['delay'] = torch.zeros(graph.num_edges(), device=device)
        graph.edata['w'] = A_tensor[A_tensor > 0]

        return graph


    def forward(self, A, all_delay, coeffs):
        # coeffs: List of 4 elements, element shape is batch * node * (seq-1) * in_dim
        device = coeffs[0].device
        g = self.get_graph(A, all_delay, device)
        # g = self.graph

        # times = torch.linspace(0, 11, 12).to(device)
        times = torch.arange(12).to(device) + self.back
        spline = NaturalCubicSpline(times, coeffs)
        initial_state = spline.evaluate(times[0])
        batch_size, num_node, _ = initial_state.shape

        # generate history hidden state from input, the time horizon of history state is set as 1
        hist_hidden = self.input_to_hidden(initial_state)
        # print(hist_hidden.shape, initial_state.shape)
        # fill in future hidden state with zero
        fur_length = int(self.out_dim / self.step_size)
        fur_hidden = torch.zeros(num_node, fur_length, batch_size, self.hidden_dim, device=device)
        hidden_state = torch.cat([hist_hidden, fur_hidden], dim=1)

        g.ndata['state'] = hidden_state
        y_hidden = self.dde(g, y0=hist_hidden[:, int(self.back/self.step_size) - 1, :, :], funcx=spline.derivative, t=times)
        
        # print(y_hidden.shape, hidden_state.shape)

        # autoregressive
        regression_loss = 0
        for i in range(self.out_dim):
            regression = self.regress(y_hidden[i]).squeeze(-1)
            target = spline.evaluate(times[i])[:, :, 0].T
            regression_loss += F.smooth_l1_loss(regression, target)

        # prediction
        # ## dde multi step prediction
        # fur_length = int((self.out_dim + self.back) / self.step_size)
        # hidden = torch.zeros(num_node, fur_length, batch_size, self.hidden_dim, device=device)
        # hidden[:, :self.back, :, :] = y_hidden[-self.back:].transpose(0, 1)
        # g.ndata['state'] = hidden

        # coeffs = torch.stack(coeffs, dim=-1)
        # fake_control = self.get_control(coeffs.reshape(batch_size, num_node, -1))
        # fake_control = fake_control.reshape(batch_size, num_node, self.out_dim - 1, self.in_dim, 4)
        # fake_control = tuple(fake_control[:, :, :, :, i] for i in range(4))
        # spline = NaturalCubicSpline(times, fake_control)

        # # y_hidden = self.dde2(g, y0=hidden[:, 0, :, :], funcx=None, t=times+1)
        # y_hidden = self.dde2(g, y0=hidden[:, int(self.back/self.step_size) - 1, :, :], funcx=spline.derivative, t=times+1)
        
        # y_pred = self.pred(y_hidden).squeeze(-1).permute(2, 1, 0)

        ## one step prediction 
        y_pred = self.pred_one_step(y_hidden[-1]).permute(1, 0, 2)

        if self.training:
            return y_pred, regression_loss/12
        else:
            return y_pred
