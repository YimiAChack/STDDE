import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from loguru import logger
import dgl 
from scipy import sparse
# from torch.utils.tensorboard import SummaryWriter

from args import args
from model import STDDE
from dataset import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np
from get_delay import cal_all_delay


torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

def train(loader, model, A, all_delay, optimizer, criterion, device):
    batch_loss = 0
    batch_regression_loss = 0
    for idx, batch in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()

        batch = tuple(b.to(device) for b in batch)
        *inputs, targets = batch
        outputs, regression_loss = model(A, all_delay, inputs) 
        loss = criterion(outputs, targets) 
        loss += regression_loss
        loss.backward()
        optimizer.step()

        batch_loss += loss.detach().cpu().item() 
        batch_regression_loss += regression_loss.detach().cpu().item()

    # print(f'##on train data## total loss: {batch_loss/(idx+1)}, pred loss: {(batch_loss - batch_regression_loss)/(idx+1)}, regression loss: {batch_regression_loss/(idx+1)}')
    return batch_loss/(idx + 1), (batch_loss - batch_regression_loss)/(idx+1), batch_regression_loss/(idx+1)


@torch.no_grad()
def eval(loader, model, A, all_delay, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    for idx, batch in enumerate(tqdm(loader)):
        model.eval()

        batch = tuple(b.to(device) for b in batch)
        *inputs, targets = batch
        output = model(A, all_delay, inputs).reshape(*targets.shape)
        
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)


def main(args):
    # random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    if args.log:
        logger.add('logs/log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)

    data, mean, std, A = read_data(args)
    train_loader, valid_loader, test_loader = generate_dataset(data, args)
    num_node = data.shape[1]
    all_delay = cal_all_delay(data, args.filename)

    # A = get_normalized_adj(A)
    # # A, delay = cal_delay(A, data, args.back)
    # A_tensor = torch.from_numpy(A).to(torch.float32).to(device)
    # A_tensor, delay = check_delay(A_tensor, data, back=args.back)
    # A_numpy = A_tensor.cpu().numpy()
    
    # # delay = torch.from_numpy(delay).to(torch.float32).to(device)

    # graph = dgl.from_scipy(sparse.coo_matrix(A_numpy)).to(device)
    # print('num of nodes:', graph.num_nodes(), 'num of edges:', graph.num_edges())
    # graph.edata['delay'] = delay[delay < 0]
    # # graph.edata['delay'] = torch.zeros(graph.num_edges()).to(device)
    # graph.edata['w'] = A_tensor[A_tensor > 0]

    net = STDDE(adj=A, num_node=num_node, in_dim=data.shape[2]+1, hidden_dim=args.hidden_dim, out_dim=12, 
                step_size=args.step_size, back=args.back, thres=args.thres, extra=[mean, std])
    net = net.to(device)

    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_valid_rmse = 1000 
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)

    for epoch in range(1, args.epochs+1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        total_loss, pred_loss, regression_loss = train(train_loader, net, A, all_delay, optimizer, criterion, device)
        print('Evaluating...')
        train_rmse, train_mae, train_mape = eval(train_loader, net, A, all_delay, std, mean, device)
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, A, all_delay, std, mean, device)
        test_rmse, test_mae, test_mape = eval(test_loader, net, A, all_delay, std, mean, device)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            print('New best results!')
            torch.save(net.state_dict(), f'net_params_{args.filename}_{args.device}.pkl')

        if args.log:
            logger.info(f'=====Epoch {epoch}=====\n' + 
                        f'\n##on train data## total loss: {total_loss}, pred loss: {pred_loss}, regression loss: {regression_loss} \n' + 
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n' + 
                        f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}\n')
        else:
            print(f'\n##on train data## total loss: {total_loss}, pred loss: {pred_loss}, regression loss: {regression_loss} \n' + 
                    f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                    f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n' 
                    f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}\n'
                )
        
        scheduler.step()

    # # net.load_state_dict(torch.load(f'net_params_{args.filename}_{args.device}.pkl'))
    # net.load_state_dict(torch.load('net_params_PEMS04_7.pkl', map_location=torch.device('cpu')))
    test_rmse, test_mae, test_mape = eval(test_loader, net, A, all_delay, std, mean, device)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')


if __name__ == '__main__':
    main(args)
