import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--remote', action='store_true', help='the code run on a server')
parser.add_argument('--device', type=int, default=0, help='the number of the gpu to use')
parser.add_argument('--epochs', type=int, default=200, help='train epochs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')

parser.add_argument('--filename', type=str, default='PEMS04')
parser.add_argument('--train-ratio', type=float, default=0.6, help='the ratio of training dataset')
parser.add_argument('--valid-ratio', type=float, default=0.2, help='the ratio of validating dataset')
parser.add_argument('--his-length', type=int, default=12, help='the length of history time series of input')
parser.add_argument('--pred-length', type=int, default=12, help='the length of target time series for prediction')
parser.add_argument('--num-layers', type=int, default=1, help='layer number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--step-size', type=float, default=1, help='step size for dde')
parser.add_argument('--hidden-dim', type=int, default=64, help='hidden dim')
parser.add_argument('--back', type=int, default=3, help='max back history length')
parser.add_argument('--thres', type=float, default=5e-3, help='the threshold of edge existing')

parser.add_argument('--log', action='store_true', help='if write log to files')
args = parser.parse_args()
