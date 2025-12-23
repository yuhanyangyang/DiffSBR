import argparse
import pickle
import time
from util import Data, split_validation, init_seed
import models.gaussian_diffusion as gd
from model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Sports_and_Outdoors', help='dataset name: Office_Products/Cell_Phones_and_Accessories/Sports_and_Outdoors')
parser.add_argument('--epoch', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size 512')
parser.add_argument('--embSize', type=int, default=100, help='embedding size 100')
parser.add_argument('--textEmbSize', type=int, default=100, help='modality embedding size 100')
parser.add_argument('--imgEmbSize', type=int, default=100, help='modality embedding size 100')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay 3')
parser.add_argument('--co_layer', type=float, default=4, help='the number of item co-occurence graph')
parser.add_argument('--num_negatives', type=int, default=100, help='the number of negatives 100')
parser.add_argument('--lamdba', type=float, default=1e-2, help='item co-occurrence/proxy/conterfact constance')
parser.add_argument('--cuda', default='true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='1', help='gpu card ID')
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=32, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--dims', type=str, default='[200,600]', help='the dims for the DNN')
parser.add_argument('--recdim', type=int,default=10,
                        help="the embedding size of lightGCN")
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--decay', type=float, default=1e-4,
                    help="the weight decay for l2 normalizaton")

opt = parser.parse_args()
print(opt)

def main():

    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))


    if opt.dataset == 'Cell_Phones_and_Accessories':
        n_node = 9091
    elif opt.dataset == 'Sports_and_Outdoors':
        n_node = 14650
    elif opt.dataset == 'Grocery_and_Gourmet_Food':
        n_node = 7286
    elif opt.dataset == 'Instacart':
        n_node = 10009
    else:
        print("unkonwn dataset")

    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=True, n_node=n_node)
    model = trans_to_cuda(Diff(opt, adjacency=train_data.adjacency, n_node=n_node, lr=opt.lr, co_layer =opt.co_layer, l2=opt.l2, item_beta=opt.lamdba, pro_beta=opt.lamdba, cri_beta = opt.lamdba, dataset=opt.dataset, num_heads=opt.num_heads, emb_size=opt.embSize, text_emb_size=opt.textEmbSize, batch_size=opt.batchSize, num_negatives=opt.num_negatives))

    if opt.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif opt.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % opt.mean_type)
    diffusion = trans_to_cuda(gd.GaussianDiffusion(opt, opt.embSize,mean_type, opt.noise_schedule, opt.noise_scale, opt.noise_min, opt.noise_max, opt.steps))
    out_dims = eval(opt.dims) + [opt.embSize]
    in_dims = out_dims[::-1]
    id_reverse_model = trans_to_cuda(DNN(opt, opt.embSize, in_dims, out_dims, opt.recdim, time_type="cat", norm=opt.norm, lr=opt.lr))
    mo_reverse_model = trans_to_cuda(DNN(opt, opt.embSize, in_dims, out_dims, opt.recdim, time_type="cat", norm=opt.norm, lr=opt.lr))
    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model,diffusion,id_reverse_model,mo_reverse_model, train_data, test_data,epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    main()
