import os, sys
import argparse
import time
import math
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import scipy.spatial as spatial

from network import Network
from datasets import BaseDataset
from mesh import extract_mesh
from misc import seed_all, get_log, get_logger, creat_logger, knn_gather_np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--data_set', type=str, default='',
                        choices=['PCPNet', 'FamousShape', 'FamousShape5k', 'SceneNN', 'Others', 'KITTI_sub', 'Semantic3D', '3DScene', 'WireframePC', 'NestPC', 'Plane'])
    ### Train
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--save_inter', type=int, default=10000)
    parser.add_argument('--warn_up', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    ### Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='/data1/lq/Dataset/')
    parser.add_argument('--testset_list', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_points', type=int, default=5000)
    parser.add_argument('--num_query', type=int, default=10)
    parser.add_argument('--num_knn', type=int, default=64)
    parser.add_argument('--dis_k', type=int, default=50)
    parser.add_argument('--dis_scale', type=float, default=0.15)
    ### Test
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--ckpt_iter', type=int, default=None)
    parser.add_argument('--save_normal_npy', type=eval, default=False, choices=[True, False])
    parser.add_argument('--save_normal_xyz', type=eval, default=False, choices=[True, False])
    parser.add_argument('--save_mesh', type=eval, default=False, choices=[True, False])
    parser.add_argument('--avg_nor', type=eval, default=False, choices=[True, False])
    parser.add_argument('--mesh_far', type=float, default=-1.0)
    args = parser.parse_args()
    return args


def update_learning_rate(optimizer, iter_step, init_lr, max_iter):
    warn_up = args.warn_up  # 2000, 10000
    lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
    lr = lr * init_lr
    for g in optimizer.param_groups:
        g['lr'] = lr


def train(data_list, log_dir, log_name, ckpt_dir, id=None):
    ### Dataset
    train_set = BaseDataset(root=args.dataset_root,
                            data_set=args.data_set,
                            data_list=data_list,
                            num_points=args.num_points,
                            num_query=args.num_query,
                            num_knn=args.num_knn,
                            dis_k=args.dis_k,
                            dis_scale=args.dis_scale,
                        )
    dataloader = torch.utils.data.DataLoader(
                            train_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,   # faster speed
                        )

    log_flag = True
    num_shapes = len(train_set.cur_sets)
    for shape_idx, shape_name in enumerate(train_set.cur_sets):
        ### Model
        my_model = Network(args.num_points, num_knn=args.num_knn).to(_device).train()
        optimizer = optim.Adam(my_model.parameters(), lr=args.lr)

        train_set.process_data(shape_name)
        iter_dataloader = iter(dataloader)

        if log_flag:
            log_name = 'train(%s)(%d)' % (log_name, os.getpid())
            if id is not None:
                log_name = log_name + '-%d' % id
            logger = get_logger(args, log_dir, log_name, file_name='log_'+data_list, model=my_model)
            log_flag = False

        time_sum = 0
        for iter_i in range(1, args.max_iter+1):
            update_learning_rate(optimizer, iter_i, init_lr=args.lr, max_iter=args.max_iter)

            data = iter_dataloader.next()
            start_time = time.time()

            pcl_raw = data['pcl_raw'].to(_device)                                                # (B, M, 3),  M > N
            pcl_source = data['pcl_source'].to(_device)                                          # (B, N, 3)
            knn_idx = data['knn_idx'].to(_device)                                                # (B, N, K)
            pcl_raw_sub = data['pcl_raw_sub'].to(_device) if 'pcl_raw_sub' in data else None     # (B, N, 3)

            ### Reset gradient and model state
            my_model.train()
            optimizer.zero_grad()

            pcl_source = torch.cat([pcl_source, pcl_raw_sub], dim=-2)

            grad_norm = my_model(pcl_source)
            loss, loss_tuple = my_model.get_loss(pcl_raw=pcl_raw, pcl_source=pcl_source, knn_idx=knn_idx)

            ### Backward and optimize
            loss.backward()
            optimizer.step()

            elapsed_time = time.time() - start_time
            time_sum += elapsed_time

            if iter_i % (args.save_inter//10) == 0:
                ss = ''
                for l in loss_tuple:
                    ss += '%.6f+' % l.item()
                logger.info('shape:%d/%d, iter:%d/%d, loss=%.6f(%s), lr=%.6f' % (
                            shape_idx+1, num_shapes, iter_i, args.max_iter, loss, ss[:-1], optimizer.param_groups[0]['lr']))

            if iter_i % args.save_inter == 0 or iter_i == args.max_iter:
                model_filename = os.path.join(ckpt_dir, shape_name + '_%d.pt' % iter_i)
                torch.save(my_model.state_dict(), model_filename)
                logger.info('Save model: ' + model_filename)

                # pc_nor = torch.cat([pcl_source, grad_norm], dim=-1)[0].cpu().detach().numpy()
                # np.savetxt(model_filename[:-3] + '.txt', pc_nor, fmt='%.6f')

        del my_model, optimizer
        logger.info('Time: %.2f sec\n' % time_sum)

    return 1


def test(data_list):
    ckpt_paths = os.path.join(args.log_root, args.ckpt_dir, 'ckpts/*.pt')
    assert len(ckpt_paths) > 0

    ### Dataset
    test_set = BaseDataset(root=args.dataset_root,
                            data_set=args.data_set,
                            data_list=data_list,
                        )

    ### Model
    print('Building model ...')
    my_model = Network(args.num_points, num_knn=args.num_knn).to(_device).eval()

    ### Log
    PID = os.getpid()
    output_dir = os.path.join(args.log_root, args.ckpt_dir, 'test_%s' % args.ckpt_iter)
    os.makedirs(output_dir, exist_ok=True)
    logger = creat_logger('test(%d)(%s-%s)' % (PID, args.ckpt_dir, args.ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    trainable_num = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info('Num_params_trainable: %d' % trainable_num)

    max_n = int(2e5)
    list_bad = {}
    list_rms = []
    list_rms_o = []
    list_p90 = []
    time_sum = 0

    for shape_idx, shape_name in enumerate(test_set.cur_sets):
        ### load the trained model
        ckpt_path = os.path.join(args.log_root, args.ckpt_dir, 'ckpts/%s_%s.pt' % (shape_name, args.ckpt_iter))
        if not os.path.exists(ckpt_path):
            logger.info('File not exist: ' + ckpt_path)
            continue
        my_model.load_state_dict(torch.load(ckpt_path, map_location=_device), strict=False)

        ### load a point cloud and shuffle the order of points
        pcl_raw, nor_gt = test_set.get_data(shape_name)         # (N, 3)
        start_time = time.time()

        num_point = pcl_raw.shape[0]
        rand_idxs = np.random.choice(num_point, num_point, replace=False)
        pcl = pcl_raw[rand_idxs, :3]

        ### if there are too many points, the point cloud will be processed in batches,
        ### the number of output vectors may be less than the number of initial points (decided by remainder).
        if num_point <= max_n:
            pcl_source = torch.from_numpy(pcl).float().to(_device)
            with torch.no_grad():
                grad_norm = my_model(pcl_source)
                grad_norm = grad_norm.cpu().detach().numpy()
        else:
            k = math.ceil(num_point / max_n)
            remainder = int(max_n * k % num_point)
            print('Split data: ', num_point, k, remainder)
            pcl_new = np.concatenate((pcl, pcl[:remainder]), axis=0)
            pcl_source = torch.from_numpy(pcl_new).float()           # (max_n*k, D)
            grad_norm = np.zeros((pcl_new.shape[0], 3))              # (N, 3)
            with torch.no_grad():
                for i in range(k):
                    grad_norm_s = my_model(pcl_source[max_n*i:max_n*(i+1)].to(_device))
                    grad_norm[max_n*i:max_n*(i+1)] = grad_norm_s.cpu().detach().numpy()
            grad_norm = grad_norm[:max_n*k-remainder]

        ### reorder and normalize the vectors, eliminate zero values
        pred_norm = np.zeros_like(grad_norm)
        pred_norm[rand_idxs, :] = grad_norm
        pred_norm[np.linalg.norm(pred_norm, axis=-1) == 0.0] = 1.0
        pred_norm /= np.linalg.norm(pred_norm, axis=-1, keepdims=True)

        elapsed_time = time.time() - start_time
        time_sum += elapsed_time

        assert pcl_raw.shape == pred_norm.shape
        if args.avg_nor:
            # k_idex = []
            ptree = spatial.cKDTree(pcl_raw)
            _, k_idex = ptree.query(pcl_raw, k=1, distance_upper_bound=0.3)
            if k_idex.ndim == 1:
                k_idex = k_idex[:, None]
            pred_norm = knn_gather_np(pred_norm, k_idex)
            pred_norm = pred_norm.mean(axis=1)

        if args.save_normal_npy or args.save_normal_xyz:
            normal_dir = os.path.join(output_dir, 'pred_normal')
            os.makedirs(normal_dir, exist_ok=True)
            path_save = os.path.join(normal_dir, shape_name)

            if args.save_normal_npy:
                np.save(path_save + '_normal.npy', pred_norm)
            if args.save_normal_xyz:
                pc_nor = np.concatenate([pcl_raw, pred_norm], axis=-1)
                # k = 1000; n = 50 # 10
                # pc_nor = pc_nor[n*k:n*k+k, :]
                np.savetxt(path_save + '.xyz', pc_nor, fmt='%.6f')

        ### evaluation
        nn = np.sum(np.multiply(-1 * nor_gt, pred_norm), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1
        ang = np.rad2deg(np.arccos(np.abs(nn)))
        rms = np.sqrt(np.mean(np.square(ang)))

        ang_o = np.rad2deg(np.arccos(nn))
        ids = ang_o < 90.0
        p90 = sum(ids) / pred_norm.shape[0] * 100

        ### if more than half of points have wrong orientation, then flip all normals
        if p90 < 50.0:
            nn = np.sum(np.multiply(nor_gt, pred_norm), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1
            ang_o = np.rad2deg(np.arccos(nn))
            ids = ang_o < 90.0
            p90 = sum(ids) / pred_norm.shape[0] * 100

        rms_o = np.sqrt(np.mean(np.square(ang_o)))
        list_rms.append(rms)
        list_rms_o.append(rms_o)
        list_p90.append(p90)
        if np.mean(p90) < 90.0:
            list_bad[shape_name] = p90
        logger.info('RMSE_U: %.3f, RMSE_O: %.3f, Correct orientation: %.3f %% (%s)' % (rms, rms_o, p90, shape_name))

        if args.save_mesh:
            mesh_dir = os.path.join(output_dir, 'recon_mesh')
            os.makedirs(mesh_dir, exist_ok=True)
            mesh = extract_mesh(my_model.net.forward, bbox_min=test_set.bbox_min, bbox_max=test_set.bbox_max,
                                points_gt=pcl_raw, mesh_far=args.mesh_far)
            mesh.export(os.path.join(mesh_dir, '%s.obj' % shape_name))

    if len(list_p90) > 0:
        logger.info('Time: %.2f sec\n' % time_sum)
        logger.info('Average || RMSE_U: %.3f, RMSE_O: %.3f, Correct orientation: %.3f %%' % (np.mean(list_rms), np.mean(list_rms_o), np.mean(list_p90)))
        ss = ''
        for k, v in list_bad.items():
            ss += '%s: %.3f %%\n' % (k, v)
        logger.info('Bad results in %d shapes: \n%s' % (len(list_p90), ss))
    return 1



### Arguments
args = parse_arguments()

if len(args.testset_list) == 0:
    args.testset_list = 'testset_' + args.data_set

if args.data_set in ['SceneNN', 'Semantic3D', 'KITTI_sub', 'Others', '3DScene']:
    args.lr = 0.00001
    args.dis_k = 64

if args.data_set in ['PCPNet']:
    args.dis_k = 25
    # args.lr = 0.0007
    eval_list = ['testset_no_noise', 'testset_low_noise', 'testset_med_noise', 'testset_high_noise',
                'testset_vardensity_striped', 'testset_vardensity_gradient']

if args.data_set in ['FamousShape']:
    args.dis_k = 50
    args.lr = 0.002
    eval_list = ['testset_noise_clean', 'testset_noise_low', 'testset_noise_med', 'testset_noise_high',
                'testset_density_stripe', 'testset_density_gradient']

if args.data_set == 'FamousShape5k':
    args.num_points = 1000
    args.dis_k = 10

if args.data_set == 'WireframePC':
    args.max_iter = 10000
    args.save_inter = 2500
    args.num_points = 300
    args.dis_k = 3
    args.warn_up = 2000
    # args.lr = 0.0001

if args.data_set == 'NestPC':
    args.dis_k = 50
    # args.num_knn = 6
    args.lr = 0.0001

torch.cuda.set_device(args.gpu)
_device = torch.device('cuda')

seed_all(args.seed)
args.tag = args.data_set


import torch.multiprocessing as mp
if __name__ == '__main__':
    if args.mode == 'train':
        num_processes = 1

        log_dir, log_name, ckpt_dir = get_log(args)

        if num_processes > 1:
            torch.multiprocessing.set_start_method('spawn')

            processes = []
            for i in range(num_processes):
                p = mp.Process(target=train, args=(eval_list[i], log_dir, log_name, ckpt_dir, i+1))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            train(args.testset_list, log_dir, log_name, ckpt_dir)

    elif args.mode == 'test':
        test(data_list=args.testset_list)
    else:
        print('The mode is unsupported!')