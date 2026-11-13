import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import json
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.spectrack import SPECTRACK
import lib.test.parameter.spectrack as rgbe_prompt_params
from lib.train.dataset.depth_utils import get_x_frame
import multiprocessing
import torch
import time


def genConfig(seq_path, set_type):
    if set_type == 'VisEvent':
        RGB_img_list = sorted([seq_path + '/vis_imgs/' + p for p in os.listdir(seq_path + '/vis_imgs') if os.path.splitext(p)[1] == '.bmp'])
        E_img_list = sorted([seq_path + '/event_imgs/' + p for p in os.listdir(seq_path + '/event_imgs') if os.path.splitext(p)[1] == '.bmp'])

        RGB_gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        absent_label = np.loadtxt(seq_path + '/absent_label.txt')

    elif set_type == 'VisEvent_miss':
        RGB_img_list = sorted([seq_path + '/vis_imgs/' + p for p in os.listdir(seq_path + '/vis_imgs') if os.path.splitext(p)[1] == '.bmp'])
        E_img_list = sorted([seq_path + '/event_imgs/' + p for p in os.listdir(seq_path + '/event_imgs') if os.path.splitext(p)[1] == '.bmp'])

        RGB_gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        absent_label = np.loadtxt(seq_path + '/absent_label.txt')

    return RGB_img_list, E_img_list, RGB_gt, absent_label


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=60, debug=0, script_name='prompt', missing_json_path=None):
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    seq_txt = seq_name

    save_name = '{}'.format(yaml_name)
    save_path = f'./RGBE_workspace/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'./RGBE_workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    if script_name == 'spectrack':
        params = rgbe_prompt_params.parameters(yaml_name)
        ostrack = SPECTRACK(params, dataset_name)
        tracker = SPECTRACK_RGBE(tracker=ostrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+ seq_name +'——————————————')
    RGB_img_list, E_img_list, RGB_gt, absent_label = genConfig(seq_path, dataset_name)


    seq_mask_list = None
    if missing_json_path is not None:
        with open(missing_json_path, 'r') as f:
            mask_dict = json.load(f)
        seq_key = seq_name.split('/')[-1] if '/' in seq_name else seq_name
        if seq_key in mask_dict:
            seq_mask_list = mask_dict[seq_key]['data']
            print(f'[MissModal] Sequence "{seq_key}" loaded, {len(seq_mask_list)} frames.')
        else:
            print(f'[MissModal] WARNING: "{seq_key}" not found in JSON, no masking applied.')


    frame_offset = 0
    if absent_label[0] == 0:
        first_present_idx = absent_label.argmax()
        frame_offset = first_present_idx
        RGB_img_list = RGB_img_list[first_present_idx:]
        E_img_list = E_img_list[first_present_idx:]
        RGB_gt = RGB_gt[first_present_idx:]

    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0

    for frame_idx, (rgb_path, E_path) in enumerate(zip(RGB_img_list, E_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:

            image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))

            global_idx = frame_offset + frame_idx
            if seq_mask_list is not None and global_idx < len(seq_mask_list):
                if seq_mask_list[global_idx][0] == 0.0:
                    image[:, :, :3] = 0
                if seq_mask_list[global_idx][1] == 0.0:
                    image[:, :, 3:] = 0
            tracker.initialize(image, RGB_gt[0].tolist())
        elif frame_idx > 0:

            image = get_x_frame(rgb_path, E_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbrgb'))

            global_idx = frame_offset + frame_idx
            if seq_mask_list is not None and global_idx < len(seq_mask_list):
                if seq_mask_list[global_idx][0] == 0.0:
                    image[:, :, :3] = 0
                if seq_mask_list[global_idx][1] == 0.0:
                    image[:, :, 3:] = 0
            region, confidence = tracker.track(image)
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class SPECTRACK_RGBE(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)

        init_info = {'init_bbox': list(gt_bbox_np)}  
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBE dataset.')
    parser.add_argument('--script_name', type=str, default='', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='VisEvent', help='Name of dataset (VisEvent).')
    parser.add_argument('--threads', default=4, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=60, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    parser.add_argument('--missing_json', default='', type=str, help='path to missing modal annotation JSON file')  
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))

    seq_list = None
    if dataset_name == 'VisEvent':
        seq_home = '/data/jiangchenxi/dataset/visevent/testall/test_subset'
        with open(join(seq_home, 'testlist.txt'), 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    missing_json_path = args.missing_json if args.missing_json != '' else None  

    start = time.time()
    if args.mode == 'parallel':

        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, missing_json_path) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list

        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, missing_json_path) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
