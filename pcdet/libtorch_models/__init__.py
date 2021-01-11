from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector
from .dense_heads.anchor_head_single import AnchorHeadSingle

import time

def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    PredModelReturn = namedtuple('PredModelReturn', ['pred_dicts', 'recall_dicts'])

    def model_func(model, batch_dict, script_model=None):
        assert model.training == False

        load_data_to_gpu(batch_dict)
        voxels = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        voxel_coords = batch_dict['voxel_coords']

        start = time.time()
        if script_model:
            output_list = script_model(voxels, voxel_num_points, voxel_coords)
        else:
            output_list = model(voxels, voxel_num_points, voxel_coords)
        end = time.time()
        print("Inference cost {} ms.".format((end - start) * 1000))

        batch_dict['cls_preds'] = output_list[0]
        batch_dict['box_preds'] = output_list[1]
        batch_dict['dir_cls_preds'] = output_list[2]

        start = time.time()
        if not model.training or model.dense_head.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = model.dense_head.generate_predicted_boxes(
                batch_size=int(batch_dict['batch_size']),
                cls_preds=batch_dict['cls_preds'],
                box_preds=batch_dict['box_preds'],
                dir_cls_preds=batch_dict['dir_cls_preds']
            )

            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False

            pred_dicts, recall_dicts = model.post_processing(batch_dict)
            end = time.time()
            print("Postprocessing cost {} ms.".format((end-start)*1000))
            return PredModelReturn(pred_dicts, recall_dicts)

    return model_func
