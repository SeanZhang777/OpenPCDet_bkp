from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from .detectors import build_detector
from .dense_heads.anchor_head_single import AnchorHeadSingle

import time
import onnx
import onnxruntime

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def model_fn_decorator():
    PredModelReturn = namedtuple('PredModelReturn', ['pred_dicts', 'recall_dicts'])

    def model_func(model, batch_dict, script_model=None):
        assert model.training == False

        load_data_to_gpu(batch_dict)
        voxels = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        voxel_coords = batch_dict['voxel_coords']

        pfe_input = (voxels, voxel_num_points, voxel_coords)
        torch.onnx.export(model, pfe_input, "pointpillars.onnx", export_params=True,
                          input_names=['voxels', 'voxel_num_points', 'coords'],
                          output_names=['conv_cls', 'conv_reg', 'conv_dir'],
                          dynamic_axes={'voxels' : {0 : 'num_voxels'},
                                        'voxel_num_points' : {0 : 'num_voxels'},
                                        'coords' : {0 : 'num_voxels'}},
                          opset_version=11)
        # pp_model = onnx.load("pointpillars.onnx")
        # onnx.checker.check_model(pp_model)
        #
        # ort_session = onnxruntime.InferenceSession("pointpillars.onnx")
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(voxels),
        #               ort_session.get_inputs()[1].name: to_numpy(voxel_num_points),
        #               ort_session.get_inputs()[2].name: to_numpy(voxel_coords)}
        # ort_outs = ort_session.run(None, ort_inputs)
        # torch_out =model(voxels, voxel_num_points, voxel_coords)
        # np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
        # np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
        # np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
        # pfe_input = (voxels, voxel_num_points, voxel_coords)
        # torch.onnx.export(model.vfe, pfe_input, "pfe.onnx", export_params=True,
        #                   input_names=['voxels', 'voxel_num_points', 'coords'],
        #                   output_names=['pillar_features'],
        #                   dynamic_axes={'voxels' : {0 : 'num_voxels'},
        #                                 'voxel_num_points' : {0 : 'num_voxels'},
        #                                 'coords' : {0 : 'num_voxels'},
        #                                 'pillar_features' : {0 : 'num_voxels'}})
        # voxel_feature = model.vfe(voxels, voxel_num_points, voxel_coords)
        # batch_spatial_features = model.map_to_bev(voxel_feature, voxel_coords)
        # rpn = nn.Sequential(model.backbone_2d, model.dense_head)
        # torch.onnx.export(rpn, batch_spatial_features, "rpn.onnx", export_params=True,
        #                     input_names=['batch_spatial_features'],
        #                     output_names=['conv_cls', 'conv_reg', 'conv_angle']
        #                  )
        # pfe_model = onnx.load("pfe.onnx")
        # onnx.checker.check_model(pfe_model)
        #
        # ort_session = onnxruntime.InferenceSession("pfe.onnx")
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(voxels),
        #               ort_session.get_inputs()[1].name: to_numpy(voxel_num_points),
        #               ort_session.get_inputs()[2].name: to_numpy(voxel_coords)}
        # ort_outs = ort_session.run(None, ort_inputs)
        #
        # voxel_feature = model.vfe(voxels, voxel_num_points, voxel_coords)
        # batch_spatial_features = model.map_to_bev(voxel_feature, voxel_coords)
        # sp_feature2d = model.backbone_2d(batch_spatial_features)
        # torch_out = model.dense_head(sp_feature2d)

        # # compare ONNX Runtime and PyTorch results
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-01, atol=1e-03)

        # rpn_model = onnx.load("rpn.onnx")
        # onnx.checker.check_model(rpn_model)
        #
        # ort_session = onnxruntime.InferenceSession("rpn.onnx")
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch_spatial_features)}
        # ort_outs = ort_session.run(None, ort_inputs)
        #
        # # compare ONNX Runtime and PyTorch results
        # np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-01, atol=1e-03)
        # np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-01, atol=1e-03)
        # np.testing.assert_allclose(to_numpy(torch_out[2]), ort_outs[2], rtol=1e-01, atol=1e-03)

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
