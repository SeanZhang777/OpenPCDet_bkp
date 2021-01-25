from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from .detectors import build_detector
from .dense_heads.anchor_head_single import AnchorHeadSingle

import time
import onnx
import torch.onnx
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

def gather_point_features(model, voxel_features, voxel_num_points, coords):
    vfe = model.vfe

    points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    f_cluster = voxel_features[:, :, :3] - points_mean

    f_center_x = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * vfe.voxel_x + vfe.x_offset)
    f_center_y = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * vfe.voxel_y + vfe.y_offset)
    f_center_z = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * vfe.voxel_z + vfe.z_offset)
    f_center = torch.stack([f_center_x, f_center_y, f_center_z], dim=2)

    if vfe.use_absolute_xyz:
        features = [voxel_features, f_cluster, f_center]
    else:
        features = [voxel_features[..., 3:], f_cluster, f_center]

    if vfe.with_distance:
        points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        features.append(points_dist)
    features = torch.cat(features, dim=-1)

    voxel_count = features.shape[1]
    mask = vfe.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, 2).type_as(voxel_features)
    features *= mask

    return features

def model_fn_decorator():
    PredModelReturn = namedtuple('PredModelReturn', ['pred_dicts', 'recall_dicts'])

    def model_func(model, batch_dict, script_model=None):
        assert model.training == False

        load_data_to_gpu(batch_dict)
        voxels = batch_dict['voxels']
        voxel_num_points = batch_dict['voxel_num_points']
        voxel_coords = batch_dict['voxel_coords']

        points = batch_dict['points'][:, 1:5]

        # coords_x = voxel_coords[:, 3].cpu().numpy().astype(np.int32)
        # pad_coords_x = np.zeros((12000 - coords_x.shape[0],))
        # coords_x = np.concatenate([coords_x, pad_coords_x], axis=0)
        # np.savetxt("/home/vzz/test/tensor_check/py_coords_x.txt", coords_x, fmt="%d")
        # coords_y = voxel_coords[:, 2].cpu().numpy().astype(np.int32)
        # pad_coords_y = np.zeros((12000 - coords_y.shape[0],))
        # coords_y = np.concatenate([coords_y, pad_coords_y], axis=0)
        # np.savetxt("/home/vzz/test/tensor_check/py_coords_y.txt", coords_y, fmt="%d")

        # np_coords = voxel_coords.flatten().cpu().numpy().astype(np.int)
        # np_voxel_num_points = voxel_num_points.flatten().cpu().numpy().astype(np.int)
        # np_voxels = voxels.flatten().cpu().numpy().astype(np.float32)
        # np_points = points.flatten().cpu().numpy().astype(np.float32).reshape(1, -1)
        # # # np.savetxt("py_coor_x.txt", np_coords_x, fmt="%d")
        # np.savetxt("/home/vzz/test/tensor_check/py_coords.txt", np_coords, fmt="%d", delimiter=" ")
        # np.savetxt("/home/vzz/test/tensor_check/py_voxel_num_points.txt", np_voxel_num_points, fmt="%d", delimiter=" ")
        # np.savetxt("/home/vzz/test/tensor_check/py_voxels.txt", np_voxels, fmt="%f", delimiter=" ")
        # np.savetxt("/home/vzz/test/tensor_check/py_points.txt", np_points, fmt="%f")

        gathered_features = gather_point_features(model, voxels, voxel_num_points, voxel_coords)
        num_voxels = gathered_features.shape[0]
        pad_gather_features = torch.zeros((12000 - num_voxels, 100, 10), dtype=torch.float32).cuda()
        gathered_features = torch.cat([gathered_features, pad_gather_features], dim=0)
        gathered_features = torch.squeeze(gathered_features, 1)

        #gf = gathered_features.flatten().cpu().numpy().astype(np.float).reshape(1, -1)
        #np.savetxt("/home/vzz/test/tensor_check/py_gather_features.txt", gf, fmt="%f")
        vfe_output = model.vfe(gathered_features)
        # np_vfe_output = vfe_output.flatten().cpu().numpy().astype(np.float).reshape(1, -1)
        # np.savetxt("/home/vzz/test/tensor_check/py_pfe_output.txt", np_vfe_output, fmt="%f")
        vfe_output = torch.squeeze(vfe_output[:num_voxels])
        scatter_output = model.map_to_bev(vfe_output, voxel_coords)
        sp_feature2d = model.backbone_2d(scatter_output)
        output_list = model.dense_head(sp_feature2d)

        # np_scatter_output = scatter_output.flatten().cpu().numpy().astype(np.float).reshape(1, -1)
        # np.savetxt("/home/vzz/test/tensor_check/py_scatter_output.txt", np_scatter_output, fmt="%f")
        np_rpn_cls= output_list[0].flatten().cpu().numpy().astype(np.float32).reshape(1, -1)
        np.savetxt("/home/vzz/test/tensor_check/py_rpn_cls.txt", np_rpn_cls, fmt="%f")
        np_rpn_box = output_list[1].flatten().cpu().numpy().astype(np.float32).reshape(1, -1)
        np.savetxt("/home/vzz/test/tensor_check/py_rpn_box.txt", np_rpn_box, fmt="%f")
        np_rpn_dir = output_list[2].flatten().cpu().numpy().astype(np.float32).reshape(1, -1)
        np.savetxt("/home/vzz/test/tensor_check/py_rpn_dir.txt", np_rpn_dir, fmt="%f")

        # pfe_input = (voxels, voxel_num_points, voxel_coords)
        # torch.onnx.export(model, pfe_input, "pointpillars.onnx", export_params=True,
        #                   input_names=['voxels', 'voxel_num_points', 'coords'],
        #                   output_names=['conv_cls', 'conv_reg', 'conv_dir'],
        #                   dynamic_axes={'voxels' : {0 : 'num_voxels'},
        #                                 'voxel_num_points' : {0 : 'num_voxels'},
        #                                 'coords' : {0 : 'num_voxels'}},
        #                   opset_version=11)
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

        # dummy_voxels = torch.randn((12000, 1, 100, 10)).float().cuda()
        # # dummy_voxel_num_points = torch.randn((16000,)).int().cuda()
        # # dummy_coords = torch.randn((16000, 4)).int().cuda()
        # pfe_input = (dummy_voxels,)
        # torch.onnx.export(model.vfe, pfe_input, "pfe.onnx", export_params=True,
        #                   input_names=['voxels'],
        #                   output_names=['pillar_features'],
        #                   # dynamic_axes={'voxels' : {0 : 'num_voxels'},
        #                   #               # 'voxel_num_points' : {0 : 'num_voxels'},
        #                   #               # 'coords' : {0 : 'num_voxels'},
        #                   #               'pillar_features' : {0 : 'num_voxels'}}
        #                                   )
        # # voxel_feature = model.vfe(voxels, voxel_num_points, voxel_coords)
        # # batch_spatial_features = model.map_to_bev(voxel_feature, voxel_coords)
        # # rpn = nn.Sequential(model.backbone_2d, model.dense_head)
        # # torch.onnx.export(rpn, batch_spatial_features, "rpn.onnx", export_params=True,
        # #                     input_names=['batch_spatial_features'],
        # #                     output_names=['conv_cls', 'conv_reg', 'conv_angle']
        # #                  )
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

        # start = time.time()
        # if script_model:
        #     output_list = script_model(gathered_features, voxel_num_points, voxel_coords)
        # else:
        #     output_list = model(gathered_features, voxel_num_points, voxel_coords)
        # end = time.time()
        # print("Inference cost {} ms.".format((end - start) * 1000))

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
