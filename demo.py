# Copyright (c) HKUST SAIL-Lab and Horizon Robotics.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
from tqdm import tqdm

from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(dtype)


def demo(args):
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    # _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
    model_dir = args.ckpt
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model = SailRecon(kv_cache=True)
    print('loading sailrecon model....')
    # if model_dir is not None:
    #     model.load_state_dict(torch.load(model_dir))
    # else:
    #     model.load_state_dict(
    #         torch.hub.load_state_dict_from_url(_URL, model_dir=model_dir)
    #     )
    
    # 加载本地权重
    state_dict = torch.load("ckpt/sailrecon.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    print('权重文件加载成功')

    model = model.to(device=device)
    model.eval()
    print(model)

    # Load and preprocess example images
    scene_name = "1"
    if args.vid_dir is not None:
        import cv2

        image_names = []
        video_path = args.vid_dir
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        tmp_file = os.path.join("tmp_video", os.path.basename(video_path).split(".")[0])
        os.makedirs(tmp_file, exist_ok=True)
        count = 0
        video_frame_num = 0
        interval = 1
        with tqdm(total=frames//interval,desc='downloading the video') as pbar: 
            while True:
                gotit, frame = vs.read()
                if not gotit:
                    break
                if count % interval == 0:
                    image_path = os.path.join(tmp_file, f"{video_frame_num:06}.png")
                    cv2.imwrite(image_path, frame)
                    image_names.append(image_path)
                    video_frame_num += 1
                    pbar.update(1)
                count += 1
        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.basename(video_path).split(".")[0]
    else:
        # 使用 os.listdir() 函数获取 args.img_dir 指定的目录下所有文件和目录名称的列表。这个初始列表不包含完整路径，只有文件名。
        image_names = os.listdir(args.img_dir)
        # sorted(image_names): 对文件名列表进行字母/词典顺序排序。
        # 列表推导式: 遍历已排序的文件名 (f)，使用 os.path.join() 将目录路径 (args.img_dir) 与文件名连接起来，创建包含完整路径的新列表。
        image_names = [os.path.join(args.img_dir, f) for f in sorted(image_names)]
        images = load_and_preprocess_images(image_names).to(device)
        # 使用 os.path.basename() 函数从 args.img_dir 中提取目录路径的最后一个组成部分。这通常就是文件夹本身的名称，并将其赋值给变量 scene_name。
        scene_name = os.path.basename(args.img_dir)

    # anchor image selection
    # change it
    select_indices = uniform_sample(len(image_names), min(100, len(image_names)))
    anchor_images = images[select_indices]

    os.makedirs(os.path.join(args.out_dir, scene_name), exist_ok=True)

    with torch.no_grad():
        with torch.amp.autocast('cuda',dtype=dtype):
            # processing anchor images to build scene representation (kv_cache)
            print("Processing anchor images ...")
            model.tmp_forward(anchor_images)
            # remove the global transformer blocks to save memory during relocalization
            del model.aggregator.global_blocks
            # relocalization on all images
            predictions = []

            # 使用了 tqdm 库来显示一个动态的进度条，让用户知道任务的执行状态。
            '''
            tqdm(...): 这是一个用于创建进度条的函数，它将整个 for 循环包裹起来。
            total=len(image_names): 设置进度条的总任务量。这里使用 len(image_names)，说明总共需要处理的图片数量等于 image_names 列表中的元素个数。
            desc="Relocalizing": 设置进度条前面显示的描述文本。在这里，进度条会显示 “Relocalizing: [进度]”。
            as pbar: 将创建的进度条对象命名为 pbar，用于在循环内部更新进度。
            with ...: 这是一个 Python 的上下文管理器语法。它确保无论循环是否完成或发生错误，进度条都会被正确关闭和清理。
            '''
            with tqdm(total=len(image_names), desc="Relocalizing") as pbar:
                for img_split in images.split(20, dim=0):
                    pbar.update(20)
                    predictions += to_cpu(model.reloc(img_split, memory_save=False))

            # save the predicted point cloud and camera poses

            from eval.utils.geometry import save_pointcloud_with_plyfile

            save_pointcloud_with_plyfile(
                predictions, os.path.join(args.out_dir, scene_name, "pred.ply")
            )

            import numpy as np

            from eval.utils.eval_utils import save_kitti_poses

            poses_w2c_estimated = [
                one_result["extrinsic"][0].cpu().numpy() for one_result in predictions
            ]
            poses_c2w_estimated = [
                np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])]))
                for pose in poses_w2c_estimated
            ]

            save_kitti_poses(
                poses_c2w_estimated,
                os.path.join(args.out_dir, scene_name, "pred.txt"),
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--img_dir", type=str, default="img_set/test", help="input image folder"
    )
    args.add_argument("--vid_dir", type=str, default=None, help="input video path")
    args.add_argument("--out_dir", type=str, default="outputs", help="output folder")
    args.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model checkpoint"
    )
    args = args.parse_args()
    demo(args)

