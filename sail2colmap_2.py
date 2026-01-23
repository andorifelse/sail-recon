import argparse
import os
import torch
import numpy as np
import cv2
import pycolmap
from PIL import Image
from tqdm import tqdm

# 导入 sail-recon 模块
from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images
from sailrecon.dependency.np_to_pycolmap import _build_pycolmap_intri

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


# ==============================================================================
# 修复函数 V8：修复 ListPoint2D 缺失问题
# ==============================================================================
def fixed_batch_np_matrix_to_pycolmap_wo_track(
        points3d,
        points_xyf,
        points_rgb,
        extrinsics,
        intrinsics,
        image_size,
        shared_camera=False,
        camera_type="PINHOLE",
):
    print(f"Detected pycolmap version: {pycolmap.__version__}")

    N = len(extrinsics)
    P = len(points3d)

    reconstruction = pycolmap.Reconstruction()

    # 1. 添加 3D 点
    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # 2. 遍历每一帧
    for fidx in range(N):
        # --- 设置相机 (Camera) ---
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera()
            try:
                camera.model = camera_type
            except TypeError:
                try:
                    camera.model = pycolmap.CameraModelId[camera_type]
                except:
                    pass  # Keep default or try string

            camera.width = int(image_size[0])
            camera.height = int(image_size[1])
            camera.params = pycolmap_intri
            camera.camera_id = int(fidx + 1)

            if hasattr(reconstruction, "add_camera_with_trivial_rig"):
                reconstruction.add_camera_with_trivial_rig(camera)
            else:
                reconstruction.add_camera(camera)

        # --- 设置图像 (Image) ---
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )

        image_name = f"image_{fidx + 1}.png"
        image_id = fidx + 1

        # 构造 Image
        try:
            image = pycolmap.Image(name=image_name, camera_id=int(camera.camera_id))
        except TypeError:
            try:
                image = pycolmap.Image(image_name, int(camera.camera_id))
            except TypeError:
                # 最后的保底: 空构造 + 赋值
                image = pycolmap.Image()
                image.name = image_name
                image.camera_id = int(camera.camera_id)

        try:
            image.image_id = image_id
        except:
            pass

        # --- 设置 2D 点 ---
        points2D_list = []
        point2D_idx = 0
        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]

            # Point2D 构造
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # Track 关联
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        # [V8 FIX] 移除 ListPoint2D，直接赋值或 extend
        try:
            # 现代 pybind11 通常支持直接赋值 list
            image.points2D = points2D_list
        except (TypeError, AttributeError):
            try:
                # 如果是 vector 且赋值失败，尝试 extend (针对只读属性但内容可变)
                image.points2D.extend(points2D_list)
            except AttributeError:
                # 极少数情况可能需要 Point2DVector
                try:
                    image.points2D = pycolmap.Point2DVector(points2D_list)
                except AttributeError:
                    print(f"Warning: Could not set points2D for frame {fidx}")

        # 注册状态
        # 3.13+ 中 registered 属性可能是只读的（由位姿决定），尝试赋值但不强求
        try:
            image.registered = (len(points2D_list) > 0)
        except AttributeError:
            pass

        # 添加图像并关联位姿
        if hasattr(reconstruction, "add_image_with_trivial_frame"):
            reconstruction.add_image_with_trivial_frame(image, cam_from_world)
        else:
            try:
                image.cam_from_world = cam_from_world
            except:
                pass
            reconstruction.add_image(image)

    return reconstruction


# ==============================================================================
# 主逻辑
# ==============================================================================
def run_conversion(args):
    # 1. 初始化模型
    print('loading sailrecon model....')
    model = SailRecon(kv_cache=True)

    ckpt_path = args.ckpt if args.ckpt else "ckpt/sailrecon.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f'权重文件加载成功: {ckpt_path}')
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}, trying huggingface url...")
        _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model = model.to(device=device)
    model.eval()

    # 2. 加载与预处理图像
    if not os.path.exists(args.img_dir):
        raise ValueError(f"Image directory not found: {args.img_dir}")

    image_names = sorted([
        os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(image_names) == 0:
        raise ValueError(f"No images found in {args.img_dir}")

    print(f"Processing {len(image_names)} images...")
    images = load_and_preprocess_images(image_names).to(device)
    _, _, H, W = images.shape
    image_size = np.array([W, H])

    # 3. 执行推理
    select_indices = uniform_sample(len(image_names), min(100, len(image_names)))
    anchor_images = images[select_indices]
    predictions = []

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            print("Building scene representation (Anchor Images)...")
            model.tmp_forward(anchor_images)
            del model.aggregator.global_blocks

            print("Relocalizing all frames...")
            chunk_size = 10
            for i in tqdm(range(0, len(images), chunk_size)):
                img_split = images[i: i + chunk_size]
                preds = model.reloc(img_split, memory_save=False, save_depth=True, ret_img=True)
                predictions += to_cpu(preds)

    # 4. 准备数据
    print("Preparing data for COLMAP conversion...")
    all_points3d = []
    all_points_xyf = []
    all_points_rgb = []
    extrinsics_list = []
    intrinsics_list = []

    for frame_idx, pred in enumerate(tqdm(predictions, desc="Extracting points")):
        ext = pred['extrinsic'][0].cpu().numpy().astype(np.float64)
        extrinsics_list.append(ext)
        intr = pred['intrinsic'][0].cpu().numpy().astype(np.float64)
        intrinsics_list.append(intr)

        pts = pred['point_map'][0].cpu().numpy()
        cnf = pred['xyz_cnf'][0].cpu().numpy()

        img_tensor = pred['images'][0].permute(1, 2, 0).cpu()
        clr = (img_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        mask = cnf > args.conf_thres
        valid_h, valid_w = np.nonzero(mask)

        all_points3d.append(pts[mask])
        all_points_rgb.append(clr[mask])
        all_points_xyf.append(np.stack([valid_w, valid_h, np.full_like(valid_h, frame_idx)], axis=1))

    all_points3d = np.concatenate(all_points3d, axis=0)
    all_points_rgb = np.concatenate(all_points_rgb, axis=0)
    all_points_xyf = np.concatenate(all_points_xyf, axis=0)
    all_extrinsics = np.stack(extrinsics_list)
    all_intrinsics = np.stack(intrinsics_list)

    print(f"Total points extracting: {len(all_points3d)}")
    print("Constructing pycolmap reconstruction...")

    # 5. 调用修复后的转换函数
    reconstruction = fixed_batch_np_matrix_to_pycolmap_wo_track(
        points3d=all_points3d,
        points_xyf=all_points_xyf,
        points_rgb=all_points_rgb,
        extrinsics=all_extrinsics,
        intrinsics=all_intrinsics,
        image_size=image_size,
        camera_type="PINHOLE"
    )

    # 6. 保存输出
    out_path = os.path.join(args.out_dir, os.path.basename(args.img_dir.strip("/")))
    colmap_path = os.path.join(out_path, "sparse", "0")
    images_path = os.path.join(out_path, "images")

    os.makedirs(colmap_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    print(f"Saving COLMAP model to {colmap_path}...")
    reconstruction.write(colmap_path)

    print(f"Saving preprocessed images to {images_path}...")
    for i, pred in enumerate(predictions):
        img_tensor = pred['images'][0].permute(1, 2, 0).cpu()
        img_np = (img_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        img_name = f"image_{i + 1}.png"
        img_pil.save(os.path.join(images_path, img_name))

    print("\nConversion Complete!")
    print(f"Output directory: {out_path}")
    print("You can now run gsplat using:")
    print(f"gsplat-train --colmap_dir {out_path} --image_dir {images_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="img_set/test", help="Input images folder")
    parser.add_argument("--out_dir", type=str, default="outputs_colmap", help="Output folder")
    parser.add_argument("--ckpt", type=str, default="ckpt/sailrecon.pt", help="Model checkpoint path")
    parser.add_argument("--conf_thres", type=float, default=2.0, help="Confidence threshold")
    args = parser.parse_args()

    run_conversion(args)