import argparse
import os
import numpy as np
from PIL import Image
from plyfile import PlyData

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(img_dir):
    names = [f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]
    names.sort()
    return [os.path.join(img_dir, f) for f in names]


def load_poses_txt(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] == 12:
        poses = data.reshape(-1, 3, 4)
    elif data.shape[1] == 16:
        poses = data.reshape(-1, 4, 4)[:, :3, :]
    else:
        raise ValueError(
            f"Unsupported pose format: expected 12 or 16 values per line, got {data.shape[1]}"
        )
    return poses.astype(np.float64)


def invert_c2w_to_w2c(c2w):
    w2c = np.zeros_like(c2w)
    for i in range(c2w.shape[0]):
        R = c2w[i, :3, :3]
        t = c2w[i, :3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        w2c[i, :3, :3] = R_inv
        w2c[i, :3, 3] = t_inv
    return w2c


def rotmat_to_qvec(R):
    qvec = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qvec[0] = 0.25 / s
        qvec[1] = (R[2, 1] - R[1, 2]) * s
        qvec[2] = (R[0, 2] - R[2, 0]) * s
        qvec[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qvec[0] = (R[2, 1] - R[1, 2]) / s
            qvec[1] = 0.25 * s
            qvec[2] = (R[0, 1] + R[1, 0]) / s
            qvec[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qvec[0] = (R[0, 2] - R[2, 0]) / s
            qvec[1] = (R[0, 1] + R[1, 0]) / s
            qvec[2] = 0.25 * s
            qvec[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qvec[0] = (R[1, 0] - R[0, 1]) / s
            qvec[1] = (R[0, 2] + R[2, 0]) / s
            qvec[2] = (R[1, 2] + R[2, 1]) / s
            qvec[3] = 0.25 * s
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def load_intrinsics_txt(path):
    intrinsics = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if len(values) == 9:
                K = np.array(values, dtype=np.float64).reshape(3, 3)
            elif len(values) == 4:
                fx, fy, cx, cy = values
                K = np.array(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            else:
                raise ValueError(
                    f"Unsupported intrinsics format: {len(values)} values per line"
                )
            intrinsics.append(K)
    if not intrinsics:
        raise ValueError(f"No intrinsics found in {path}")
    return intrinsics


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def compute_preprocess_plan(image_paths, preprocess_mode, target_size=980):
    if preprocess_mode == "pad":
        return target_size, target_size

    max_width = target_size
    max_height = 0
    for img_path in image_paths:
        with Image.open(img_path) as img:
            width, height = img.size
        new_height = round(height * (target_size / width) / 14) * 14
        final_height = target_size if new_height > target_size else new_height
        max_height = max(max_height, final_height)
    return max_width, max_height


def pad_to_size(img, target_width, target_height, color=(255, 255, 255)):
    pad_w = target_width - img.width
    pad_h = target_height - img.height
    if pad_w < 0 or pad_h < 0:
        raise ValueError("Target size must be larger than image size.")
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    new_img = Image.new("RGB", (target_width, target_height), color)
    new_img.paste(img, (pad_left, pad_top))
    return new_img


def preprocess_image(img, preprocess_mode, target_size, pad_width, pad_height):
    width, height = img.size
    if preprocess_mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    if preprocess_mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img.crop((0, start_y, new_width, start_y + target_size))

    if preprocess_mode == "pad":
        img = pad_to_size(img, target_size, target_size)
    else:
        img = pad_to_size(img, pad_width, pad_height)

    return img


def save_images(image_paths, out_dir, mode, preprocess_mode, target_size):
    ensure_dir(out_dir)
    image_names = []
    image_sizes = []
    if mode == "preprocess":
        pad_width, pad_height = compute_preprocess_plan(
            image_paths, preprocess_mode, target_size
        )
    for idx, img_path in enumerate(image_paths):
        out_name = f"image_{idx + 1}.png"
        out_path = os.path.join(out_dir, out_name)
        if mode == "preprocess":
            with Image.open(img_path) as img:
                if img.mode == "RGBA":
                    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(background, img)
                img = img.convert("RGB")
                img = preprocess_image(
                    img, preprocess_mode, 518, pad_width, pad_height
                )
                img.save(out_path)
                width, height = img.size
        elif mode == "original":
            with Image.open(img_path) as img:
                if img.mode == "RGBA":
                    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(background, img)
                img = img.convert("RGB")
                img.save(out_path)
                width, height = img.size
        else:
            raise ValueError(f"Unsupported image mode: {mode}")
        image_names.append(out_name)
        image_sizes.append((width, height))
    return image_names, image_sizes


def resolve_intrinsics(
    image_sizes,
    intrinsics_txt=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    fov_deg=None,
):
    if intrinsics_txt:
        Ks = load_intrinsics_txt(intrinsics_txt)
        if len(Ks) == 1 and len(image_sizes) > 1:
            Ks = Ks * len(image_sizes)
        if len(Ks) != len(image_sizes):
            raise ValueError(
                "Intrinsics count does not match number of images."
            )
        return Ks

    if fx is None and fov_deg is None:
        raise ValueError(
            "Provide intrinsics via --intrinsics_txt, --fx/--fy, or --fov_deg."
        )

    Ks = []
    for width, height in image_sizes:
        if fov_deg is not None:
            fov_rad = np.deg2rad(fov_deg)
            fx_val = (width / 2.0) / np.tan(fov_rad / 2.0)
            fy_val = (height / 2.0) / np.tan(fov_rad / 2.0)
        else:
            fx_val = fx
            fy_val = fy if fy is not None else fx
        cx_val = cx if cx is not None else width / 2.0
        cy_val = cy if cy is not None else height / 2.0
        K = np.array(
            [[fx_val, 0.0, cx_val], [0.0, fy_val, cy_val], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Ks.append(K)
    return Ks


def load_ply_xyz_rgb(path):
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    if {"red", "green", "blue"}.issubset(vertices.data.dtype.names):
        rgb = np.stack(
            [vertices["red"], vertices["green"], vertices["blue"]], axis=1
        )
    else:
        rgb = np.zeros_like(xyz, dtype=np.uint8)
    return xyz.astype(np.float64), rgb.astype(np.uint8)


def write_cameras_txt(path, Ks, image_sizes):
    with open(path, "w") as f:
        f.write(
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"# Number of cameras: {len(Ks)}\n"
        )
        for i, (K, (width, height)) in enumerate(zip(Ks, image_sizes), start=1):
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            f.write(
                f"{i} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n"
            )


def write_images_txt(path, w2c, image_names):
    with open(path, "w") as f:
        f.write(
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(image_names)}, mean observations per image: 0\n"
        )
        for i, name in enumerate(image_names, start=1):
            R = w2c[i - 1, :3, :3]
            t = w2c[i - 1, :3, 3]
            qvec = rotmat_to_qvec(R)
            f.write(
                f"{i} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{t[0]} {t[1]} {t[2]} {i} {name}\n"
            )
            f.write("\n")


def write_points3d_txt(path, xyz, rgb):
    with open(path, "w") as f:
        f.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            f"# Number of points: {len(xyz)}, mean track length: 0\n"
        )
        for i, (pt, col) in enumerate(zip(xyz, rgb), start=1):
            f.write(
                f"{i} {pt[0]} {pt[1]} {pt[2]} "
                f"{int(col[0])} {int(col[1])} {int(col[2])} 0\n"
            )


def run(args):
    image_paths = list_images(args.img_dir)
    if not image_paths:
        raise ValueError(f"No images found in {args.img_dir}")

    poses = load_poses_txt(args.pose_path)
    if len(poses) != len(image_paths):
        raise ValueError(
            f"Pose count {len(poses)} does not match image count {len(image_paths)}"
        )

    if args.pose_format == "c2w":
        w2c = invert_c2w_to_w2c(poses)
    else:
        w2c = poses

    out_scene = os.path.join(args.out_dir, args.scene_name)
    images_out = os.path.join(out_scene, "images")
    sparse_out = os.path.join(out_scene, "sparse", "0")
    ensure_dir(images_out)
    ensure_dir(sparse_out)

    image_names, image_sizes = save_images(
        image_paths, images_out, args.image_mode, args.preprocess_mode
    )

    Ks = resolve_intrinsics(
        image_sizes,
        intrinsics_txt=args.intrinsics_txt,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        fov_deg=args.fov_deg,
    )

    xyz, rgb = load_ply_xyz_rgb(args.ply_path)

    write_cameras_txt(os.path.join(sparse_out, "cameras.txt"), Ks, image_sizes)
    write_images_txt(os.path.join(sparse_out, "images.txt"), w2c, image_names)
    write_points3d_txt(os.path.join(sparse_out, "points3D.txt"), xyz, rgb)

    print("Conversion complete.")
    print(f"COLMAP directory: {out_scene}")
    print(f"Images directory: {images_out}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert SAIL-Recon outputs (pred.txt + pred.ply) to COLMAP text format."
    )
    parser.add_argument("--img_dir", type=str, default="/root/autodl-tmp/truck/images", help="Input image directory")
    parser.add_argument("--pose_path", default="/root/autodl-tmp/outputs_truck/truck/pred.txt", help="Path to pred.txt poses")
    parser.add_argument("--ply_path", default="/root/autodl-tmp/outputs_truck/truck/pred.ply", help="Path to pred.ply point cloud")
    parser.add_argument("--intrinsics_txt", default="/root/autodl-tmp/outputs_truck/truck/intrinsics.txt", help="Path to intrinsics_txt")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/outputs_colmap", help="Output root")
    parser.add_argument("--scene_name", default="truck", help="Output scene folder name")
    parser.add_argument(
        "--pose_format",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Pose format in pred.txt (demo.py saves c2w)",
    )
    parser.add_argument(
        "--image_mode",
        choices=["preprocess", "original"],
        default="preprocess",
        help="preprocess uses SAIL-Recon resizing; original keeps native resolution",
    )
    parser.add_argument(
        "--preprocess_mode",
        choices=["crop", "pad"],
        default="crop",
        help="Preprocess mode for SAIL-Recon image resizing",
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=980,
        help="Target image size."
    )

    parser.add_argument("--fx", type=float, default=None, help="Focal length in x")
    parser.add_argument("--fy", type=float, default=None, help="Focal length in y")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y")
    parser.add_argument(
        "--fov_deg",
        type=float,
        default=None,
        help="Fallback symmetric FOV in degrees if fx/fy not provided",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
