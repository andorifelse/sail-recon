import open3d as o3d
import numpy as np

# -----------------------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------------------
FILE_PATH = "outputs_Barn/Barn/pred.ply"  # 你的 PLY 文件路径
ENABLE_FILTER = True  # 是否开启去噪 (推荐 True)
POINT_SIZE = 2.0  # 点的大小
BACKGROUND_COLOR = [0, 0, 0]  # 背景颜色 [R, G, B] (0-1)


# -----------------------------------------------------------------------------

def main():
    print(f"Loading {FILE_PATH}...")
    try:
        pcd = o3d.io.read_point_cloud(FILE_PATH)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if not pcd.has_points():
        print("Error: The point cloud is empty.")
        return

    print(f"Original points: {len(pcd.points)}")
    print(f"Has colors: {pcd.has_colors()}")

    # 1. 坐标系修正 (可选)
    # 很多深度学习模型输出的点云是上下颠倒的，或者 Y 轴朝下
    # 如果你发现建筑是倒着的，可以取消下面这行的注释来翻转
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 2. 去噪处理 (核心改进)
    if ENABLE_FILTER:
        print("Applying statistical outlier removal (cleaning noise)...")
        # nb_neighbors: 考虑每个点周围的 50 个邻居
        # std_ratio: 如果一个点到邻居的平均距离超过 1.0 倍标准差，就被认为是噪点
        # (对于天空噪点，通常它们很稀疏，这个方法非常有效)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

        # 仅保留内点 (Inliers)
        pcd_clean = pcd.select_by_index(ind)
        print(f"Filtered points: {len(pcd_clean.points)} (Removed {len(pcd.points) - len(pcd_clean.points)} points)")
        pcd_to_show = pcd_clean
    else:
        pcd_to_show = pcd

    # 3. 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Sail-Recon Result (Press 'R' to Reset View)", width=1280, height=720)

    # 添加几何体
    vis.add_geometry(pcd_to_show)

    # 4. 渲染设置
    opt = vis.get_render_option()
    opt.background_color = np.asarray(BACKGROUND_COLOR)
    opt.point_size = POINT_SIZE

    # 开启光照关闭模式 (让纯点云颜色更鲜艳，防止黑屏)
    # Open3D 的 light_on 默认为 True，对于没有法线的点云，关掉效果更好
    opt.light_on = False

    print("\n---------------------------------------")
    print("Controls:")
    print("  [Mouse] Rotate/Pan/Zoom")
    print("  [R]     Reset View")
    print("  [+/-]   Increase/Decrease Point Size")
    print("  [Q]     Quit")
    print("---------------------------------------")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()