# -----------------------------------------------------------------------------
# Main.py - 实景三维模型体素化及3D Tiles (B3DM) 生成脚本
#
# 功能:
# 1. 加载三维模型文件 (支持网格和点云 .ply)。
# 2. 对模型进行体素化处理。
# 3. 生成多个层次细节 (LOD)。
# 4. 将每个LOD的体素打包成 B3DM 格式的瓦片。
# 5. 生成一个包含地理位置信息的 tileset.json 文件。
#
# 依赖:
# pip install trimesh numpy open3d py3dtiles pygltflib scipy
# -----------------------------------------------------------------------------

import trimesh
import numpy as np
import open3d as o3d
import json
import os
import shutil
import traceback
from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tileset import TileSet
from py3dtiles.tileset.content.b3dm import B3dm, B3dmHeader
import pygltflib

# --- 全局配置参数 ---
DEBUG = True  # 设置为 True 以启用详细的调试输出

# --- 输入与输出配置 ---
MODEL_FILE_PATH = r"D:\Datasets\PCLAsset\point_cloud.ply"  # 输入模型文件路径 (.ply, .obj, .stl等)
OUTPUT_DIR = "output_3dtiles_b3dm"  # 输出3D Tiles数据集的目录

# --- 地理定位配置 ---
GEOLOCATION_ENABLED = False  # 是否启用地理定位
# 武汉大学附近某点
TARGET_LONGITUDE = 114.36
TARGET_LATITUDE = 30.54
TARGET_HEIGHT = 50.0  # 相对于WGS84椭球的高度(米)

# --- 体素化与LOD配置 ---
HIGHEST_LOD_PITCH = 1.0  # 最高细节LOD的体素边长(米)
LOD_LEVELS = 3  # 要生成的LOD层级总数
# 每个LOD相对于最高LOD的降采样聚合因子 (例如, 2表示2x2x2个体素合并)
LOD_DOWNSAMPLE_FACTORS = [1, 2, 4]

# --- 3D Tiles 配置 ---
# 根瓦片的几何误差，通常是场景尺寸的一个比例
# geometricError 越大，瓦片在更远处被加载（更粗糙的LOD）
GEOMETRIC_ERROR_BASE = 500
# 每深入一层LOD，geometricError 的衰减因子
GEOMETRIC_ERROR_LOD_FACTOR = 0.5


def load_and_preprocess_model(file_path):
    """
    加载并预处理三维模型。
    支持直接加载网格，或将点云转换为 trimesh.PointCloud 对象。
    """
    print("--- 1. 加载与预处理模型 ---")
    try:
        # 尝试使用 trimesh 直接加载
        mesh_object = trimesh.load(file_path)
        if isinstance(mesh_object, trimesh.Trimesh) and mesh_object.vertices.size > 0:
            print(f"成功加载为网格模型 (Trimesh)，顶点数: {len(mesh_object.vertices)}")
            # 对网格模型进行预处理
            if not mesh_object.is_watertight:
                print("警告: 网格模型非水密，尝试修复...")
                mesh_object.fill_holes()
            return mesh_object
        elif isinstance(mesh_object, trimesh.PointCloud) and mesh_object.vertices.size > 0:
            print(f"成功加载为点云模型 (trimesh.PointCloud)，顶点数: {len(mesh_object.vertices)}")
            return mesh_object
        else:
            # 如果 trimesh 加载失败或为空，尝试使用 Open3D 作为备选方案加载点云
            print("Trimesh 加载结果为空或非支持类型，尝试使用 Open3D...")
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                raise ValueError("Open3D 加载点云失败或点云为空。")
            points = np.asarray(pcd.points)
            point_cloud = trimesh.PointCloud(points)
            print(f"使用 Open3D 成功加载点云，顶点数: {len(point_cloud.vertices)}")
            return point_cloud
    except Exception as e:
        print(f"加载模型失败: {e}")
        traceback.print_exc()
        return None


def voxelize_model(model, pitch):
    """
    将输入的 trimesh 对象 (网格或点云) 进行体素化。
    返回一个 trimesh.voxel.VoxelGrid 对象。
    """
    print(f"\n--- 2. 执行体素化 (Pitch: {pitch}) ---")
    if isinstance(model, trimesh.Trimesh):
        # 对于网格模型，可以直接调用 .voxelized 方法
        print("对网格模型进行体素化...")
        return model.voxelized(pitch=pitch)
        
    elif isinstance(model, trimesh.PointCloud):
        # 对于点云模型，需要手动创建 VoxelGrid (恢复您原来的正确方法)
        print("对点云模型进行体素化...")
        if len(model.vertices) == 0:
            print("错误：点云对象不包含顶点。")
            return None

        points = model.vertices
        
        # 1. 将点坐标转换为离散的体素索引
        # 使用 np.floor 比 np.round 更能保证体素对齐的稳定性
        discrete_indices = np.floor(points / pitch).astype(int)
        
        # 2. 找到所有被占据的体素的唯一索引
        unique_indices = np.unique(discrete_indices, axis=0)
        
        if unique_indices.shape[0] == 0:
            print("警告：根据PITCH参数，点云未占据任何体素单元。")
            # 返回一个空的 VoxelGrid
            return trimesh.voxel.VoxelGrid(
                trimesh.voxel.encoding.SparseBinaryEncoding(np.empty((0, 3), dtype=int)),
                trimesh.transformations.scale_and_translate(scale=pitch)
            )

        # 3. 计算编码的原点和相对索引
        origin_index = unique_indices.min(axis=0)
        sparse_relative_indices = unique_indices - origin_index
        encoding_shape = sparse_relative_indices.max(axis=0) + 1

        # 4. 创建稀疏二进制编码对象
        voxel_encoding = trimesh.voxel.encoding.SparseBinaryEncoding(
            sparse_relative_indices,
            shape=encoding_shape
        )

        # 5. 计算从0基体素索引到世界坐标的变换矩阵
        transform_translation = origin_index * pitch
        voxel_transform = trimesh.transformations.scale_and_translate(
            scale=pitch,
            translate=transform_translation
        )

        # 6. 创建并返回 VoxelGrid 对象
        return trimesh.voxel.VoxelGrid(voxel_encoding, voxel_transform)
        
    else:
        print(f"错误：未知的模型类型 {type(model)}。")
        return None


def generate_lods(voxel_grid, num_levels, downsample_factors, base_pitch):
    """
    从最高细节的体素网格生成多个LOD层级的体素中心点。
    采用可靠的数学方法对点云中心进行降采样，不再使用 .voxelized()。
    """
    print("\n--- 3. 生成多层次细节 (LOD) ---")
    if not voxel_grid or voxel_grid.is_empty:
        print("输入的体素网格为空，无法生成LOD。")
        return None

    all_lods_data = {}
    lod0_centers = voxel_grid.points
    
    if lod0_centers.shape[0] == 0:
        print("LOD 0 没有体素，无法生成后续LOD。")
        return None

    # LOD 0 直接使用体素化的结果
    all_lods_data[0] = {
        'centers': lod0_centers,
        'pitch': base_pitch 
    }
    print(f"LOD 0: {len(lod0_centers)} 个体素, Pitch: {base_pitch:.2f}")

    # 通过降采样生成更低细节的LOD
    for i in range(1, num_levels):
        factor = downsample_factors[i]
        prev_lod_centers = all_lods_data[i-1]['centers']
        new_pitch = base_pitch * factor

        if len(prev_lod_centers) == 0:
            print(f"LOD {i}: 上一级LOD为空，跳过。")
            all_lods_data[i] = {'centers': np.array([]), 'pitch': new_pitch}
            continue

        # --- 开始决定性的修正：手动降采样点云 ---
        # 1. 计算每个点所属的更大、更粗糙的体素的索引
        coarse_voxel_indices = np.floor(prev_lod_centers / new_pitch).astype(int)

        # 2. 找到唯一的粗糙体素索引，并获取每个原始点对应的组ID
        unique_coarse_indices, inverse_indices = np.unique(
            coarse_voxel_indices, axis=0, return_inverse=True
        )

        # 3. 为每个唯一的粗糙体素组计算一个新的中心点（通过平均值）
        new_lod_centers = []
        # 遍历每一个唯一的组 (j 是从 0 到 组数-1 的索引)
        for j in range(len(unique_coarse_indices)):
            # 找到所有属于当前组 j 的原始中心点
            points_in_group = prev_lod_centers[inverse_indices == j]
            # 计算这些点的平均值，作为新LOD的体素中心
            new_center = np.mean(points_in_group, axis=0)
            new_lod_centers.append(new_center)
        
        # 将结果存入字典
        all_lods_data[i] = {
            'centers': np.array(new_lod_centers) if new_lod_centers else np.array([]),
            'pitch': new_pitch
        }
        # --- 修正结束 ---

        print(f"LOD {i}: {len(all_lods_data[i]['centers'])} 个体素, Pitch: {new_pitch:.2f}")

    return all_lods_data


def create_b3dm_content(voxel_centers, pitch):
    """
    为给定的体素中心点和大小创建一个 B3DM 瓦片内容。
    将所有体素（立方体）合并成一个大的网格，并导出为 GLB。
    返回一个 py3dtiles 的 B3dm 对象。
    """
    if len(voxel_centers) == 0:
        return None

    # 为每个体素中心点创建一个立方体网格
    list_of_cube_meshes = []
    for center_point in voxel_centers:
        # trimesh.creation.box 默认中心在原点，我们需要移动它
        cube_mesh = trimesh.creation.box(extents=[pitch, pitch, pitch])
        cube_mesh.apply_translation(center_point)
        list_of_cube_meshes.append(cube_mesh)

    # 合并所有小立方体为一个大网格
    if not list_of_cube_meshes:
        return None
    merged_mesh = trimesh.util.concatenate(list_of_cube_meshes)

    # 导出合并后的网格为 GLB 字节流
    glb_body_bytes = merged_mesh.export(file_type='glb')
    if not glb_body_bytes:
        print("错误: 导出的 GLB 内容为空。")
        return None

    # 使用 pygltflib 加载 GLB 字节，以便 py3dtiles 处理
    gltf_asset = pygltflib.GLTF2.load_from_bytes(glb_body_bytes)

    # 创建 B3DM 内容对象
    # 对于简单的 B3DM，我们不需要复杂的批处理表
    b3dm_content = B3dm.from_gltf(gltf=gltf_asset)
    
    return b3dm_content, merged_mesh


def fix_numpy_types_in_dict(d):
    """
    递归地将字典中所有Numpy数值类型转换为标准的Python类型，以便JSON序列化。
    """
    if isinstance(d, dict):
        return {k: fix_numpy_types_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [fix_numpy_types_in_dict(i) for i in d]
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating):
        return float(d)
    return d


def main():
    """
    主执行函数 (已重构瓦片创建逻辑以确保健壮性)
    """
    # 0. 准备工作
    if os.path.exists(OUTPUT_DIR):
        print(f"输出目录 '{OUTPUT_DIR}' 已存在，正在清理...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"已创建输出目录: '{OUTPUT_DIR}'")

    # 1. 加载与预处理
    model = load_and_preprocess_model(MODEL_FILE_PATH)
    if model is None:
        return

    # 2. 体素化
    voxel_grid = voxelize_model(model, HIGHEST_LOD_PITCH)
    if voxel_grid is None or voxel_grid.is_empty:
        print("体素化失败或结果为空，程序终止。")
        return

    # 3. 生成LODs
    all_lods_data = generate_lods(voxel_grid, LOD_LEVELS, LOD_DOWNSAMPLE_FACTORS, HIGHEST_LOD_PITCH)
    if all_lods_data is None:
        print("LOD 生成失败，程序终止。")
        return

    # -------------------------------------------------------------------------
    # --- 关键修正: 重构 3D Tiles 数据集创建逻辑 ---
    # -------------------------------------------------------------------------
    print("\n--- 4. 创建 3D Tiles 数据集 (重构后逻辑) ---")
    
    # 4.1 初始化 Tileset 对象
    ts = TileSet()
    ts.asset = {'version': '1.0', 'tilesetVersion': '1.0.0-my-voxel-b3dm'}
    ts.geometric_error = float(GEOMETRIC_ERROR_BASE) # 使用 snake_case 并确保 float

    # 4.2 创建一个纯粹的、无内容的根瓦片作为容器
    root_tile = Tile()
    
    # 计算并设置根瓦片的包围盒
    lod0_centers = all_lods_data[0]['centers']
    lod0_pitch = all_lods_data[0]['pitch']
    if len(lod0_centers) == 0:
        print("错误：LOD0 没有体素中心，无法计算根包围盒。将使用默认包围盒。")
        root_tile.bounding_volume = {'box': [0.0,0.0,0.0, 1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]}
    else:
        min_bound = np.min(lod0_centers - lod0_pitch / 2, axis=0)
        max_bound = np.max(lod0_centers + lod0_pitch / 2, axis=0)
        center = (min_bound + max_bound) / 2
        half_size = (max_bound - min_bound) / 2
        root_tile.bounding_volume = {'box': [ # 使用 snake_case 并确保 float
            float(center[0]), float(center[1]), float(center[2]),
            float(half_size[0]), 0.0, 0.0, 
            0.0, float(half_size[1]), 0.0, 
            0.0, 0.0, float(half_size[2])
        ]}
    
    root_tile.refine = 'REPLACE' # 使用 snake_case
    ts.root = root_tile

    # 4.3 为瓦片集设置地理定位
    if GEOLOCATION_ENABLED:
        print(f"启用地理定位，锚点: LLA({TARGET_LONGITUDE}, {TARGET_LATITUDE}, {TARGET_HEIGHT})")
        try:
            transform_matrix = trimesh.transformations.east_north_up_to_ecef(
                longitude=TARGET_LONGITUDE, latitude=TARGET_LATITUDE, height=TARGET_HEIGHT
            )
            # ts.root.transform is already snake_case and expects a list of floats
            ts.root.transform = [float(x) for x in transform_matrix.flatten('F').tolist()]
        except Exception as e:
            print(f"错误: 计算地理变换矩阵时失败: {e}")

    # 4.4 循环为每个LOD创建带内容的瓦片，并建立父子链接
    parent_tile = ts.root  # 从根瓦片开始，作为第一个父节点
    tile_counter = 0

    for lod_level in range(LOD_LEVELS):
        lod_data = all_lods_data[lod_level]
        voxel_centers = lod_data['centers']
        pitch = lod_data['pitch']
        
        if len(voxel_centers) == 0:
            print(f"LOD {lod_level} 没有体素，跳过。")
            continue

        print(f"正在为 LOD {lod_level} 创建瓦片...")
        
        # A. 创建B3DM内容文件
        b3dm_content, merged_mesh = create_b3dm_content(voxel_centers, pitch)
        if b3dm_content is None:
            print(f"  LOD {lod_level} 的 B3DM 内容创建失败，跳过此瓦片。")
            continue
        b3dm_filename = f"tile_lod{lod_level}_{tile_counter}.b3dm"
        b3dm_filepath = os.path.join(OUTPUT_DIR, b3dm_filename)
        with open(b3dm_filepath, 'wb') as f:
            f.write(b3dm_content.to_array())
        print(f"  已生成内容文件: {b3dm_filename}")
        tile_counter += 1

        # B. 创建一个新的瓦片对象来承载这个LOD
        lod_tile = Tile()
        
        # C. 为这个新瓦片设置所有必需的属性
        lod_tile.content = {'uri': b3dm_filename}
        lod_tile.geometric_error = float(ts.geometric_error * (GEOMETRIC_ERROR_LOD_FACTOR ** (lod_level + 1))) # 使用 snake_case 并确保 float
        
        # 使用当前LOD合并后网格的精确边界作为其包围盒
        min_b, max_b = merged_mesh.bounds
        c = (min_b + max_b) / 2
        hs = (max_b - min_b) / 2
        lod_tile.bounding_volume = {'box': [ # 使用 snake_case 并确保 float
            float(c[0]), float(c[1]), float(c[2]), 
            float(hs[0]), 0.0, 0.0, 
            0.0, float(hs[1]), 0.0, 
            0.0, 0.0, float(hs[2])
        ]}
        print(f"  已设置包围盒和几何误差。")

        # D. 将这个属性完备的瓦片链接到其父瓦片
        if parent_tile.children is None:
            parent_tile.children = []
        parent_tile.children.append(lod_tile)
        print(f"  已将 LOD {lod_level} 瓦片链接到上一级。")

        # E. 如果这不是最后一个LOD，那么它也需要作为下一个LOD的父节点
        if lod_level < LOD_LEVELS - 1:
            lod_tile.refine = 'REPLACE'
            # 更新指针，让当前瓦片成为下一次循环的父瓦片
            parent_tile = lod_tile
            
    # --- 重构结束 ---

    print("\n--- 5. 保存 tileset.json ---")
    try:
        # 现在调用 to_dict() 应该是安全的
        tileset_dict = ts.to_dict()
        # 最后的清理，以防万一
        cleaned_tileset_dict = fix_numpy_types_in_dict(tileset_dict)
        
        tileset_json_path = os.path.join(OUTPUT_DIR, "tileset.json")
        with open(tileset_json_path, 'w') as f:
            json.dump(cleaned_tileset_dict, f, indent=4)
        print(f"Tileset JSON 已成功保存到: {tileset_json_path}")

    except Exception as e:
        print(f"保存 tileset.json 时出错: {e}")
        traceback.print_exc()

    print(f"\n处理完成！3D Tiles 数据集位于目录: '{OUTPUT_DIR}'")


if __name__ == '__main__':
    main()