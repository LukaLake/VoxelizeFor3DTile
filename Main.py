import trimesh
import numpy as np

# --- 输入参数 ---
MODEL_FILE_PATH = "path/to/your/model.obj"  # 或 .stl, .ply, .glb 等trimesh支持的格式
# 对于点云，可以使用 open3d 加载并转换为 trimesh.PointCloud 对象

# --- 加载模型 ---
try:
    mesh = trimesh.load_mesh(MODEL_FILE_PATH)
    print(f"模型 '{MODEL_FILE_PATH}' 加载成功.")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

# --- 预处理 (示例) ---
# 1. 确保模型是水密的 (如果体素化算法需要)
if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
    print("警告: 模型非水密，尝试修复...")
    mesh.fill_holes()
    # mesh.fix_normals() # 可能也需要

# 2. 坐标系对齐 (根据实际需求调整)
# 例如，如果需要将Z轴朝上，而模型是Y轴朝上：
# rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
# mesh.apply_transform(rotation_matrix)

# 3. 将模型移动到原点附近 (可选，方便后续处理)
# mesh.apply_translation(-mesh.bounds[0]) # 将最小边界点移到原点
# mesh.apply_translation(-mesh.centroid) # 将质心移到原点

# 4. 统一单位 (如果需要)
# target_scale_factor = ...
# mesh.apply_scale(target_scale_factor)

print(f"模型包围盒 (预处理后): {mesh.bounds}")
print(f"模型顶点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")


# --- 体素化参数 ---
# PITCH 决定了最高LOD的体素大小 (立方体边长)
# 例如，如果模型单位是米，PITCH = 1.0 表示1立方米的体素
HIGHEST_LOD_PITCH = 1.0  # 体素边长，也是最高LOD的分辨率

# --- 执行体素化 ---
# `voxelized` 方法返回一个 VoxelGrid 对象
# `pitch` 参数是体素的边长
voxel_grid = mesh.voxelized(pitch=HIGHEST_LOD_PITCH)
print(f"体素化完成. 体素网格形状: {voxel_grid.shape}")

# VoxelGrid.encoding.dense_indices 包含了所有被占据体素的整数索引 (i, j, k)
# VoxelGrid.transform 是从体素索引到世界坐标的变换矩阵
occupied_voxel_indices = voxel_grid.encoding.dense_indices # (N, 3) array of int indices
num_voxels_lod0 = len(occupied_voxel_indices)
print(f"LOD 0 (最高细节) 占据的体素数量: {num_voxels_lod0}")

if num_voxels_lod0 == 0:
    print("没有生成任何体素，请检查模型或PITCH参数。")
    exit()

# --- 将体素索引转换为世界坐标中心点 ---
# 体素的中心点 = transform @ (indices + 0.5).T
# voxel_grid.points 已经为我们计算好了这些中心点
voxel_centers_lod0_world = voxel_grid.points # (N, 3) array of world coordinates
print(f"LOD 0 体素中心点 (世界坐标) 示例 (前5个): \n{voxel_centers_lod0_world[:5]}")

# (可选) 提取体素颜色
# 如果需要，可以从原始网格的最近表面点采样颜色
# 这部分相对复杂，trimesh 本身不直接提供 voxel-to-color 的简单映射
# 可能需要结合 trimesh.proximity.ProximityQuery
# voxel_colors_lod0 = [] # 存储颜色信息

# --- LOD组织参数 ---
LOD_LEVELS = 3  # 例如，LOD 0, LOD 1, LOD 2
LOD_DOWNSAMPLE_FACTORS = [1, 2, 4] # LOD 0是原始，LOD 1每2x2x2个体素合并，LOD 2每4x4x4个体素合并
                                 # 确保因子是2的幂次方，便于八叉树逻辑

all_lods_voxel_centers = {} # 字典，键是LOD级别，值是该LOD的体素中心点列表

# --- LOD 0 ---
all_lods_voxel_centers[0] = voxel_centers_lod0_world
print(f"LOD 0 中心点数量: {len(all_lods_voxel_centers[0])}")

# --- 生成更低细节的LOD (通过降采样体素中心点) ---
# 这是一个简化的降采样示例，实际应用中可能需要更精细的八叉树聚合
# VoxelGrid 的 transform 和 pitch 对于理解坐标很重要
origin = voxel_grid.transform[:3, 3] # 体素网格的原点
current_pitch = voxel_grid.pitch       # 当前LOD的体素大小

for i in range(1, LOD_LEVELS):
    factor = LOD_DOWNSAMPLE_FACTORS[i]
    prev_lod_centers = all_lods_voxel_centers[i-1]

    if not len(prev_lod_centers):
        all_lods_voxel_centers[i] = np.array([])
        print(f"LOD {i} 中心点数量: 0 (上一级LOD为空)")
        continue

    # 将世界坐标转换回相对于该LOD原点的索引 (粗略)
    # 注意：这种降采样方法比较粗糙，直接在世界坐标上操作可能更好
    # 或者，如果使用八叉树体素化，可以直接从八叉树的不同层级提取
    scaled_indices = np.floor((prev_lod_centers - origin) / (current_pitch * factor)).astype(int)
    unique_scaled_indices, inv_indices = np.unique(scaled_indices, axis=0, return_inverse=True)

    # 为每个新的粗糙体素选择一个代表点（例如，原始点中的第一个，或计算平均值）
    # 这里我们简单地取每个独特粗糙索引对应的第一个原始中心点作为近似
    # 更准确的方法是计算这些原始中心点的平均值作为新LOD的中心
    new_lod_centers = []
    for j in range(len(unique_scaled_indices)):
        # 找到属于这个粗糙体素的所有原始LOD中心点
        original_points_in_coarse_voxel = prev_lod_centers[inv_indices == j]
        # 计算这些点的平均值作为新LOD体素的中心
        new_lod_centers.append(np.mean(original_points_in_coarse_voxel, axis=0))

    all_lods_voxel_centers[i] = np.array(new_lod_centers) if new_lod_centers else np.array([])
    print(f"LOD {i} 中心点数量: {len(all_lods_voxel_centers[i])}")
    # current_pitch *= factor # 如果下一级LOD的pitch是上一级的倍数


from py3dtiles.tileset.utils import TileContentReader
from py3dtiles.tileset.content.tile_content import TileContent
from py3dtiles.tileset.content.batch_table import BatchTable
from py3dtiles.tileset.content.feature_table import FeatureTable
from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tileset import Tileset
import json
import os
import shutil

# --- 3D Tiles生成参数 ---
OUTPUT_DIR = "output_3dtiles"
TEMPLATE_CUBE_GLB = "path/to/your/cube_template.glb" # 1x1x1单位立方体
TILESET_NAME = "MyVoxelTileset"
# GEOMETRIC_ERROR_BASE 和 FACTOR 用于计算每个LOD的 geometricError
# geometricError 越大，瓦片在更远处被加载（更粗糙的LOD）
# 通常，根瓦片的 geometricError 较大，叶子瓦片的 geometricError 较小
# 这里的设置需要根据场景大小和LOD层级仔细调整
GEOMETRIC_ERROR_BASE = 200 # 根瓦片的 geometricError
GEOMETRIC_ERROR_LOD_FACTOR = 0.5 # 每深入一层LOD，geometricError 乘以这个因子

# --- 清理并创建输出目录 ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# --- 读取模板立方体GLB内容 ---
try:
    with open(TEMPLATE_CUBE_GLB, 'rb') as f:
        glb_template_content = f.read()
except FileNotFoundError:
    print(f"错误: 模板立方体 GLB 文件 '{TEMPLATE_CUBE_GLB}' 未找到.")
    exit()

# --- 创建 Tileset 对象 ---
ts = Tileset()
ts.asset = {'version': '1.0', 'tilesetVersion': '1.0.0-myvoxel'}
ts.geometricError = GEOMETRIC_ERROR_BASE # 根瓦片的 geometricError

# --- 创建根瓦片 ---
# 根瓦片通常不直接包含内容，而是指向子瓦片（LOD 0的瓦片）
# 或者，根瓦片可以包含最低LOD的内容
root_tile = Tile()
root_tile.geometricError = GEOMETRIC_ERROR_BASE
# 计算整个数据集的包围盒 (所有LOD 0体素的包围盒)
if len(all_lods_voxel_centers[0]) > 0:
    min_bound = np.min(all_lods_voxel_centers[0] - HIGHEST_LOD_PITCH / 2, axis=0)
    max_bound = np.max(all_lods_voxel_centers[0] + HIGHEST_LOD_PITCH / 2, axis=0)
    center = (min_bound + max_bound) / 2
    half_size = (max_bound - min_bound) / 2
    # py3dtiles 使用的包围盒格式: [centerX, centerY, centerZ, halfX, 0, 0, 0, halfY, 0, 0, 0, halfZ]
    root_tile.boundingVolume = {'box': [
        center[0], center[1], center[2],
        half_size[0], 0, 0,
        0, half_size[1], 0,
        0, 0, half_size[2]
    ]}
else: # 如果没有体素，创建一个默认的小包围盒
    root_tile.boundingVolume = {'box': [0,0,0, 1,0,0, 0,1,0, 0,0,1]}

ts.root = root_tile
current_parent_tile = root_tile
tile_counter = 0

# --- 为每个LOD层级创建瓦片 ---
# 简化：这里我们为每个LOD创建一个单独的瓦片。
# 实际应用中，如果一个LOD的体素数量过多，需要进一步空间划分为多个瓦片（例如使用八叉树）。
for lod_level in range(LOD_LEVELS):
    voxel_centers_current_lod = all_lods_voxel_centers[lod_level]
    num_voxels_current_lod = len(voxel_centers_current_lod)

    if num_voxels_current_lod == 0:
        print(f"LOD {lod_level} 没有体素，跳过生成瓦片.")
        continue

    lod_pitch = HIGHEST_LOD_PITCH * LOD_DOWNSAMPLE_FACTORS[lod_level]

    # --- 创建i3dm内容 ---
    # 1. 特征表 (FeatureTable) - 定义实例的位置、缩放等
    # 位置 (POSITION) 是必须的
    positions = voxel_centers_current_lod.astype(np.float32) # (N, 3)

    # 缩放 (RTC_CENTER 和 SCALE_NON_UNIFORM)
    # 如果模板是1x1x1，我们需要根据lod_pitch进行缩放
    # py3dtiles i3dm 通常将实例的变换原点设置为RTC_CENTER，然后应用相对变换
    # 这里简化，我们直接在世界坐标中放置，所以RTC_CENTER可以为(0,0,0)
    # 或者，更常见的是将RTC_CENTER设置为瓦片的中心，positions是相对于RTC_CENTER的
    # 我们这里假设positions已经是世界坐标
    scales = np.full((num_voxels_current_lod, 3), lod_pitch, dtype=np.float32)

    # (可选) 法线/旋转 (NORMAL_UP, NORMAL_RIGHT 或四元数) - 对于轴对齐立方体，可以省略或使用默认值
    # (可选) 批次ID (BATCH_ID) - 如果需要每个实例有不同的属性（通过批处理表）

    feature_table_dict = {'POSITION': positions}
    if lod_pitch != 1.0: # 如果我们的模板不是预期的lod_pitch大小
         feature_table_dict['SCALE_NON_UNIFORM'] = scales
    # 如果需要RTC_CENTER
    # rtc_center = np.mean(positions, axis=0) # 例如瓦片的中心
    # feature_table_dict['RTC_CENTER'] = rtc_center
    # feature_table_dict['POSITION'] = positions - rtc_center # 相对位置

    ft = FeatureTable.from_dict(feature_table_dict)

    # 2. 批处理表 (BatchTable) - (可选) 定义每个实例的附加属性，如颜色、ID
    # bt = BatchTable.from_dict({'instance_id': np.arange(num_voxels_current_lod)})

    # 3. 创建 TileContent (i3dm)
    i3dm_content = TileContent()
    i3dm_content.feature_table = ft
    # i3dm_content.batch_table = bt # 如果有批处理表
    i3dm_content.body = glb_template_content # GLB内容
    i3dm_content.header = TileContentReader.read_binary_tile_content_header(i3dm_content.to_array()) # 自动生成头部

    # --- 保存 i3dm 文件 ---
    i3dm_filename = f"tile_lod{lod_level}_{tile_counter}.i3dm"
    i3dm_filepath = os.path.join(OUTPUT_DIR, i3dm_filename)
    with open(i3dm_filepath, 'wb') as f:
        f.write(i3dm_content.to_array())
    print(f"已生成: {i3dm_filepath}")
    tile_counter += 1

    # --- 创建或更新 Tile 对象 ---
    if lod_level == 0: # 最高LOD作为根瓦片的内容或根瓦片的直接子瓦片
        tile_for_lod = current_parent_tile # 如果根瓦片直接承载LOD0
        # 如果根瓦片不承载内容，而是作为容器：
        # tile_for_lod = Tile()
        # current_parent_tile.children.append(tile_for_lod)
    else: # 更低LOD作为上一级LOD瓦片的子瓦片 (实现LOD切换)
        new_tile = Tile()
        current_parent_tile.children.append(new_tile)
        tile_for_lod = new_tile
        current_parent_tile = new_tile # 下一个LOD的父级是当前创建的这个

    tile_for_lod.content_uri = i3dm_filename
    tile_for_lod.geometricError = GEOMETRIC_ERROR_BASE * (GEOMETRIC_ERROR_LOD_FACTOR ** lod_level)

    # 计算当前LOD瓦片的包围盒
    if num_voxels_current_lod > 0:
        min_b = np.min(voxel_centers_current_lod - lod_pitch / 2, axis=0)
        max_b = np.max(voxel_centers_current_lod + lod_pitch / 2, axis=0)
        c = (min_b + max_b) / 2
        hs = (max_b - min_b) / 2
        tile_for_lod.boundingVolume = {'box': [c[0],c[1],c[2], hs[0],0,0, 0,hs[1],0, 0,0,hs[2]]}
    else: # 默认小包围盒
        tile_for_lod.boundingVolume = {'box': [0,0,0, 0.1,0,0, 0,0.1,0, 0,0,0.1]}

    # 如果这是最后一个LOD层级，确保它没有children refine（如果有内容）
    if lod_level == LOD_LEVELS -1:
        tile_for_lod.refine = None # 或者 'REPLACE' 如果它有内容
    else:
        tile_for_lod.refine = 'REPLACE' # 或 'ADD'，取决于LOD策略

# --- 保存 tileset.json ---
tileset_json_path = os.path.join(OUTPUT_DIR, "tileset.json")
with open(tileset_json_path, 'w') as f:
    json.dump(ts.to_dict(), f, indent=2)
print(f"Tileset JSON 已保存到: {tileset_json_path}")

print(f"\n3D Tiles 数据集生成完毕，位于目录: {OUTPUT_DIR}")
print("请将此目录部署到HTTP服务器，或在UE5中从本地文件加载 tileset.json。")