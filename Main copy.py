import trimesh
import numpy as np

DEBUG = True  # 设置为 True 以启用调试输出

# --- 输入参数 ---
MODEL_FILE_PATH = r"D:\Datasets\PCLAsset\point_cloud.ply"  # 或 .stl, .ply, .glb 等trimesh支持的格式
# 对于点云，可以使用 open3d 加载并转换为 trimesh.PointCloud 对象

# --- 加载模型 ---
try:
    # 尝试使用 trimesh 直接加载
    mesh_object = trimesh.load_mesh(MODEL_FILE_PATH)
    print(f"模型 '{MODEL_FILE_PATH}' 使用 trimesh 加载成功.")
    print(f"Trimesh 加载的对象类型: {type(mesh_object)}")

    # 检查加载结果是否有点
    if not hasattr(mesh_object, 'vertices') or len(mesh_object.vertices) == 0:
        print("Trimesh 加载后顶点为空，尝试使用 Open3D...")
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(MODEL_FILE_PATH)
            if not pcd.has_points():
                raise ValueError("Open3D 加载点云失败或点云为空.")
            mesh = trimesh.PointCloud(np.asarray(pcd.points))
            # 如果原始点云有颜色，也可以尝试传递
            # if pcd.has_colors():
            #     mesh.colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            print(f"使用 Open3D 成功加载并转换为 trimesh.PointCloud，顶点数: {len(mesh.vertices)}")
        except ImportError:
            print("Open3D 未安装，无法作为备选方案加载。")
            raise ValueError("Trimesh 加载顶点为空且 Open3D 不可用。")
        except Exception as o3d_e:
            print(f"使用 Open3D 加载失败: {o3d_e}")
            raise
    else:
        mesh = mesh_object # Trimesh 直接加载成功且有点

except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

# --- 预处理 (示例) ---
# 仅当加载的是 Trimesh 对象 (网格) 时，才进行水密性检查
if isinstance(mesh, trimesh.Trimesh):
    print("加载对象为 Trimesh，进行预处理。")
    # 1. 确保模型是水密的 (如果体素化算法需要)
    if not mesh.is_watertight: # Trimesh 对象总是有 is_watertight 属性
        print("警告: 网格模型非水密，尝试修复...")
        mesh.fill_holes()
        # mesh.fix_normals() # 可能也需要
        if not mesh.vertices.size or not mesh.faces.size:
            print("警告: 修复后模型为空，请检查原始模型或修复步骤。")
elif isinstance(mesh, trimesh.PointCloud):
    print("加载对象为 PointCloud，跳过网格特定的水密性修复。")
else:
    print(f"加载的模型类型 ({type(mesh)}) 未知或不受支持，跳过预处理。")


# 确保在继续之前 mesh 对象是有效的并且有顶点
if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
    print("错误：模型在预处理后没有顶点数据，无法进行体素化。")
    exit()

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
if isinstance(mesh, trimesh.Trimesh):
    print(f"模型顶点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")
elif isinstance(mesh, trimesh.PointCloud):
    print(f"模型顶点数: {len(mesh.vertices)}") # 点云没有传统意义上的“面”

# --- 体素化参数 ---
# PITCH 决定了最高LOD的体素大小 (立方体边长)
# 例如，如果模型单位是米，PITCH = 1.0 表示1立方米的体素
HIGHEST_LOD_PITCH = 1.0  # 体素边长，也是最高LOD的分辨率

# --- 执行体素化 ---
# `voxelized` 方法返回一个 VoxelGrid 对象
# `pitch` 参数是体素的边长
if isinstance(mesh, trimesh.Trimesh):
    voxel_grid = mesh.voxelized(pitch=HIGHEST_LOD_PITCH)
elif isinstance(mesh, trimesh.PointCloud):
    if len(mesh.vertices) == 0:
        print("错误：点云对象不包含顶点，无法进行体素化。")
        exit()

    points = mesh.vertices
    pitch = HIGHEST_LOD_PITCH

    # 1. 将点转换为离散的体素索引 (相对于世界坐标系下的pitch网格)
    discrete_indices_world = np.round(points / pitch).astype(int)
    if DEBUG:
        print(f"DEBUG: discrete_indices_world.shape: {discrete_indices_world.shape}, .dtype: {discrete_indices_world.dtype}")


    # 2. 找到唯一的被占据的体素索引
    if discrete_indices_world.size == 0:
        unique_occupied_indices_world = np.empty((0, 3), dtype=discrete_indices_world.dtype if discrete_indices_world.size > 0 else int)
        if DEBUG:
            print("DEBUG: discrete_indices_world is empty, unique_occupied_indices_world set to empty.")
    else:
        # 使用 numpy.unique 来获取唯一的行，它会保持原始的dtype和2D形状
        unique_occupied_indices_world = np.unique(discrete_indices_world, axis=0)
        if DEBUG:
            print(f"DEBUG: Using np.unique. unique_occupied_indices_world.shape: {unique_occupied_indices_world.shape}, .dtype: {unique_occupied_indices_world.dtype}")

    # 后续的 DEBUG 打印和警告可以保留，以验证 np.unique 的结果
    if DEBUG:
        print(f"DEBUG: unique_occupied_indices_world.shape after assignment: {unique_occupied_indices_world.shape}")
        print(f"DEBUG: unique_occupied_indices_world.ndim after assignment: {unique_occupied_indices_world.ndim}")
    if unique_occupied_indices_world.size > 0 and (unique_occupied_indices_world.ndim != 2 or unique_occupied_indices_world.shape[1] != 3):
        print(f"WARNING: unique_occupied_indices_world (from np.unique) is not shaped (M, 3)! Actual shape: {unique_occupied_indices_world.shape}")


    if len(unique_occupied_indices_world) == 0:
        print("警告：根据PITCH参数，点云未占据任何体素单元。")
        # 创建一个空的 VoxelGrid
        # 使用点云的最小边界（如果存在）或原点作为变换参考
        ref_point_for_transform = points.min(axis=0) if points.size > 0 else np.array([0.0, 0.0, 0.0])
        empty_transform = trimesh.transformations.scale_and_translate(scale=pitch, translate=ref_point_for_transform)
        empty_encoding = trimesh.voxel.encoding.SparseBinaryEncoding(np.empty((0, 3), dtype=int), shape=(0,0,0))
        voxel_grid = trimesh.voxel.VoxelGrid(empty_encoding, empty_transform)
    else:
        # unique_occupied_indices_world is not empty here
        # 3. 确定体素网格编码的原点索引 (在世界索引空间中)
        if unique_occupied_indices_world.ndim == 2 and unique_occupied_indices_world.shape[1] == 3:
            # This is the "good" path where unique_occupied_indices_world has the expected shape (M, 3)
            encoding_origin_world_index = unique_occupied_indices_world.min(axis=0)
            print(f"DEBUG: encoding_origin_world_index.shape: {encoding_origin_world_index.shape}")

            # 4. 将唯一索引转换为相对于编码原点的0基索引
            # 确保 encoding_origin_world_index 形状兼容 (it should be (3,) here)
            if hasattr(encoding_origin_world_index, 'shape') and encoding_origin_world_index.shape == (3,):
                sparse_relative_indices = unique_occupied_indices_world - encoding_origin_world_index
            else:
                # This case should ideally not be reached if the above logic is sound
                print(f"ERROR: encoding_origin_world_index has unexpected shape: {encoding_origin_world_index.shape if hasattr(encoding_origin_world_index, 'shape') else type(encoding_origin_world_index)}")
                sparse_relative_indices = np.empty((0,3), dtype=int) # Fallback

            print(f"DEBUG: sparse_relative_indices.shape: {sparse_relative_indices.shape}")

            # 5. 创建 SparseBinaryEncoding 对象
            if sparse_relative_indices.size > 0 and sparse_relative_indices.ndim == 2 and sparse_relative_indices.shape[1] == 3 :
                encoding_shape = sparse_relative_indices.max(axis=0) + 1
            else: 
                print(f"DEBUG: sparse_relative_indices is empty or has wrong shape ({sparse_relative_indices.shape}), setting encoding_shape to (0,0,0)")
                encoding_shape = (0,0,0)

            print(f"DEBUG: encoding_shape: {encoding_shape}")
            voxel_encoding = trimesh.voxel.encoding.SparseBinaryEncoding(
                sparse_relative_indices, 
                shape=encoding_shape
            )

            # 6. 计算从0基体素索引到世界坐标的变换矩阵
            # 变换的原点是编码原点索引在世界坐标中的位置
            if hasattr(encoding_origin_world_index, 'shape') and encoding_origin_world_index.shape == (3,):
                 transform_translation = encoding_origin_world_index * pitch
                 voxel_transform = trimesh.transformations.scale_and_translate(
                    scale=pitch, 
                    translate=transform_translation
                )
            else: # Fallback if encoding_origin_world_index was not (3,)
                print(f"ERROR: Cannot compute voxel_transform due to encoding_origin_world_index shape.")
                ref_point_for_transform = points.min(axis=0) if points.size > 0 else np.array([0.0, 0.0, 0.0])
                voxel_transform = trimesh.transformations.scale_and_translate(scale=pitch, translate=ref_point_for_transform)


            # 7. 创建 VoxelGrid 对象
            voxel_grid = trimesh.voxel.VoxelGrid(voxel_encoding, voxel_transform)
            print(f"体素化完成 (有效数据). 体素网格形状: {voxel_grid.shape}")

        else: # unique_occupied_indices_world is not empty BUT has an unexpected shape
            print(f"ERROR: Cannot compute voxel grid because unique_occupied_indices_world (non-empty) has unexpected shape: {unique_occupied_indices_world.shape}")
            # 创建一个空的 VoxelGrid 作为回退
            ref_point_for_transform = points.min(axis=0) if points.size > 0 else np.array([0.0, 0.0, 0.0])
            empty_transform = trimesh.transformations.scale_and_translate(scale=pitch, translate=ref_point_for_transform)
            empty_encoding = trimesh.voxel.encoding.SparseBinaryEncoding(np.empty((0, 3), dtype=int), shape=(0,0,0))
            voxel_grid = trimesh.voxel.VoxelGrid(empty_encoding, empty_transform)
            print(f"体素化完成 (空 due to unexpected shape). 体素网格形状: {voxel_grid.shape}")

else:
    print(f"错误：未知的对象类型 {type(mesh)}，无法进行体素化。")
    exit()

print(f"体素化完成. 体素网格形状: {voxel_grid.shape}")

# VoxelGrid.encoding.dense_indices 包含了所有被占据体素的整数索引 (i, j, k)
# VoxelGrid.transform 是从体素索引到世界坐标的变换矩阵
# 修正：使用 sparse_indices 替代 dense_indices
occupied_voxel_indices = voxel_grid.encoding.sparse_indices # (N, 3) array of int indices
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


from py3dtiles.tileset.content.tile_content import TileContent
from py3dtiles.tileset.content.batch_table import BatchTable
from py3dtiles.tileset.content.feature_table import FeatureTable
from py3dtiles.tileset.content.b3dm import B3dm # MODIFIED: Import B3dm
# from py3dtiles.tileset.content.i3dm import I3dm # REMOVED or if you had it

from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tileset import TileSet
import json
import os
import shutil

# --- 3D Tiles生成参数 ---
OUTPUT_DIR = "output_3dtiles"
# TEMPLATE_CUBE_GLB = r"D:\Datasets\OBJAsset\gltf\1x1x1.glb" # REMOVED: No longer needed as we generate GLBs dynamically
TILESET_NAME = "MyVoxelTileset"
GEOMETRIC_ERROR_BASE = 200 
GEOMETRIC_ERROR_LOD_FACTOR = 0.5 

# --- 清理并创建输出目录 ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# --- 读取模板立方体GLB内容 ---
# REMOVED: glb_template_content is no longer read from a file here
# try:
#     with open(TEMPLATE_CUBE_GLB, 'rb') as f:
#         glb_template_content = f.read()
# except FileNotFoundError:
#     print(f"错误: 模板立方体 GLB 文件 '{TEMPLATE_CUBE_GLB}' 未找到.")
#     exit()

# --- 创建 Tileset 对象 ---
ts = TileSet()
ts.asset = {'version': '1.0', 'tilesetVersion': '1.0.0-myvoxel-b3dm'} # MODIFIED: Indicate b3dm
ts.geometricError = GEOMETRIC_ERROR_BASE 

# --- 创建根瓦片 ---
root_tile = Tile()
root_tile.geometricError = GEOMETRIC_ERROR_BASE
if len(all_lods_voxel_centers[0]) > 0:
    min_bound = np.min(all_lods_voxel_centers[0] - HIGHEST_LOD_PITCH / 2, axis=0)
    max_bound = np.max(all_lods_voxel_centers[0] + HIGHEST_LOD_PITCH / 2, axis=0)
    center = (min_bound + max_bound) / 2
    half_size = (max_bound - min_bound) / 2
    root_tile.boundingVolume = {'box': [
        center[0], center[1], center[2],
        half_size[0], 0, 0,
        0, half_size[1], 0,
        0, 0, half_size[2]
    ]}
else: 
    root_tile.boundingVolume = {'box': [0,0,0, 1,0,0, 0,1,0, 0,0,1]}

ts.root = root_tile
current_parent_tile = root_tile
tile_counter = 0

# --- 为每个LOD层级创建瓦片 ---
for lod_level in range(LOD_LEVELS):
    voxel_centers_current_lod = all_lods_voxel_centers[lod_level]
    num_voxels_current_lod = len(voxel_centers_current_lod)

    if num_voxels_current_lod == 0:
        print(f"LOD {lod_level} 没有体素，跳过生成瓦片.")
        continue

    lod_pitch = HIGHEST_LOD_PITCH * LOD_DOWNSAMPLE_FACTORS[lod_level]

    # --- 创建 B3DM 内容 ---
    # 1. 为当前瓦片生成一个包含所有实例几何的 GLB
    list_of_cube_meshes = []
    current_batch_id = 0
    for center_point in voxel_centers_current_lod:
        # 创建一个立方体网格
        cube_mesh = trimesh.creation.box(extents=[lod_pitch, lod_pitch, lod_pitch])
        # 将其移动到体素中心
        cube_mesh.apply_translation(center_point)
        
        # 为顶点添加 _BATCHID 属性，以便在GLB中区分实例
        # 这对于后续使用BatchTable在着色器中对单个立方体进行操作（如着色）很有用
        # 如果不打算这样做，可以跳过 _BATCHID 的添加
        vertex_batch_ids = np.full(len(cube_mesh.vertices), current_batch_id, dtype=np.uint32)
        cube_mesh.vertex_attributes['_BATCHID'] = vertex_batch_ids
        
        list_of_cube_meshes.append(cube_mesh)
        current_batch_id += 1

    if not list_of_cube_meshes:
        print(f"LOD {lod_level} 没有生成任何立方体网格，跳过瓦片.")
        continue

    merged_mesh = trimesh.util.concatenate(list_of_cube_meshes)
    
    # 导出合并后的网格为 GLB 字节流
    # 确保 trimesh 版本支持导出带有自定义顶点属性的 GLB
    # 通常，以 '_' 开头的顶点属性会被导出
    glb_body_bytes = merged_mesh.export(file_type='glb')

    if not glb_body_bytes:
        print(f"错误: LOD {lod_level} 的 GLB 内容为空.")
        continue
        
    # 2. 特征表 (FeatureTable)
    # 对于B3DM，最基本的是BATCH_LENGTH
    # 我们将直接构建 B3DM 特征表头部所需的字典
    b3dm_feature_table_header_dict = {'BATCH_LENGTH': num_voxels_current_lod}
    # ft = FeatureTable.from_dict(feature_table_dict) # REMOVE THIS LINE

    # 3. 批处理表 (BatchTable) - (可选)
    # 示例：为每个实例（立方体）创建一个ID
    # 这些ID可以用于在着色器中通过 gl_InstanceID 或 _BATCHID 顶点属性来查找和应用特定属性
    # 我们将直接构建 B3DM 批处理表头部所需的字典
    # 注意：BatchTable 类本身可能有 from_dict，但我们这里是为 B3DM 的批处理表准备数据
    b3dm_batch_table_header_dict = {
        'instanceId': { # 描述 instanceId 属性
            'byteOffset': 0, # 假设它是批处理表二进制体中的第一个属性
            'componentType': 'UNSIGNED_INT', # 对应 np.uint32
            'type': 'SCALAR'
        }
        # 如果有更多属性，例如 'color':
        # 'color': {
        #     'byteOffset': num_voxels_current_lod * np.dtype(np.uint32).itemsize, # 在 instanceId 之后
        #     'componentType': 'UNSIGNED_BYTE',
        #     'type': 'VEC3'
        # }
    }
    # 创建批处理表的二进制体
    # instanceId 数据
    instance_id_binary = np.arange(num_voxels_current_lod, dtype=np.uint32).tobytes()
    # color_binary = np.random.randint(0, 255, size=(num_voxels_current_lod, 3), dtype=np.uint8).tobytes()
    # b3dm_batch_table_binary_body = instance_id_binary + color_binary
    b3dm_batch_table_binary_body = instance_id_binary
    
    # 4. 创建 B3DM 内容对象
    # 使用已生成的 GLB 字节流

    try:
        # 方法1: 尝试使用 B3dm.from_gltf，但需要正确的 pygltflib.GLTF2 对象
        import pygltflib
        
        # 将 GLB 字节加载为 GLTF2 对象
        gltf_asset = pygltflib.GLTF2.load_from_bytes(glb_body_bytes)
        
        # 创建批处理表数据 - 使用更简单的方法
        # 不手动创建 BatchTable 对象，而是传递原始数据让 py3dtiles 处理
        batch_table_data_simple = {
            'instanceId': list(range(num_voxels_current_lod))  # 转换为普通 Python 列表
        }
        
        # 尝试直接使用 from_gltf，让它处理批处理表
        b3dm_content = B3dm.from_gltf(
            gltf=gltf_asset,
            batch_table=None  # 先不传递批处理表，看是否能成功创建基本的 B3DM
        )
        
        print(f"B3DM 内容创建成功（基本版本），LOD {lod_level}")

    except Exception as e:
        print(f"错误: 使用 from_gltf 创建 B3DM 内容失败: {e}")
        
        try:
            # 方法2: 尝试最简单的方法 - 直接从合并的网格导出 GLB，然后使用 from_array
            # 但我们需要手动构建完整的 B3DM 字节结构
            
            # 简化：创建最小的 B3DM，只包含 GLB，没有复杂的批处理表
            from py3dtiles.tileset.content.b3dm_feature_table import B3dmFeatureTable
            from py3dtiles.tileset.content.b3dm import B3dmHeader, B3dmBody
            
            # 创建特征表 - 只包含 BATCH_LENGTH
            feature_table_header = {'BATCH_LENGTH': num_voxels_current_lod}
            feature_table_json = json.dumps(feature_table_header).encode('utf-8')
            
            # 对齐到4字节边界
            ft_json_padding = (4 - len(feature_table_json) % 4) % 4
            feature_table_json_padded = feature_table_json + b' ' * ft_json_padding
            
            # 特征表二进制体（空）
            feature_table_binary = b''
            
            # 批处理表（简化为空）
            batch_table_json = b''
            batch_table_binary = b''
            
            # 创建 B3DM 头部
            header = B3dmHeader()
            header.magic = b'b3dm'
            header.version = 1
            header.tile_byte_length = 0  # 将在后面计算
            header.feature_table_json_byte_length = len(feature_table_json_padded)
            header.feature_table_binary_byte_length = len(feature_table_binary)
            header.batch_table_json_byte_length = len(batch_table_json)
            header.batch_table_binary_byte_length = len(batch_table_binary)
            
            # 计算总长度
            header_size = 28  # B3DM 头部固定大小
            total_length = (header_size + 
                          len(feature_table_json_padded) + 
                          len(feature_table_binary) + 
                          len(batch_table_json) + 
                          len(batch_table_binary) + 
                          len(glb_body_bytes))
            header.tile_byte_length = total_length
            
            # 组装完整的 B3DM 字节流
            b3dm_bytes = (header.to_array().tobytes() + 
                         feature_table_json_padded + 
                         feature_table_binary + 
                         batch_table_json + 
                         batch_table_binary + 
                         glb_body_bytes)
            
            # 使用 from_array 创建 B3dm 对象
            b3dm_content = B3dm.from_array(np.frombuffer(b3dm_bytes, dtype=np.uint8))
            
            print(f"B3DM 内容创建成功（手动构建），LOD {lod_level}")
            
        except Exception as e2:
            print(f"错误: 手动构建 B3DM 内容也失败: {e2}")
            import traceback
            traceback.print_exc()
            continue
    
    # --- 保存 b3dm 文件 ---
    b3dm_filename = f"tile_lod{lod_level}_{tile_counter}.b3dm"
    b3dm_filepath = os.path.join(OUTPUT_DIR, b3dm_filename)
    with open(b3dm_filepath, 'wb') as f:
        f.write(b3dm_content.to_array())
    print(f"已生成: {b3dm_filepath}")
    tile_counter += 1

    # --- 创建或更新 Tile 对象 ---
    # --- 创建或更新 Tile 对象 ---
    if lod_level == 0: 
        tile_for_lod = current_parent_tile 
    else: 
        new_tile = Tile()
        # 确保新瓦片被正确初始化
        if not hasattr(current_parent_tile, 'children'):
            current_parent_tile.children = []
        current_parent_tile.children.append(new_tile)
        tile_for_lod = new_tile
        # 重要：只有当我们成功创建了瓦片内容后，才更新 current_parent_tile
        # current_parent_tile = new_tile # 移动到后面

    tile_for_lod.content_uri = b3dm_filename
    tile_for_lod.geometricError = GEOMETRIC_ERROR_BASE * (GEOMETRIC_ERROR_LOD_FACTOR ** lod_level)

    if num_voxels_current_lod > 0:
        # 使用合并后网格的实际边界来计算瓦片包围盒，更准确
        min_b = merged_mesh.bounds[0]
        max_b = merged_mesh.bounds[1]
        c = (min_b + max_b) / 2
        hs = (max_b - min_b) / 2
        tile_for_lod.boundingVolume = {'box': [c[0],c[1],c[2], hs[0],0,0, 0,hs[1],0, 0,0,hs[2]]}
    else: 
        tile_for_lod.boundingVolume = {'box': [0,0,0, 0.1,0,0, 0,0.1,0, 0,0,0.1]}

    if lod_level == LOD_LEVELS -1:
        tile_for_lod.refine = None 
    else:
        tile_for_lod.refine = 'REPLACE'
    
    # 只有在成功设置了所有属性后，才更新父瓦片指针
    if lod_level > 0:
        current_parent_tile = tile_for_lod


# --- 修复瓦片数值属性的数据类型问题 ---
def fix_tile_numeric_properties(tile):
    """递归地将瓦片属性中的 numpy 数据类型转换为标准 Python 数据类型"""
    if hasattr(tile, 'boundingVolume') and tile.boundingVolume:
        if 'box' in tile.boundingVolume and tile.boundingVolume['box']:
            box = tile.boundingVolume['box']
            tile.boundingVolume['box'] = [float(x) for x in box]
        elif 'sphere' in tile.boundingVolume and tile.boundingVolume['sphere']:
            sphere = tile.boundingVolume['sphere']
            tile.boundingVolume['sphere'] = [float(x) for x in sphere]
        elif 'region' in tile.boundingVolume and tile.boundingVolume['region']:
            region = tile.boundingVolume['region']
            tile.boundingVolume['region'] = [float(x) for x in region]

    if hasattr(tile, 'geometricError') and tile.geometricError is not None:
        tile.geometricError = float(tile.geometricError)
    
    # 递归处理子瓦片
    if hasattr(tile, 'children') and tile.children:
        for child_tile in tile.children: # Changed from 'child' to 'child_tile' for clarity
            fix_tile_numeric_properties(child_tile)

print("修复瓦片数值属性数据类型...")
fix_tile_numeric_properties(ts.root)

# --- 保存 tileset.json 前进行额外的验证 ---
def validate_tile_hierarchy(tile, depth=0):
    """递归验证瓦片层次结构中的每个瓦片都有包围盒和有效的几何误差"""
    indent = "  " * depth
    print(f"{indent}验证瓦片 (深度 {depth}):")
    
    valid = True
    if not hasattr(tile, 'boundingVolume') or not tile.boundingVolume:
        print(f"{indent}  错误: 瓦片没有包围盒")
        valid = False
    else:
        bv_type_str = 'N/A'
        if 'box' in tile.boundingVolume and tile.boundingVolume['box']:
            bv_type_str = str(type(tile.boundingVolume['box'][0]))
        elif 'sphere' in tile.boundingVolume and tile.boundingVolume['sphere']:
            bv_type_str = str(type(tile.boundingVolume['sphere'][0]))
        elif 'region' in tile.boundingVolume and tile.boundingVolume['region']:
            bv_type_str = str(type(tile.boundingVolume['region'][0]))
        print(f"{indent}  包围盒: OK - 类型: {bv_type_str}")
    
    if not hasattr(tile, 'geometricError') or tile.geometricError is None:
        print(f"{indent}  错误: 瓦片没有几何误差")
        valid = False
    else:
        print(f"{indent}  几何误差: OK ({tile.geometricError}) - 类型: {type(tile.geometricError)}")
    
    if hasattr(tile, 'content_uri') and tile.content_uri: # Check if content_uri is not None or empty
        print(f"{indent}  内容URI: {tile.content_uri}")
    elif hasattr(tile, 'content') and tile.content: # Check for inline content
        print(f"{indent}  内容: (内联)")
    else:
        # Tiles without content are valid (e.g. parent tiles for refinement)
        print(f"{indent}  无内容 (可能是父瓦片)")


    if hasattr(tile, 'children') and tile.children:
        print(f"{indent}  子瓦片数量: {len(tile.children)}")
        for i, child_tile_val in enumerate(tile.children): # Changed from 'child' to 'child_tile_val'
            print(f"{indent}  验证子瓦片 {i}:")
            if not validate_tile_hierarchy(child_tile_val, depth + 1):
                valid = False # Propagate failure
    else:
        print(f"{indent}  无子瓦片")
    
    return valid

print("开始验证瓦片层次结构...")
if not validate_tile_hierarchy(ts.root):
    print("瓦片层次结构验证失败. 请检查日志.")
    # exit() # Decide if you want to exit or try to save simplified
else:
    print("瓦片层次结构验证通过.")

# --- 保存 tileset.json ---
tileset_json_path = os.path.join(OUTPUT_DIR, "tileset.json")

# 调试信息
print(f"\nDEBUG: 准备保存 Tileset...")
print(f"DEBUG: ts.asset: {ts.asset}")
print(f"DEBUG: ts.geometricError: {ts.geometricError} (类型: {type(ts.geometricError)})")
print(f"DEBUG: 根瓦片 geometricError: {ts.root.geometricError} (类型: {type(ts.root.geometricError)})")
print(f"DEBUG: 根瓦片包围盒: {ts.root.boundingVolume}")
if ts.root.boundingVolume and 'box' in ts.root.boundingVolume and ts.root.boundingVolume['box']:
    print(f"DEBUG: 根瓦片包围盒数据类型: {type(ts.root.boundingVolume['box'][0])}")
print(f"DEBUG: 根瓦片子瓦片数量: {len(getattr(ts.root, 'children', []))}")
if hasattr(ts.root, 'children'):
    for i, child_debug in enumerate(ts.root.children): # Changed from 'child' to 'child_debug'
        print(f"DEBUG: 子瓦片 {i} geometricError: {child_debug.geometricError} (类型: {type(child_debug.geometricError)})")
        print(f"DEBUG: 子瓦片 {i} 包围盒: {getattr(child_debug, 'boundingVolume', 'None')}")
        if getattr(child_debug, 'boundingVolume', None) and 'box' in child_debug.boundingVolume and child_debug.boundingVolume['box']:
             print(f"DEBUG: 子瓦片 {i} 包围盒数据类型: {type(child_debug.boundingVolume['box'][0])}")
        print(f"DEBUG: 子瓦片 {i} content_uri: {getattr(child_debug, 'content_uri', 'None')}")


try:
    with open(tileset_json_path, 'w') as f:
        json.dump(ts.to_dict(), f, indent=2) # This calls ts.root.sync_bounding_volume_with_children()
    print(f"Tileset JSON 已成功保存到: {tileset_json_path}")
except Exception as e:
    print(f"保存 tileset.json 时出错: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n尝试创建并保存简化的 tileset 字典...")
    try:
        # 确保顶层 geometricError 也是 float
        ts_geometric_error = float(ts.geometricError) if ts.geometricError is not None else 0.0

        simple_tileset_dict = {
            "asset": ts.asset,
            "geometricError": ts_geometric_error,
            "root": {}
        }

        def build_tile_dict(tile_obj):
            tile_data = {}
            if not (hasattr(tile_obj, 'boundingVolume') and tile_obj.boundingVolume):
                print(f"警告: 瓦片 {getattr(tile_obj, 'content_uri', '未知URI')} 在简化构建时缺少包围盒。将使用默认值。")
                # 提供一个默认包围盒以避免错误，但这表明原始数据有问题
                tile_data["boundingVolume"] = {'box': [0,0,0,1,0,0,0,1,0,0,0,1]}
            else:
                tile_data["boundingVolume"] = tile_obj.boundingVolume # 已被 fix_tile_numeric_properties 清理

            tile_data["geometricError"] = float(tile_obj.geometricError) if tile_obj.geometricError is not None else 0.0
            
            if hasattr(tile_obj, 'content_uri') and tile_obj.content_uri:
                tile_data["content"] = {"uri": tile_obj.content_uri}
            elif hasattr(tile_obj, 'content') and tile_obj.content: # For inline content
                 tile_data["content"] = tile_obj.content.to_dict() if hasattr(tile_obj.content, 'to_dict') else tile_obj.content


            if hasattr(tile_obj, 'refine') and tile_obj.refine:
                tile_data["refine"] = tile_obj.refine
            
            children_list = []
            if hasattr(tile_obj, 'children') and tile_obj.children:
                for child_obj_recurse in tile_obj.children: # Changed from 'child_obj' to 'child_obj_recurse'
                    children_list.append(build_tile_dict(child_obj_recurse))
            
            if children_list:
                tile_data["children"] = children_list
            
            return tile_data

        simple_tileset_dict["root"] = build_tile_dict(ts.root)
        
        with open(tileset_json_path, 'w') as f:
            json.dump(simple_tileset_dict, f, indent=2)
        print(f"简化的 Tileset JSON 已保存到: {tileset_json_path}")
        
    except Exception as simple_e:
        print(f"创建并保存简化 tileset 也失败: {simple_e}")
        import traceback
        traceback.print_exc()

print(f"\n3D Tiles 数据集生成完毕，位于目录: {OUTPUT_DIR}")
print("请将此目录部署到HTTP服务器，或在UE5中从本地文件加载 tileset.json。")