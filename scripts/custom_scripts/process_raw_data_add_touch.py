#!/usr/bin/env python3
"""
Calvin数据集处理器 - 修正版（正确的触觉键名）
支持完整的传感器套件，包括触觉RGB和触觉深度

使用方法:
    python process_calvin_fixed.py --dataset_path <路径> --include_tactile
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import cv2


class CalvinDatasetProcessor:
    """Calvin数据集处理器 - 支持完整传感器套件（修正版）"""
    
    # Calvin传感器规格（基于实际数据）
    SENSOR_SPECS = {
        'rgb_static': {'shape': (200, 200, 3), 'type': 'image'},
        'depth_static': {'shape': (200, 200), 'type': 'depth'},
        'rgb_gripper': {'shape': (84, 84, 3), 'type': 'image'},
        'depth_gripper': {'shape': (84, 84), 'type': 'depth'},
        'rgb_tactile': {'shape': (160, 120, 6), 'type': 'tactile'},      # ⭐ 修正键名
        'depth_tactile': {'shape': (160, 120, 2), 'type': 'tactile'},    # ⭐ 新增
        'robot_obs': {'shape': (15,), 'type': 'state'},
        'scene_obs': {'shape': (24,), 'type': 'state'},
        'actions': {'shape': (7,), 'type': 'action'},
        'rel_actions': {'shape': (7,), 'type': 'action'},
    }
    
    def __init__(self, dataset_path: str, output_path: str, 
                 include_tactile: bool = False,
                 include_gripper_cam: bool = True):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.include_tactile = include_tactile
        self.include_gripper_cam = include_gripper_cam
        
        print(f"📊 传感器配置:")
        print(f"   • Static Camera RGB: ✅")
        print(f"   • Static Camera Depth: ✅")
        print(f"   • Gripper Camera RGB: {'✅' if include_gripper_cam else '❌'}")
        print(f"   • Gripper Camera Depth: {'✅' if include_gripper_cam else '❌'}")
        print(f"   • Tactile RGB (160x120x6): {'✅' if include_tactile else '❌'}")
        print(f"   • Tactile Depth (160x120x2): {'✅' if include_tactile else '❌'}")
        
    def load_language_annotations(self) -> Dict:
        """加载语言标注 - 支持扁平化格式"""
        lang_folder = self.dataset_path / 'lang_annotations'
        
        if not lang_folder.exists():
            print(f"⚠️  未找到 lang_annotations 文件夹")
            return {}
        
        auto_lang_file = lang_folder / 'auto_lang_ann.npy'
        if not auto_lang_file.exists():
            print(f"⚠️  未找到 auto_lang_ann.npy")
            return {}
        
        print(f"✅ 加载语言标注: auto_lang_ann.npy")
        data = np.load(auto_lang_file, allow_pickle=True).item()
        
        # 检查是否是扁平化格式
        if 'language' in data and 'info' in data:
            return self._parse_flat_annotations(data)
        else:
            return data
    
    def _parse_flat_annotations(self, data: Dict) -> Dict:
        """解析扁平化的标注格式"""
        annotations = {}
        
        language = data.get('language', {})
        info = data.get('info', {})
        
        anns = language.get('ann', [])
        tasks = language.get('task', [])
        indxs = info.get('indx', [])
        
        print(f"📊 解析到 {len(anns)} 个语言标注episodes")
        
        for i, (ann, task, (start, end)) in enumerate(zip(anns, tasks, indxs)):
            episode_id = f"lang_ann_{i:04d}"
            annotations[episode_id] = {
                'start_idx': int(start),
                'end_idx': int(end),
                'language': ann,
                'task': task
            }
            
        return annotations
    
    def find_episode_sequences(self) -> Dict[str, Dict]:
        """查找并组织episode序列 - 只使用语言标注"""
        npz_files = sorted(self.dataset_path.glob('episode_*.npz'))
        
        if not npz_files:
            print(f"⚠️  在 {self.dataset_path} 中未找到episode文件")
            return {}
        
        print(f"✅ 找到 {len(npz_files)} 个episode帧文件")
        
        lang_annotations = self.load_language_annotations()
        
        if not lang_annotations:
            print("❌ 未找到语言标注")
            return {}
        
        episodes = {}
        for episode_id, ann_data in lang_annotations.items():
            episodes[episode_id] = {
                'start': ann_data['start_idx'],
                'end': ann_data['end_idx'],
                'language_instruction': ann_data['language'],
                'task': ann_data.get('task', '')
            }
        
        print(f"✅ 将处理 {len(episodes)} 个episodes（仅包含语言标注的episodes）")
        return episodes
    
    def load_episode_data(self, start_idx: int, end_idx: int) -> Dict:
        """加载一个episode的所有数据（包括触觉RGB和触觉深度）"""
        robot_observations = []
        actions = []
        relative_actions = []
        scene_observations = []
        tactile_rgb_observations = []      # ⭐ RGB触觉
        tactile_depth_observations = []    # ⭐ 深度触觉
        missing_frames = []
        
        # 传感器可用性统计
        sensor_availability = {
            'rgb_static': 0,
            'depth_static': 0,
            'rgb_gripper': 0,
            'depth_gripper': 0,
            'rgb_tactile': 0,      # ⭐ 修正键名
            'depth_tactile': 0,    # ⭐ 新增
            'robot_obs': 0,
            'actions': 0,
            'scene_obs': 0
        }
        
        for frame_idx in range(start_idx, end_idx + 1):
            npz_file = self.dataset_path / f'episode_{frame_idx:07d}.npz'
            
            if not npz_file.exists():
                missing_frames.append(frame_idx)
                continue
            
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # 机器人状态
                if 'robot_obs' in data:
                    robot_observations.append(data['robot_obs'].tolist())
                    sensor_availability['robot_obs'] += 1
                
                # 动作
                if 'actions' in data:
                    actions.append(data['actions'].tolist())
                    sensor_availability['actions'] += 1
                
                if 'rel_actions' in data:
                    relative_actions.append(data['rel_actions'].tolist())
                
                # 场景观测
                if 'scene_obs' in data:
                    scene_observations.append(data['scene_obs'].tolist())
                    sensor_availability['scene_obs'] += 1
                
                # ⭐ RGB触觉数据（修正键名）
                if self.include_tactile and 'rgb_tactile' in data:
                    tactile_rgb_observations.append(data['rgb_tactile'])
                    sensor_availability['rgb_tactile'] += 1
                
                # ⭐ 深度触觉数据（新增）
                if self.include_tactile and 'depth_tactile' in data:
                    tactile_depth_observations.append(data['depth_tactile'])
                    sensor_availability['depth_tactile'] += 1
                
                # 检查其他传感器可用性
                for sensor in ['rgb_static', 'depth_static', 'rgb_gripper', 'depth_gripper']:
                    if sensor in data:
                        sensor_availability[sensor] += 1
                    
            except Exception as e:
                print(f"⚠️  读取 {npz_file} 失败: {e}")
                missing_frames.append(frame_idx)
        
        return {
            'robot_observations': robot_observations,
            'actions': actions,
            'relative_actions': relative_actions,
            'scene_observations': scene_observations,
            'tactile_rgb_observations': tactile_rgb_observations,      # ⭐ RGB触觉
            'tactile_depth_observations': tactile_depth_observations,  # ⭐ 深度触觉
            'expected_count': end_idx - start_idx + 1,
            'actual_count': len(robot_observations),
            'missing_count': len(missing_frames),
            'missing_frames': missing_frames,
            'sensor_availability': sensor_availability
        }
    
    def _normalize_depth(self, depth_array: np.ndarray) -> np.ndarray:
        """归一化深度图 - 保证返回uint8"""
        # 处理边界情况
        if depth_array.size == 0:
            return np.zeros_like(depth_array, dtype=np.uint8)
        
        # 只考虑有效深度值
        valid_depth = depth_array[depth_array > 0]
        
        if len(valid_depth) == 0:
            # 全是无效值，返回全零
            return np.zeros_like(depth_array, dtype=np.uint8)
        
        # 使用百分位数避免异常值
        min_depth = np.percentile(valid_depth, 1)
        max_depth = np.percentile(valid_depth, 99)
        
        # 归一化到0-1
        if max_depth > min_depth:
            normalized = (depth_array - min_depth) / (max_depth - min_depth + 1e-8)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(depth_array, dtype=np.float32)
        
        # 转换到0-255的uint8
        result = (normalized * 255).astype(np.uint8)
        
        return result

    def _process_tactile_depth(self, tactile_depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理触觉深度数据 - 确保返回uint8"""
        ch0 = tactile_depth[:, :, 0]
        ch1 = tactile_depth[:, :, 1]
        
        # 归一化每个通道
        ch0_viz = self._normalize_depth(ch0)
        ch1_viz = self._normalize_depth(ch1)
        
        # 双重保险：确保是uint8
        assert ch0_viz.dtype == np.uint8, f"ch0类型错误: {ch0_viz.dtype}"
        assert ch1_viz.dtype == np.uint8, f"ch1类型错误: {ch1_viz.dtype}"
        
        return ch0_viz, ch1_viz
        
    def _process_tactile_rgb_for_visualization(self, tactile_array: np.ndarray) -> np.ndarray:
        """处理触觉RGB图像用于可视化
        
        触觉RGB是160x120x6，包含6个通道
        将其转换为标准RGB图像
        
        Args:
            tactile_array: (160, 120, 6) 触觉RGB数据
        
        Returns:
            rgb_image: (160, 120, 3) RGB图像
        """
        # 方法: 使用前3个通道
        if tactile_array.shape[2] >= 3:
            rgb = tactile_array[:, :, :3]
        else:
            rgb = np.stack([tactile_array[:, :, 0]] * 3, axis=-1)
        
        # 确保是uint8
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        
        return rgb
    
    def _create_tactile_heatmap(self, tactile_array: np.ndarray) -> np.ndarray:
        """创建触觉热图（所有6个通道的平均）
        
        Args:
            tactile_array: (160, 120, 6)
        
        Returns:
            heatmap: (160, 120, 3) RGB热图
        """
        # 计算所有通道的平均
        tactile_mean = tactile_array.mean(axis=2)
        
        # 归一化
        tactile_norm = (tactile_mean - tactile_mean.min()) / (tactile_mean.max() - tactile_mean.min() + 1e-8)
        tactile_norm = (tactile_norm * 255).astype(np.uint8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(tactile_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def save_episode_images(self, episode_id: str, start_idx: int, end_idx: int) -> Dict:
        """保存episode的图像、深度图和触觉数据"""
        image_folder = self.output_path / 'images' / episode_id
        image_folder.mkdir(parents=True, exist_ok=True)
        
        folders = {
            'rgb_static': image_folder / 'rgb_static',
            'depth_static': image_folder / 'depth_static',
        }
        
        if self.include_gripper_cam:
            folders['rgb_gripper'] = image_folder / 'rgb_gripper'
            folders['depth_gripper'] = image_folder / 'depth_gripper'
        
        if self.include_tactile:
            # RGB触觉
            folders['tactile_rgb'] = image_folder / 'tactile_rgb'              # 原始6通道
            folders['tactile_rgb_viz'] = image_folder / 'tactile_rgb_viz'      # RGB可视化
            folders['tactile_rgb_heatmap'] = image_folder / 'tactile_rgb_heatmap'  # 热图
            
            # 深度触觉
            folders['tactile_depth'] = image_folder / 'tactile_depth'          # 原始2通道
            folders['tactile_depth_ch0'] = image_folder / 'tactile_depth_ch0'  # 通道0
            folders['tactile_depth_ch1'] = image_folder / 'tactile_depth_ch1'  # 通道1
        
        for folder in folders.values():
            folder.mkdir(exist_ok=True)
        
        counts = {k: 0 for k in folders.keys()}
        
        for frame_idx in range(start_idx, end_idx + 1):
            npz_file = self.dataset_path / f'episode_{frame_idx:07d}.npz'
            if not npz_file.exists():
                continue
            
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # RGB Static Camera
                if 'rgb_static' in data:
                    Image.fromarray(data['rgb_static']).save(
                        folders['rgb_static'] / f'frame_{frame_idx:07d}.png')
                    counts['rgb_static'] += 1
                
                # Depth Static Camera
                if 'depth_static' in data:
                    depth = data['depth_static']
                    Image.fromarray(self._normalize_depth(depth)).save(
                        folders['depth_static'] / f'frame_{frame_idx:07d}.png')
                    np.save(folders['depth_static'] / f'frame_{frame_idx:07d}.npy', depth)
                    counts['depth_static'] += 1
                
                # RGB Gripper Camera
                if self.include_gripper_cam and 'rgb_gripper' in data:
                    Image.fromarray(data['rgb_gripper']).save(
                        folders['rgb_gripper'] / f'frame_{frame_idx:07d}.png')
                    counts['rgb_gripper'] += 1
                
                # Depth Gripper Camera
                if self.include_gripper_cam and 'depth_gripper' in data:
                    depth = data['depth_gripper']
                    Image.fromarray(self._normalize_depth(depth)).save(
                        folders['depth_gripper'] / f'frame_{frame_idx:07d}.png')
                    np.save(folders['depth_gripper'] / f'frame_{frame_idx:07d}.npy', depth)
                    counts['depth_gripper'] += 1
                
                # ⭐ RGB触觉数据（修正键名）
                if self.include_tactile and 'rgb_tactile' in data:
                    tactile_rgb = data['rgb_tactile']
                    
                    # 验证形状
                    expected_shape = self.SENSOR_SPECS['rgb_tactile']['shape']
                    if tactile_rgb.shape != expected_shape:
                        print(f"  ⚠️  RGB触觉形状不匹配: 期望 {expected_shape}, 实际 {tactile_rgb.shape}")
                    
                    # 1. 保存原始6通道数据（numpy格式）
                    np.save(folders['tactile_rgb'] / f'frame_{frame_idx:07d}.npy', tactile_rgb)
                    
                    # 2. 保存RGB可视化（前3个通道）
                    tactile_rgb_viz = self._process_tactile_rgb_for_visualization(tactile_rgb)
                    Image.fromarray(tactile_rgb_viz).save(
                        folders['tactile_rgb_viz'] / f'frame_{frame_idx:07d}.png')
                    
                    # 3. 保存热图（所有通道的平均）
                    tactile_rgb_heatmap = self._create_tactile_heatmap(tactile_rgb)
                    Image.fromarray(tactile_rgb_heatmap).save(
                        folders['tactile_rgb_heatmap'] / f'frame_{frame_idx:07d}.png')
                    
                    counts['tactile_rgb'] += 1
                    counts['tactile_rgb_viz'] += 1
                    counts['tactile_rgb_heatmap'] += 1
                
                # ⭐ 深度触觉数据（新增）
                if self.include_tactile and 'depth_tactile' in data:
                    tactile_depth = data['depth_tactile']
                    
                    # 验证形状
                    expected_shape = self.SENSOR_SPECS['depth_tactile']['shape']
                    if tactile_depth.shape != expected_shape:
                        print(f"  ⚠️  深度触觉形状不匹配: 期望 {expected_shape}, 实际 {tactile_depth.shape}")
                    
                    # 1. 保存原始2通道数据
                    np.save(folders['tactile_depth'] / f'frame_{frame_idx:07d}.npy', tactile_depth)
                    
                    # 2. 分别保存两个通道的可视化
                    ch0_viz, ch1_viz = self._process_tactile_depth(tactile_depth)
                    
                    Image.fromarray(ch0_viz).save(
                        folders['tactile_depth_ch0'] / f'frame_{frame_idx:07d}.png')
                    Image.fromarray(ch1_viz).save(
                        folders['tactile_depth_ch1'] / f'frame_{frame_idx:07d}.png')
                    
                    counts['tactile_depth'] += 1
                    counts['tactile_depth_ch0'] += 1
                    counts['tactile_depth_ch1'] += 1
                    
            except Exception as e:
                print(f"⚠️  保存图像失败 {npz_file}: {e}")
                import traceback
                traceback.print_exc()
        
        return {'image_folder': f'images/{episode_id}', **counts}
    
    def process_episode(self, episode_id: str, episode_info: Dict, 
                       save_images: bool = True) -> Dict:
        """处理单个episode"""
        start_idx = episode_info['start']
        end_idx = episode_info['end']
        
        trajectory_data = self.load_episode_data(start_idx, end_idx)
        visual_info = {}
        if save_images:
            visual_info = self.save_episode_images(episode_id, start_idx, end_idx)
        
        # 数据维度统计
        dims = {'num_frames': trajectory_data['actual_count']}
        
        if trajectory_data['robot_observations']:
            dims['robot_obs_dim'] = len(trajectory_data['robot_observations'][0])
        
        if trajectory_data['actions']:
            dims['actions_dim'] = len(trajectory_data['actions'][0])
        
        if trajectory_data['relative_actions']:
            dims['rel_actions_dim'] = len(trajectory_data['relative_actions'][0])
        
        if trajectory_data['scene_observations']:
            dims['scene_obs_dim'] = len(trajectory_data['scene_observations'][0])
        
        # ⭐ RGB触觉
        if trajectory_data['tactile_rgb_observations']:
            tactile_shape = trajectory_data['tactile_rgb_observations'][0].shape
            dims['tactile_rgb_shape'] = list(tactile_shape)
            dims['has_tactile_rgb'] = True
        else:
            dims['has_tactile_rgb'] = False
        
        # ⭐ 深度触觉
        if trajectory_data['tactile_depth_observations']:
            tactile_depth_shape = trajectory_data['tactile_depth_observations'][0].shape
            dims['tactile_depth_shape'] = list(tactile_depth_shape)
            dims['has_tactile_depth'] = True
        else:
            dims['has_tactile_depth'] = False
        
        # 传感器可用性
        sensor_stats = trajectory_data['sensor_availability']
        total_frames = trajectory_data['actual_count']
        
        episode_json = {
            'episode_id': episode_id,
            'language_instruction': episode_info['language_instruction'],
            'task': episode_info.get('task', ''),
            'source_frames': {
                'start': start_idx,
                'end': end_idx,
                'expected_count': trajectory_data['expected_count'],
                'actual_count': trajectory_data['actual_count'],
                'missing_count': trajectory_data['missing_count']
            },
            'trajectory': {
                'robot_observations': trajectory_data['robot_observations'],
                'actions': trajectory_data['actions'],
                'relative_actions': trajectory_data['relative_actions'],
                'scene_observations': trajectory_data['scene_observations']
            },
            'data_statistics': dims,
            'sensor_coverage': {
                'rgb_static': f"{sensor_stats['rgb_static']}/{total_frames}",
                'depth_static': f"{sensor_stats['depth_static']}/{total_frames}",
                'rgb_gripper': f"{sensor_stats['rgb_gripper']}/{total_frames}",
                'depth_gripper': f"{sensor_stats['depth_gripper']}/{total_frames}",
                'rgb_tactile': f"{sensor_stats['rgb_tactile']}/{total_frames}",    # ⭐
                'depth_tactile': f"{sensor_stats['depth_tactile']}/{total_frames}", # ⭐
                'robot_obs': f"{sensor_stats['robot_obs']}/{total_frames}",
            },
            'visual_info': visual_info
        }
        
        if trajectory_data['missing_count'] > 0:
            episode_json['source_frames']['missing_frames'] = trajectory_data['missing_frames']
        
        return episode_json
    
    def process_all(self, save_images: bool = True, max_episodes: int = None):
        """处理所有episodes"""
        print("="*70)
        print("🚀 开始处理Calvin数据集")
        print("="*70)
        
        episodes = self.find_episode_sequences()
        if not episodes:
            print("❌ 未找到任何episode")
            return {}
        
        print(f"\n📊 Episodes列表:")
        print("-"*70)
        for i, (ep_id, ep_info) in enumerate(episodes.items()):
            length = ep_info['end'] - ep_info['start'] + 1
            lang = ep_info['language_instruction']
            if len(lang) > 50:
                lang = lang[:47] + '...'
            print(f"{i+1:2d}. {ep_id}: 帧 {ep_info['start']:6d}-{ep_info['end']:6d} "
                  f"({length:3d}帧) - {lang}")
        
        if max_episodes:
            episodes = dict(list(episodes.items())[:max_episodes])
            print(f"\n⚠️  限制处理前 {max_episodes} 个episodes")
        
        print()
        all_episodes = {}
        
        for episode_id, episode_info in tqdm(episodes.items(), desc="处理episodes"):
            try:
                episode_json = self.process_episode(episode_id, episode_info, save_images)
                all_episodes[episode_id] = episode_json
                
                json_path = self.output_path / f'{episode_id}.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(episode_json, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"\n❌ 处理 {episode_id} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成数据集摘要
        summary = {
            'dataset_name': 'calvin',
            'total_episodes': len(all_episodes),
            'sensor_config': {
                'static_camera_rgb': True,
                'static_camera_depth': True,
                'gripper_camera_rgb': self.include_gripper_cam,
                'gripper_camera_depth': self.include_gripper_cam,
                'tactile_rgb': self.include_tactile,      # ⭐
                'tactile_depth': self.include_tactile     # ⭐
            },
            'episodes': list(all_episodes.keys())
        }
        
        # 统计传感器覆盖率
        if all_episodes:
            sensor_coverage_avg = {}
            for ep in all_episodes.values():
                for sensor, coverage in ep['sensor_coverage'].items():
                    available, total = map(int, coverage.split('/'))
                    if sensor not in sensor_coverage_avg:
                        sensor_coverage_avg[sensor] = []
                    sensor_coverage_avg[sensor].append(available / total if total > 0 else 0)
            
            summary['sensor_coverage_avg'] = {
                sensor: f"{np.mean(values)*100:.1f}%"
                for sensor, values in sensor_coverage_avg.items()
            }
        
        with open(self.output_path / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✅ 处理完成! 总episodes: {len(all_episodes)}")
        print(f"   输出路径: {self.output_path}")
        if 'sensor_coverage_avg' in summary:
            print(f"\n📊 传感器平均覆盖率:")
            for sensor, coverage in summary['sensor_coverage_avg'].items():
                print(f"   • {sensor}: {coverage}")
        print(f"{'='*70}")
        
        return all_episodes


def main():
    parser = argparse.ArgumentParser(
        description='Calvin数据集处理器 - 修正版（正确的触觉键名）'
    )
    
    parser.add_argument('--dataset_path', type=str, 
                       default='./calvin_debug_dataset/training',
                       help='Calvin数据集路径')
    parser.add_argument('--output_path', type=str,
                       default='./calvin_processed_with_tactile_training',
                       help='输出路径')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='限制处理的episode数量')
    
    # 传感器选择
    parser.add_argument('--include_tactile', action='store_true',
                       help='包含触觉数据 (RGB 160x120x6 + Depth 160x120x2)')
    parser.add_argument('--no_gripper_cam', action='store_true',
                       help='不包含夹爪相机')
    parser.add_argument('--no_images', action='store_true',
                       help='不保存图像（仅保存轨迹JSON）')
    
    args = parser.parse_args()
    
    processor = CalvinDatasetProcessor(
        args.dataset_path, 
        args.output_path,
        include_tactile=args.include_tactile,
        include_gripper_cam=not args.no_gripper_cam
    )
    
    results = processor.process_all(
        save_images=not args.no_images, 
        max_episodes=args.max_episodes
    )
    
    if results:
        print("\n📋 示例episode:")
        first_ep = list(results.values())[0]
        print(f"  ID: {first_ep['episode_id']}")
        print(f"  任务: {first_ep['task']}")
        print(f"  指令: {first_ep['language_instruction']}")
        print(f"  帧数: {first_ep['data_statistics']['num_frames']}")
        print(f"  机器人状态维度: {first_ep['data_statistics']['robot_obs_dim']}")
        print(f"  动作维度: {first_ep['data_statistics']['actions_dim']}")
        if first_ep['data_statistics'].get('has_tactile_rgb'):
            print(f"  触觉RGB形状: {first_ep['data_statistics']['tactile_rgb_shape']}")
        if first_ep['data_statistics'].get('has_tactile_depth'):
            print(f"  触觉深度形状: {first_ep['data_statistics']['tactile_depth_shape']}")
        
        print(f"\n📊 传感器覆盖率:")
        for sensor, coverage in first_ep['sensor_coverage'].items():
            print(f"  • {sensor}: {coverage}")


if __name__ == "__main__":
    main()