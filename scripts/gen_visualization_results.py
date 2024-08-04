import os
from argparse import ArgumentParser

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from nuscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion

from datasets.nusc_det_dataset import map_name_from_general_to_detection
import gc

def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('idx',
                        type=int,
                        help='Index of the dataset to be visualized.')
    parser.add_argument('result_path', help='Path of the result json file.')
    parser.add_argument('target_path',
                        help='Target path to save the visualization result.')

    args = parser.parse_args()
    return args


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    box = Box(
        box_dict['translation'],
        box_dict['size'],
        Quaternion(box_dict['rotation']),
    )
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    box.translate(trans)
    box.rotate(rot)
    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
        axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center
    Returns:
    """
    template = (np.array((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2)

    corners3d = np.tile(boxes3d[:, None, 3:6],
                        [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3),
                                      boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def get_bev_lines(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners,offset_x,offset_y,resize_div):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0]/resize_div+offset_x, corners[ed, 0]/resize_div+offset_x],
                        [corners[st, 1]/resize_div+offset_y, corners[ed, 1]/resize_div+offset_y]])
    return ret


def get_cam_corners(corners, translation, rotation, cam_intrinsics):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners

def merge_images(info,IMG_KEYS,resize_dim):
    images = []
    for key in IMG_KEYS:
        img = mmcv.imread(
            os.path.join('data/nuScenes', info['cam_infos'][key]['filename']))
        
        if img is not None:
            img = cv2.resize(img, resize_dim)  # 이미지 리사이징
            # CAM_BACK 카메라만 좌우 반전
            if 'BACK' in key:
                img = cv2.flip(img, 1)
            
            if img is not None:    
                images.append(img)
            else:
                print(f"Error reading image from {info['cam_infos'][key]['filename']}")
                
    combined_row1 = np.hstack(images[:3])
    combined_row2 = np.hstack(images[3:])
    combined_image = np.vstack([combined_row1, combined_row2]) 
    image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    
    return image

def visualize_predictions(ax,info,IMG_KEYS,resize_dim, pred_corners,pred_class,show_classes, resize_div):
    for k, key in enumerate(IMG_KEYS):
        if k < 3:
            offset_x, offset_y = k * resize_dim[0], 0
        else:
            offset_x, offset_y = (k - 3) * resize_dim[0], resize_dim[1]
            
        if 'BACK' in key:
            sign = -1
        else:
            sign = 1

        for corners, cls in zip(pred_corners, pred_class):
            cam_corners = get_cam_corners(
                corners,
                info['cam_infos'][key]['calibrated_sensor']['translation'],
                info['cam_infos'][key]['calibrated_sensor']['rotation'],
                info['cam_infos'][key]['calibrated_sensor']['camera_intrinsic'])
            
            lines = get_3d_lines(cam_corners,offset_x,offset_y,resize_div)
            num_lines = len(lines)
            if num_lines > 0:
                num_lines = num_lines - 3
                if lines[0][0][0] > offset_x and lines[0][0][0] < (resize_dim[0]+offset_x) \
                    and lines[0][1][0] > offset_y and lines[0][1][0] < (offset_y + resize_dim[1]): 
                    if sign < 0:
                        tmp_point= resize_dim[0] + offset_x - lines[num_lines][0][1] + offset_x 
                        plt.text(tmp_point, lines[num_lines][1][1], f'{cls}', fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
                    else:
                        plt.text(lines[num_lines][0][1], lines[num_lines][1][1], f'{cls}', fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))                       
                    
                    for line in lines:
                        if sign < 0:
                            tmp_line = [resize_dim[0] + offset_x - point + offset_x for point in line[0]]
                            ax.plot(
                                tmp_line,
                                line[1],
                                    c=cm.get_cmap('tab10')(show_classes.index(cls)))
                        else:
                            ax.plot(
                                line[0],
                                line[1],
                                    c=cm.get_cmap('tab10')(show_classes.index(cls)))

def demo(
    idx = 0,
    nusc_results_file ='',
    dump_file ='',
    threshold=0.5,
    show_range=60,
    show_classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ],
):
    # Set cameras
    IMG_KEYS = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    
    import sys
    import os

    # 현재 파일의 절대 경로를 가져옵니다.
    current_file_path = os.path.abspath(__file__)
    # 현재 파일의 디렉토리 경로를 가져옵니다.
    current_dir = os.path.dirname(current_file_path)
    # 목표 디렉토리로 가는 상대 경로를 설정합니다.
    relative_path_to_add = os.path.join(current_dir, '../../CRN')
    # 상대 경로를 절대 경로로 변환합니다.
    absolute_path_to_add = os.path.abspath(relative_path_to_add)
    
    nusc_results_file = os.path.join(absolute_path_to_add, 'outputs/det/CRN_r18_256x704_128x128_4key/results_nusc.json')
    infos = mmcv.load('data/nuScenes/nuscenes_infos_val.pkl')
    
    fig = plt.figure(figsize=(30, 10))
    resize_div = 2
    resize_dim = ((int)(1600/resize_div), (int)(900/resize_div))  # 원하는 크기로 설정
    
    print('len(infos)',len(infos))
    for i in range(len(infos)):
        if i < len(infos):
            if i >= 0:
                dump_file = os.path.join(absolute_path_to_add,f'results/CRN_r18_256x704_128x128_4key_{i}.jpg')
            # assert 
            # Get data from dataset
                results = mmcv.load(nusc_results_file)['results']
                info = infos[i]
                lidar_path = info['lidar_infos']['LIDAR_TOP']['filename']
                lidar_points = np.fromfile(os.path.join('data/nuScenes', lidar_path),
                                        dtype=np.float32,
                                        count=-1).reshape(-1, 5)[..., :4]
                lidar_calibrated_sensor = info['lidar_infos']['LIDAR_TOP'][
                    'calibrated_sensor']
                # Get point cloud
                pts = lidar_points.copy()
                ego2global_rotation = np.mean(
                    [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in IMG_KEYS],
                    0)
                ego2global_translation = np.mean([
                    info['cam_infos'][cam]['ego_pose']['translation'] for cam in IMG_KEYS
                ], 0)
                lidar_points = LidarPointCloud(lidar_points.T)
                lidar_points.rotate(
                    Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
                lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))
                pts = lidar_points.points.T

                #Get GT corners
                gt_corners = []
                for i in range(len(info['ann_infos'])):
                    if map_name_from_general_to_detection[
                            info['ann_infos'][i]['category_name']] in show_classes:
                        box = get_ego_box(
                            dict(
                                size=info['ann_infos'][i]['size'],
                                rotation=info['ann_infos'][i]['rotation'],
                                translation=info['ann_infos'][i]['translation'],
                            ), ego2global_rotation, ego2global_translation)
                        if np.linalg.norm(box[:2]) <= show_range:
                            corners = get_corners(box[None])[0]
                            gt_corners.append(corners)

                # Get prediction corners
                pred_corners, pred_class = [], []
                for box in results[info['sample_token']]:
                    if box['detection_score'] >= threshold and box[
                            'detection_name'] in show_classes:
                        box3d = get_ego_box(box, ego2global_rotation,
                                            ego2global_translation)
                        box3d[2] += 0.5 * box3d[5]  # NOTE
                        if np.linalg.norm(box3d[:2]) <= show_range:
                            corners = get_corners(box3d[None])[0]
                            pred_corners.append(corners)
                            pred_class.append(box['detection_name'])

                # Set figure size
                
                fig.clf()
                ax = fig.add_subplot(1, 2, 1)
                img = merge_images(info,IMG_KEYS,resize_dim)
                ax.axis('off')
                ax.set_xlim(0, resize_dim[0]*3)
                ax.set_ylim(resize_dim[1]*2, 0)                
                plt.imshow(img)
                
                visualize_predictions(ax,info,IMG_KEYS,resize_dim, pred_corners,pred_class,show_classes, resize_div)
       
                # # Draw BEV 
                ax = fig.add_subplot(1, 2, 2)

                # Set BEV attributes
                ax.set_title('LIDAR_TOP')
                ax.axis('equal')
                ax.set_xlim(-60, 60)
                ax.set_ylim(-60, 60)

                # Draw point cloud
                ax.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')

                # Draw BEV GT boxes
                for corners in gt_corners:
                    lines = get_bev_lines(corners)
                    for line in lines:
                        ax.plot([-x for x in line[1]],
                                line[0],
                                c='r',
                                label='ground truth')

                # Draw BEV predictions
                for corners in pred_corners:
                    lines = get_bev_lines(corners)
                    for line in lines:
                        ax.plot([-x for x in line[1]], line[0], c='g', label='prediction')

                # Set legend
                handles, labels = fig.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(),
                        by_label.keys(),
                        loc='upper right',
                        framealpha=1)
                
                # plt.show()
                # Save figure
                fig.tight_layout(w_pad=0, h_pad=2)
                fig.savefig(dump_file)
                


# Manually trigger garbage collection
gc.collect()


def make_mf4(enable_save_avi):
    
    if enable_save_avi == True:
        # 동영상 작성
        output_dir = './results'
        video_filename = os.path.join(output_dir, f'CRN_r18_256x704_128x128_4key.mp4')
        frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')])

        # 첫 번째 프레임에서 프레임 크기 가져오기
        frame = cv2.imread(frame_files[0])
        height, width, layers = frame.shape

        # 동영상 작성자 초기화
        video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        # 각 프레임을 동영상 작성자에 추가
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video.write(frame)

        # 동영상 작성 종료
        video.release()

        print(f"Video saved as {video_filename}")

enable_save_avi = True

if __name__ == '__main__':
    # args = parse_args()
    demo()
    make_mf4(enable_save_avi)