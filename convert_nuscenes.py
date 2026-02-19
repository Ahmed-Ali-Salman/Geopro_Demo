
import json
import os
import shutil
import numpy as np
import laspy
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Config
DATA_ROOT = Path("data/v1.0-mini")
OUTPUT_DIR = Path("data/nuscenes_processed")
SCENE_TOKEN = "cc8c0bf57f984915a77078b10eb33198"  # scene-0061
# Singapore One-North anchor
ANCHOR_LAT = 1.299
ANCHOR_LON = 103.787
METERS_PER_DEG_LAT = 111320.0

def load_table(name):
    with open(DATA_ROOT / f"v1.0-mini/{name}.json", 'r') as f:
        return json.load(f)

def get_rotation_matrix(quaternion):
    # NuScenes quaternion is [w, x, y, z]
    # Scipy expects [x, y, z, w]
    return R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()

def get_euler_angles(quaternion):
    r = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    return r.as_euler('xyz', degrees=True)

def main():
    print("🚀 Starting NuScenes Conversion...")
    
    # 1. Load Tables
    print("Loading metadata...")
    scenes = {s['token']: s for s in load_table('scene')}
    samples = {s['token']: s for s in load_table('sample')}
    sample_data = {s['token']: s for s in load_table('sample_data')}
    ego_pose = {s['token']: s for s in load_table('ego_pose')}
    calibrated_sensor = {s['token']: s for s in load_table('calibrated_sensor')}
    
    print("Indexing sample_data...")
    sample_to_data = {}
    for sd in sample_data.values():
        if sd['is_key_frame']:
            s_tok = sd['sample_token']
            if s_tok not in sample_to_data:
                sample_to_data[s_tok] = {}
            
            # Identify sensor from filename or other means? 
            # filename usually 'samples/SENSOR_NAME/...'
            # Let's rely on path
            path = sd['filename']
            if 'LIDAR_TOP' in path:
                sample_to_data[s_tok]['LIDAR_TOP'] = sd['token']
            elif 'CAM_FRONT' in path and 'CAM_FRONT_LEFT' not in path and 'CAM_FRONT_RIGHT' not in path:
                sample_to_data[s_tok]['CAM_FRONT'] = sd['token']
    
    scene = scenes[SCENE_TOKEN]
    print(f"Processing Scene: {scene['name']}")
    
    # 2. Setup Output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "images").mkdir(parents=True)
    
    # 3. Iterate Samples
    curr_token = scene['first_sample_token']
    
    # Store data for LAS and CSV
    all_points = []
    meta_rows = []
    
    origin_trans = None
    
    sample_count = 0
    
    while curr_token:
        sample = samples[curr_token]
        
        # Get Sensors
        try:
            lidar_token = sample_to_data[curr_token]['LIDAR_TOP']
            cam_token = sample_to_data[curr_token]['CAM_FRONT']
        except KeyError:
            print(f"Skipping sample {curr_token} - Missing LIDAR or CAM")
            curr_token = sample['next']
            continue
        
        lidar_sd = sample_data[lidar_token]
        cam_sd = sample_data[cam_token]
        
        # --- Process LiDAR ---
        lidar_path = DATA_ROOT / lidar_sd['filename']
        lidar_pose = ego_pose[lidar_sd['ego_pose_token']]
        lidar_calib = calibrated_sensor[lidar_sd['calibrated_sensor_token']]
        
        # Transform Matrices
        # Sensor -> Ego
        R_s2e = get_rotation_matrix(lidar_calib['rotation'])
        T_s2e = np.array(lidar_calib['translation'])
        
        # Ego -> Global
        R_e2g = get_rotation_matrix(lidar_pose['rotation'])
        T_e2g = np.array(lidar_pose['translation'])
        
        # Set Origin if first frame
        if origin_trans is None:
            origin_trans = T_e2g
            print(f"Origin set to: {origin_trans}")

        # Read Binary PCM (x, y, z, intensity, ring)
        # NuScenes .bin is N x 5 float32
        scan = np.fromfile(str(lidar_path), dtype=np.float32).reshape((-1, 5))
        points_s = scan[:, :3] # x,y,z
        
        # Transform Points: P_global = R_e2g * (R_s2e * P_s + T_s2e) + T_e2g
        points_e = (points_s @ R_s2e.T) + T_s2e
        points_g = (points_e @ R_e2g.T) + T_e2g
        
        # Transform to Local LAS (P_global - Origin)
        points_local = points_g - origin_trans
        
        all_points.append(points_local)
        
        # --- Process Camera ---
        cam_path = DATA_ROOT / cam_sd['filename']
        cam_pose = ego_pose[cam_sd['ego_pose_token']]
        cam_calib = calibrated_sensor[cam_sd['calibrated_sensor_token']]
        
        # Camera Global Pose = Ego Pose * Sensor Extrinsics
        R_c2e = get_rotation_matrix(cam_calib['rotation'])
        T_c2e = np.array(cam_calib['translation'])
        
        R_e2g = get_rotation_matrix(cam_pose['rotation'])
        T_e2g = np.array(cam_pose['translation'])
        
        # GLOBAL Camera Rotation/Translation
        # R_cg = R_e2g @ R_c2e
        # T_cg = R_e2g @ T_c2e + T_e2g
        R_cg = R_e2g @ R_c2e
        T_cg = (R_e2g @ T_c2e) + T_e2g
        
        # Calculate Camera Lat/Lon based on offset from origin
        # Note: We use Camera Global Pose for the CSV track
        
        track_offset = T_cg - origin_trans
        
        # Approx Lat/Lon
        lat = ANCHOR_LAT + (track_offset[1] / METERS_PER_DEG_LAT)
        lon = ANCHOR_LON + (track_offset[0] / (METERS_PER_DEG_LAT * np.cos(np.radians(ANCHOR_LAT))))
        alt = T_cg[2]
        
        # Orientation (Ego)
        r, p, y = get_euler_angles(cam_pose['rotation'])
        
        # VIEW MATRIX (Global LAS -> Camera)
        # Points in LAS are P_local = P_global - origin_trans
        # We need P_cam = R_view * P_local + T_view
        # P_cam = R_cg^T * (P_global - T_cg)
        #       = R_cg^T * (P_local + origin_trans - T_cg)
        #       = R_cg^T * P_local + R_cg^T * (origin_trans - T_cg)
        # So:
        # R_view = R_cg.T
        # T_view = R_cg.T @ (origin_trans - T_cg)
        
        R_view = R_cg.T
        T_view = R_cg.T @ (origin_trans - T_cg)
        
        # Copy Image
        new_img_name = f"NUSC_{sample['timestamp']}.jpg"
        shutil.copy2(cam_path, OUTPUT_DIR / "images" / new_img_name)
        
        row = {
            'image_filename': new_img_name,
            'timestamp': sample['timestamp'],
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'roll': r,
            'pitch': p,
            'yaw': y,
            # Extrinsics
            'tx': T_view[0], 'ty': T_view[1], 'tz': T_view[2],
            'r11': R_view[0,0], 'r12': R_view[0,1], 'r13': R_view[0,2],
            'r21': R_view[1,0], 'r22': R_view[1,1], 'r23': R_view[1,2],
            'r31': R_view[2,0], 'r32': R_view[2,1], 'r33': R_view[2,2],
            # Intrinsics (NuScenes provides them)
            # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            # cam_calib['camera_intrinsic'] is 3x3 list
        }
        
        if 'camera_intrinsic' in cam_calib:
            K = cam_calib['camera_intrinsic']
            row['fx'] = K[0][0]
            row['fy'] = K[1][1]
            row['cx'] = K[0][2]
            row['cy'] = K[1][2]
            
        meta_rows.append(row)
        
        curr_token = sample['next']
        sample_count += 1
        if sample_count % 10 == 0:
            print(f"Processed {sample_count} samples...")

    # 4. Write LAS
    print("Writing LAS file...")
    grand_cloud = np.vstack(all_points)
    
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [grand_cloud[:,0].min(), grand_cloud[:,1].min(), grand_cloud[:,2].min()]
    
    las = laspy.LasData(header)
    las.x = grand_cloud[:, 0]
    las.y = grand_cloud[:, 1]
    las.z = grand_cloud[:, 2]
    
    las.write(str(OUTPUT_DIR / "lidar.las"))
    
    # 5. Write CSV
    print("Writing CSV...")
    df = pd.DataFrame(meta_rows)
    df.to_csv(OUTPUT_DIR / "meta.csv", index=False)
    
    print("✅ Conversion Complete.")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
