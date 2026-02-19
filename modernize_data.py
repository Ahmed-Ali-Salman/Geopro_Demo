import pandas as pd
import numpy as np
import laspy
import os
import shutil
from datetime import datetime, timedelta
from scipy.spatial import cKDTree

# Target date: "Early 2026"
TARGET_DATE = datetime(2026, 2, 18, 10, 0, 0)

def modernize():
    print("🚀 Starting Data Modernization...")
    
    # Check inputs
    pos_path = 'data/sample/data1/pos.txt'
    pts_path = 'data/sample/data1/pts.txt'
    
    if not os.path.exists(pos_path):
        print(f"❌ Error: {pos_path} not found.")
        return

    # 1. Load POS
    print("Reading trajectory data...")
    # Explicitly set names to avoid header parsing issues
    col_names = ['Filename', 'Width', 'Height', 'Latitude', 'Longitude', 'Altitude', 'Roll', 'Pitch', 'Heading', 'X', 'Y']
    pos = pd.read_csv(pos_path, delim_whitespace=True, skiprows=1, names=col_names)

    print(f"Loaded {len(pos)} trajectory points.")
    print(f"Sample Filename: {pos.iloc[0]['Filename']}") 

    # 2. Update Timestamps and Rename Images
    timestamps = []
    new_filenames = []
    
    # Extract original date from first file
    first_fname = str(pos.iloc[0]['Filename'])
    try:
        # LB3-20140924-033636-000000_000900.jpg
        original_date_str = first_fname.split('-')[1] # 20140924
        base_date = datetime.strptime(original_date_str, "%Y%m%d")
    except Exception as e:
        print(f"⚠️ Failed to parse date from {first_fname}: {e}")
        base_date = datetime(2014, 9, 24)

    time_offset = TARGET_DATE - base_date
    print(f"Time offset applied: {time_offset.days} days")

    # Build source image map
    src_img_dir = 'data/sample/data1/images'
    src_map = {} # '000900' -> 'path/ladybug_...000900.jpg'
    if os.path.exists(src_img_dir):
        for f in os.listdir(src_img_dir):
            if f.endswith('.jpg'):
                # Extract suffix: ladybug_panoramic_000900.jpg -> 000900
                # Split by _ and take last part, strip .jpg
                try:
                    suffix = f.split('_')[-1].split('.')[0]
                    src_map[suffix] = os.path.join(src_img_dir, f)
                except:
                    pass
    
    print("Processing images...")
    processed_count = 0
    for idx, row in pos.iterrows():
        fname = str(row['Filename'])
        try:
            parts = fname.split('-')
            d_str = parts[1]
            t_str = parts[2]
            
            orig_dt = datetime.strptime(f"{d_str}{t_str}", "%Y%m%d%H%M%S")
            new_dt = orig_dt + time_offset
            
            # Construct new filename LB3-2026...
            new_fname = fname.replace(d_str, new_dt.strftime("%Y%m%d"))
            
            # Extract suffix from POS filename: ..._000900.jpg
            suffix = fname.split('_')[-1].split('.')[0]
            
            src = src_map.get(suffix)
            dst = f"data/images/{new_fname}"
            
            if src and os.path.exists(src):
                shutil.copy2(src, dst)
                processed_count += 1
            else:
                # Debug only first few failures
                if processed_count < 3 and src is None: 
                    print(f"Warning: No match for suffix {suffix} (pos: {fname})")
            
            new_filenames.append(new_fname)
            timestamps.append(new_dt)
            
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            new_filenames.append(fname)
            timestamps.append(datetime.now())

    print(f"Copied {processed_count} images.")

    # 3. Write Meta CSV
    print("Writing metadata...")
    meta = pd.DataFrame({
        'image_filename': new_filenames,
        'timestamp': [t.isoformat() for t in timestamps],
        'latitude': pos['Latitude'],
        'longitude': pos['Longitude'],
        'altitude': pos['Altitude'],
        'roll': pos['Roll'],
        'pitch': pos['Pitch'],
        'yaw': pos['Heading']
    })
    meta.to_csv('data/meta.csv', index=False)
    print("✅ Created data/meta.csv")

    # 4. Convert Point Cloud
    print("Reading point cloud (this may take a moment)...")
    try:
        pts_df = pd.read_csv(pts_path, delim_whitespace=True, header=None, names=['x', 'y', 'z'], nrows=None)
        print(f"Loaded {len(pts_df)} points.")
        
        # Header
        header = laspy.LasHeader(point_format=3, version="1.4")
        # Auto-scale
        header.scales = [0.001, 0.001, 0.001]
        header.offsets = [pts_df['x'].min(), pts_df['y'].min(), pts_df['z'].min()]
        
        las = laspy.LasData(header)
        las.x = pts_df['x']
        las.y = pts_df['y']
        las.z = pts_df['z']
        
        # Spatial sync
        print("Syncing GPS time...")
        # pos.txt X Y are likely projected coords matching lidar
        traj_xy = pos[['X', 'Y']].values
        cloud_xy = pts_df[['x', 'y']].values
        
        tree = cKDTree(traj_xy)
        dists, idxs = tree.query(cloud_xy)
        
        time_refs = np.array([t.timestamp() for t in timestamps])
        point_times = time_refs[idxs]
        
        las.gps_time = point_times
        
        # Force file creation
        out_path = 'data/lidar.las'
        if os.path.exists(out_path):
            os.remove(out_path)
            
        las.write(out_path)
        
        if os.path.exists(out_path):
            print(f"✅ Created data/lidar.las ({os.path.getsize(out_path)} bytes)")
        else:
            print("❌ Failed to write lidar.las (file missing after write)")
        
    except Exception as e:
        print(f"❌ Failed to process lidar: {e}")

if __name__ == "__main__":
    modernize()
