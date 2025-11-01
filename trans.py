# scripts/convert_pose_to_rvec.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def convert_pose_file(input_path: Path, output_path: Path, degrees: bool = False, out_mm: bool = False) -> None:
    # try to read as plain numeric file, fall back to pandas for CSV with header
    try:
        data = np.loadtxt(input_path)
    except Exception:
        # use pandas to read common CSV formats
        df_in = pd.read_csv(input_path)
        data = df_in.values

    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 6:
        raise ValueError("Need at least 6 columns: x y z roll pitch yaw")

    xyz = data[:, :3].astype(float)
    rpy = data[:, 3:6].astype(float)

    # build rotation and quaternion
    rot = Rotation.from_euler('zyx', rpy, degrees=degrees)
    quat = rot.as_quat()  # SciPy returns [x, y, z, w]
    # reorder to [w, x, y, z]
    quat_wxyz = np.hstack([quat[:, 3:4], quat[:, 0:3]])

    # convert position unit if requested
    if out_mm:
        out_xyz = xyz * 1000.0
    else:
        out_xyz = xyz

    # construct DataFrame with specified columns and save as comma-separated CSV
    df_out = pd.DataFrame(
        np.hstack([quat_wxyz, out_xyz]),
        columns=["qw", "qx", "qy", "qz", "tx", "ty", "tz"],
    )
    df_out.to_csv(output_path, index=False, float_format='%.9f')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--degrees", action="store_true",
                        help="roll/pitch/yaw are in degrees")
    parser.add_argument("--out-mm", action="store_true",
                        help="output translation in millimeters (default: meters)")
    args = parser.parse_args()

    convert_pose_file(args.input, args.output, degrees=args.degrees, out_mm=args.out_mm)


if __name__ == "__main__":
    main()
