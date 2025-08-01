import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lib.closed import SingleScanDataset
from lib.utils.misc import exists, fix_seeds, mkdir, save_image
from lib.utils.viz import render_p3d

if __name__ == '__main__':
    parser = ArgumentParser(description='CloSeNet batch scan renderer (single view)')
    parser.add_argument(
        '--scan_folder',
        '-s',
        type=str,
        default='./assets/close-di',
        help='Folder containing .npz scan files',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='./out/close_image_scan',
        help='Folder to save rendered images',
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()

    scan_folder = Path(args.scan_folder)
    assert scan_folder.exists(), f'{scan_folder} does not exist'

    save_dir = mkdir(args.output)
    fix_seeds(args.seed)

    npz_files = sorted(scan_folder.glob('*.npz'))
    if not npz_files:
        raise RuntimeError(f'No .npz files found in {scan_folder}')

    for scan_path in tqdm(npz_files, desc='Rendering scans'):
        try:
            scan = SingleScanDataset(str(scan_path))
            loader = scan.get_loader(batch_size=2048, num_workers=4)  # Not used, but required for consistency

            points = scan.points[:, :3]
            colors = scan.points[:, 3:6]
            faces = scan.faces

            points = torch.from_numpy(points).float().to(args.device)
            colors = torch.from_numpy(colors).float().to(args.device)
            faces = torch.from_numpy(faces).long().to(args.device)

            rendered = render_p3d(
                vertices=points,
                faces=faces,
                colors=colors,
                azimuths=[0],
                resolution=1024,
                input_type='mesh',
            )[0, ..., :3] * 255.0

            output_name = scan_path.stem + '.png'
            save_path = save_dir / output_name
            save_image(rendered, save_path)
        except Exception as e:
            print(f'Error processing {scan_path.name}: {e}')
