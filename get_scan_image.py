import pickle as pkl
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from lib.closed import SingleScanDataset
from lib.utils.misc import exists, fix_seeds, mkdir, save_image
from lib.utils.viz import render_p3d

if __name__ == '__main__':
    parser = ArgumentParser(description='CloSeNet scan renderer only (single view)')
    parser.add_argument(
        '--scan_path',
        type=str,
        default='./assets/10001_1923.npz',
        help='Path to input .npz file',
    )
    parser.add_argument(
        '--output', type=str, default='./out/demo_render_only', help='Path to output directory'
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for dataloader')
    args = parser.parse_args()

    assert exists(args.scan_path), f'{args.scan_path} does not exist'
    assert args.scan_path.endswith('.npz'), 'Input scan should be a .npz file'

    save_dir = mkdir(args.output)
    fix_seeds(args.seed)

    # Load the scan dataset (just one scan here)
    scan = SingleScanDataset(args.scan_path)
    loader = scan.get_loader(batch_size=2048, num_workers=args.n_workers)

    # Collect points and colors from scan
    points = scan.points[:, :3]
    colors = scan.points[:, 3:6]
    faces = scan.faces

    # Convert to tensors
    points = torch.from_numpy(points).float().to(args.device)
    colors = torch.from_numpy(colors).float().to(args.device)
    faces = torch.from_numpy(faces).long().to(args.device)

    # Render only one view (azimuth = 0)
    input_scan_img = render_p3d(
        vertices=points,
        faces=faces,
        colors=colors,
        azimuths=[0],  # Only one view
        resolution=1024,
        input_type='mesh',
    )[0, ..., :3] * 255.0  # Get the first (and only) rendered image

    # Save rendered input scan image
    img_path = save_dir / 'input_scan_only.png'
    print(f'Saving rendered input scan image to {img_path}')
    save_image(input_scan_img, img_path)
