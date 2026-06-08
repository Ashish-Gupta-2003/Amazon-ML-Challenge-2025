"""
DINOv2 Image Embedding Generator with DDP support.

Extracts 768-d embeddings from DINOv2 ViT-B14 for all images in a folder.
Supports multi-GPU via torch DDP, handles corrupted images with auto-redownload.
"""
import os, argparse, pickle, subprocess, time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd


def redownload_image(sid, url_map, folder, retries=3):
    """Re-download a corrupted image. Returns path on success, None on failure."""
    if sid not in url_map:
        return None
    target = os.path.join(folder, f"{sid}.jpg")
    os.makedirs(folder, exist_ok=True)
    for _ in range(retries):
        try:
            res = subprocess.run(
                ["wget", "-q", "--timeout=15", "--tries=1", "-O", target, url_map[sid]],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if res.returncode == 0 and os.path.exists(target):
                return target
        except Exception:
            pass
        time.sleep(2)
    return None


class ImagePathDataset(Dataset):
    """Dataset that yields (file_path, sample_id) tuples."""
    def __init__(self, folder, exts=('.jpg', '.jpeg', '.png')):
        self.files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        return os.path.join(self.folder, fname), os.path.splitext(fname)[0]


def main_worker(rank, world_size, args):
    use_cuda = torch.cuda.is_available() and world_size > 0
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(device)
    if world_size > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend, init_method='tcp://127.0.0.1:29500',
                                world_size=world_size, rank=rank)

    log = lambda msg: print(msg) if rank == 0 else None

    # Load DINOv2
    log(f"Loading DINOv2 on {device}...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    model.eval().to(device)
    if use_cuda and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    # Image URL map for re-downloads
    df = pd.read_csv(args.csv_path)
    url_map = dict(zip(df['sample_id'].astype(str), df['image_link']))

    # DataLoader
    dataset = ImagePathDataset(args.image_folder)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    batch_size = max(1, args.batch_size // max(1, world_size))
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=use_cuda,
        collate_fn=lambda batch: (list(zip(*batch)))  # → ([paths], [sids])
    )

    if sampler:
        sampler.set_epoch(0)

    # Extract embeddings
    embeddings = OrderedDict()
    log(f"Processing {len(loader)} batches (batch_size={batch_size})...")

    for paths, sids in tqdm(loader, desc=f"rank{rank}", disable=(rank != 0)):
        tensors, valid_sids = [], []
        for path, sid in zip(paths, sids):
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(preprocess(img))
                valid_sids.append(sid)
            except (UnidentifiedImageError, Exception) as e:
                print(f"[rank {rank}] Bad image {Path(path).name}: {e}")
                retry = redownload_image(sid, url_map, args.retry_folder)
                if retry:
                    try:
                        img = Image.open(retry).convert("RGB")
                        tensors.append(preprocess(img))
                        valid_sids.append(sid)
                    except Exception:
                        pass

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            embs = model(batch).cpu().numpy()
        for sid, emb in zip(valid_sids, embs):
            embeddings[str(sid)] = emb

    # Save per-rank results
    os.makedirs(args.tmp_dir, exist_ok=True)
    tmp_path = os.path.join(args.tmp_dir, f"emb_rank{rank}.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(embeddings, f)
    log(f"Rank {rank}: saved {len(embeddings)} embeddings")

    if world_size > 1:
        dist.barrier()

    # Rank 0 merges all partials
    if rank == 0 or (world_size == 1):
        merged = {}
        for r in range(max(1, world_size)):
            p = os.path.join(args.tmp_dir, f"emb_rank{r}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    merged.update(pickle.load(f))
                os.remove(p)

        out_path = os.path.join(args.output_dir, args.output_file)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(merged, f)
        log(f"Final: {len(merged)} embeddings → {out_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="DINOv2 image embedding generator")
    p.add_argument("--image-folder", required=True, help="Folder with downloaded images")
    p.add_argument("--csv-path", required=True, help="CSV with sample_id, image_link columns")
    p.add_argument("--output-dir", default="./", help="Output directory")
    p.add_argument("--output-file", default="embeddings.pkl", help="Output filename")
    p.add_argument("--tmp-dir", default="./tmp_embeddings", help="Temp dir for per-rank outputs")
    p.add_argument("--retry-folder", default="./redownloaded", help="Folder for re-downloaded images")
    p.add_argument("--batch-size", type=int, default=64, help="Global batch size")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--world-size", type=int, default=2, help="Number of GPUs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable — running single-process CPU mode.")
        args.world_size = 1

    if args.world_size == 1:
        main_worker(0, 1, args)
    else:
        mp.spawn(main_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
