"""
KNN-augmented Sentence-BERT Price Predictor.

End-to-end model that uses FAISS to find K nearest neighbors in SBERT embedding
space, then concatenates query + neighbor features for price regression.
Trained with DDP, mixed precision, and cosine LR schedule.
"""
import os, time, warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from sentence_transformers import SentenceTransformer
from transformers import get_cosine_schedule_with_warmup
from sklearn.preprocessing import MinMaxScaler

# ── Config ──────────────────────────────────────────────────────────
K_NEIGHBORS = 5
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
TRAIN_CSV = '/home/user3/amazon/train_split_final.csv'
VAL_CSV = '/home/user3/amazon/val_split_final.csv'
CHECKPOINT_IN = '/home/user3/amazon/best_finetuned_model_state_dict.pth'
CHECKPOINT_OUT = '/home/user3/amazon/best_with_image_KNN.pth'
BATCH_SIZE = 32
LR_TRANSFORMER = 1e-5
LR_MLP = 1e-4
EPOCHS = 10
DROPOUT = 0.3


# ── Utilities ───────────────────────────────────────────────────────
def smape(y_pred, y_true, eps=1e-8):
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + eps)) * 100


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def find_neighbors(index, queries, k, is_training=False):
    search_k = k + 1 if is_training else k
    _, indices = index.search(queries, k=search_k)
    return indices[:, 1:] if is_training else indices


# ── Dataset ─────────────────────────────────────────────────────────
class PriceDataset(Dataset):
    def __init__(self, query_df, ref_df, neighbor_indices):
        self.query = query_df.reset_index(drop=True)
        self.ref = ref_df.reset_index(drop=True)
        self.neighbors = neighbor_indices

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        row = self.query.iloc[idx]
        nbr = self.ref.iloc[self.neighbors[idx]]
        return {
            'query_text': row['desc_for_llm'],
            'query_features': torch.tensor([row['Count'], row['oz'], row['fl_oz']], dtype=torch.float32),
            'neighbor_texts': nbr['desc_for_llm'].tolist(),
            'neighbor_features': torch.tensor(nbr[['Count', 'oz', 'fl_oz', 'price']].values, dtype=torch.float32),
            'target': torch.tensor([row['price']], dtype=torch.float32),
        }


def make_collate(tokenizer):
    def collate(batch):
        q_texts = [b['query_text'] for b in batch]
        n_texts = [t for b in batch for t in b['neighbor_texts']]
        return {
            'query_tokens': tokenizer(q_texts, padding=True, truncation=True, return_tensors='pt', max_length=128),
            'neighbor_tokens': tokenizer(n_texts, padding=True, truncation=True, return_tensors='pt', max_length=128),
            'query_features': torch.stack([b['query_features'] for b in batch]),
            'neighbor_features': torch.stack([b['neighbor_features'] for b in batch]),
            'targets': torch.stack([b['target'] for b in batch]),
        }
    return collate


# ── Model ───────────────────────────────────────────────────────────
class KNNPricePredictor(nn.Module):
    def __init__(self, model_name, k=K_NEIGHBORS, dropout=DROPOUT):
        super().__init__()
        self.sbert = SentenceTransformer(model_name)
        self.sbert.max_seq_length = 64
        dim = self.sbert.get_sentence_embedding_dimension()
        mlp_in = (dim + 3) + k * (dim + 4)  # query(emb+3feat) + k*neighbor(emb+4feat)
        self.head = nn.Sequential(
            nn.Linear(mlp_in, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, batch):
        device = batch['query_features'].device
        q_tok = {k: v.to(device) for k, v in batch['query_tokens'].items()}
        n_tok = {k: v.to(device) for k, v in batch['neighbor_tokens'].items()}

        q_emb = self.sbert(q_tok)['sentence_embedding']
        n_emb = self.sbert(n_tok)['sentence_embedding']

        bs, dim = q_emb.shape
        k = n_emb.shape[0] // bs
        n_emb = n_emb.view(bs, k, dim)
        n_feat = batch['neighbor_features'].to(device)

        # Pad/trim to K_NEIGHBORS
        if k != K_NEIGHBORS:
            pad_emb = torch.zeros(bs, K_NEIGHBORS, dim, device=device, dtype=n_emb.dtype)
            pad_feat = torch.zeros(bs, K_NEIGHBORS, n_feat.shape[2], device=device, dtype=n_feat.dtype)
            m = min(k, K_NEIGHBORS)
            pad_emb[:, :m] = n_emb[:, :m]
            pad_feat[:, :m] = n_feat[:, :m]
            n_emb, n_feat = pad_emb, pad_feat

        query_vec = torch.cat([q_emb, batch['query_features'].to(device)], dim=1)
        nbr_vec = torch.cat([n_emb, n_feat], dim=2).view(bs, -1)
        return self.head(torch.cat([query_vec, nbr_vec], dim=1))


# ── DDP helpers ─────────────────────────────────────────────────────
def move_to(batch, device):
    batch['query_tokens'] = {k: v.to(device) for k, v in batch['query_tokens'].items()}
    batch['neighbor_tokens'] = {k: v.to(device) for k, v in batch['neighbor_tokens'].items()}
    for key in ('query_features', 'neighbor_features', 'targets'):
        batch[key] = batch[key].to(device)
    return batch


def gather_tensors(local_t):
    """All-gather tensors across DDP ranks. Returns merged tensor on rank 0, None elsewhere."""
    ws = dist.get_world_size()
    local_len = torch.tensor([local_t.shape[0]], device=local_t.device)
    lengths = [torch.zeros_like(local_len) for _ in range(ws)]
    dist.all_gather(lengths, local_len)
    lengths = [int(x.item()) for x in lengths]
    max_len = max(lengths)
    C = local_t.shape[1] if local_t.ndim == 2 else 1
    padded = torch.zeros(max_len, C, device=local_t.device)
    padded[:local_t.shape[0]] = local_t
    gathered = [torch.zeros_like(padded) for _ in range(ws)]
    dist.all_gather(gathered, padded)
    if dist.get_rank() == 0:
        return torch.cat([g[:l] for g, l in zip(gathered, lengths) if l > 0]).cpu()
    return None


# ── Training ────────────────────────────────────────────────────────
def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    is_main = rank == 0

    if is_main:
        SentenceTransformer(MODEL_NAME)  # cache download
    dist.barrier()

    # Data
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    train_nbr = np.load('train_neighbor_indices.npy')
    val_nbr = np.load('val_neighbor_indices.npy')

    # Feature normalization
    for col in ('Count', 'oz', 'fl_oz'):
        train_df[col] = np.log1p(train_df[col])
        val_df[col] = np.log1p(val_df[col])
    scaler = MinMaxScaler().fit(train_df[['Count']])
    train_df['Count'] = scaler.transform(train_df[['Count']])
    val_df['Count'] = scaler.transform(val_df[['Count']])

    # Model + data loaders
    model = KNNPricePredictor(MODEL_NAME).to(device)
    collate = make_collate(model.sbert.tokenizer)

    train_sampler = DistributedSampler(PriceDataset(train_df, train_df, train_nbr), shuffle=True)
    val_sampler = DistributedSampler(PriceDataset(val_df, train_df, val_nbr), shuffle=False)
    train_loader = DataLoader(PriceDataset(train_df, train_df, train_nbr),
                              batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(PriceDataset(val_df, train_df, val_nbr),
                            batch_size=BATCH_SIZE * 2, sampler=val_sampler, collate_fn=collate, pin_memory=True)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.sbert.parameters(), 'lr': LR_TRANSFORMER},
        {'params': model.head.parameters(), 'lr': LR_MLP},
    ])
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    # Checkpoint loading
    start_epoch, best_smape_val = 0, float('inf')
    if CHECKPOINT_IN and os.path.exists(CHECKPOINT_IN):
        ckpt = torch.load(CHECKPOINT_IN, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch']
            best_smape_val = ckpt.get('best_smape', float('inf'))
            if is_main:
                print(f"Resumed from epoch {start_epoch}, best SMAPE: {best_smape_val:.4f}%")
        else:
            model.load_state_dict(ckpt)
            if is_main:
                print("Loaded weights-only checkpoint.")

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    criterion = nn.MSELoss()
    grad_scaler = GradScaler()

    if is_main:
        print(f"\nTraining on {world_size} GPUs, batch={BATCH_SIZE}/GPU, epochs={EPOCHS}")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            batch = move_to(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = criterion(model(batch), torch.log1p(batch['targets']))
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            scheduler.step()
            grad_scaler.update()
            total_loss += loss.item()

            if is_main and (i + 1) % 100 == 0:
                print(f"  E{epoch+1} step {i+1}/{len(train_loader)}, loss: {total_loss/(i+1):.4f}")

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to(batch, device)
                with autocast():
                    out = model(batch)
                preds.append(torch.expm1(out))
                targets.append(batch['targets'])

        g_preds = gather_tensors(torch.cat(preds) if preds else torch.empty(0, 1, device=device))
        g_targets = gather_tensors(torch.cat(targets) if targets else torch.empty(0, 1, device=device))

        if is_main:
            val_smape = smape(g_preds.numpy().flatten(), g_targets.numpy().flatten())
            print(f"{'─'*60}\nEpoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.6f} | "
                  f"Val SMAPE: {val_smape:.4f}% | Time: {time.time()-t0:.1f}s\n{'─'*60}")

            if val_smape < best_smape_val:
                best_smape_val = val_smape
                torch.save({
                    'epoch': epoch + 1, 'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_smape': best_smape_val,
                }, CHECKPOINT_OUT)
                print(f"  → Saved best model (SMAPE: {best_smape_val:.4f}%)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
