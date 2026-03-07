from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from Bio.SeqRecord import SeqRecord


class ESMEmbedder:
    """Compute ESM embeddings with caching and resumable shard processing."""

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        pooling: str,
        device: str | None,
        cache_dir: str | Path,
        logger: logging.Logger,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.pooling = pooling
        self.logger = logger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers not installed. Install with: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_sequences(self, ids: list[str], seqs: list[str]) -> np.ndarray:
        """Embed a batch of sequences and return numpy matrix."""
        encoded = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            attn_mask = encoded["attention_mask"].unsqueeze(-1)
            sum_hidden = (hidden * attn_mask).sum(dim=1)
            lengths = attn_mask.sum(dim=1).clamp(min=1)
            emb = sum_hidden / lengths

        return emb.detach().cpu().numpy().astype(np.float32)

    def embed_records_with_cache(
        self,
        records: list[SeqRecord],
        dataset_name: str,
        shard_size: int,
    ) -> tuple[list[str], np.ndarray]:
        """Embed records in shards with checkpoint/resume support."""
        ids = [r.id for r in records]
        seqs = [str(r.seq) for r in records]

        done_file = self.cache_dir / f"{dataset_name}_done_ids.json"
        done_ids = set()
        if done_file.exists():
            done_ids = set(json.loads(done_file.read_text(encoding="utf-8")))

        all_ids: list[str] = []
        all_embs: list[np.ndarray] = []

        for shard_idx, start in enumerate(range(0, len(ids), shard_size)):
            end = min(start + shard_size, len(ids))
            shard_ids = ids[start:end]
            shard_seqs = seqs[start:end]

            shard_path = self.cache_dir / f"{dataset_name}_shard_{shard_idx:05d}.npz"
            if shard_path.exists():
                payload = np.load(shard_path, allow_pickle=True)
                all_ids.extend(payload["ids"].tolist())
                all_embs.append(payload["embeddings"])
                continue

            to_run = [(i, s) for i, s in zip(shard_ids, shard_seqs) if i not in done_ids]
            if not to_run:
                continue

            shard_embs: list[np.ndarray] = []
            shard_out_ids: list[str] = []
            for b in range(0, len(to_run), self.batch_size):
                batch = to_run[b : b + self.batch_size]
                b_ids = [x[0] for x in batch]
                b_seqs = [x[1] for x in batch]
                emb = self.embed_sequences(b_ids, b_seqs)
                shard_embs.append(emb)
                shard_out_ids.extend(b_ids)
                done_ids.update(b_ids)
                done_file.write_text(json.dumps(sorted(done_ids)), encoding="utf-8")

            shard_arr = np.vstack(shard_embs)
            np.savez_compressed(shard_path, ids=np.array(shard_out_ids), embeddings=shard_arr)
            all_ids.extend(shard_out_ids)
            all_embs.append(shard_arr)
            self.logger.info("Embedded shard %d (%d seqs)", shard_idx, len(shard_out_ids))

        if not all_embs:
            raise RuntimeError(f"No embeddings produced for dataset {dataset_name}")
        return all_ids, np.vstack(all_embs)


def write_runtime_log(path: str | Path, lines: Iterable[str]) -> None:
    """Write embedding runtime log lines."""
    with Path(path).open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
