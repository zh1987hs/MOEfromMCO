from __future__ import annotations

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

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backend: str
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.backend = "transformers"
        except ImportError:
            try:
                import esm
            except ImportError as exc:
                raise RuntimeError(
                    "Neither transformers nor fair-esm is installed. "
                    "Install one of: pip install transformers  OR  pip install fair-esm"
                ) from exc

            if model_name != "facebook/esm2_t33_650M_UR50D":
                raise RuntimeError(
                    "fair-esm backend currently supports default ESM2 weights only. "
                    "Use transformers for arbitrary model_name, or keep default facebook/esm2_t33_650M_UR50D."
                )
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = alphabet.get_batch_converter()
            self.backend = "fair-esm"

        self.model.to(self.device)
        self.model.eval()
        self.logger.info("ESM backend=%s device=%s model=%s", self.backend, self.device, self.model_name)

    @torch.no_grad()
    def embed_sequences(self, seqs: list[str]) -> np.ndarray:
        """Embed a batch of sequences and return numpy matrix."""
        if self.backend == "transformers":
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

        batch = [(f"seq_{i}", s) for i, s in enumerate(seqs)]
        _, _, toks = self.batch_converter(batch)
        toks = toks.to(self.device)
        out = self.model(toks, repr_layers=[33], return_contacts=False)
        hidden = out["representations"][33]
        if self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            # fair-esm token 0 is BOS, and padded tokens are 1 in toks? use non-padding mask from != alphabet.padding_idx not available here
            # approximation: use all non-zero token ids (excludes padding)
            attn_mask = (toks != 1).unsqueeze(-1)
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
        """Embed records in shards with checkpoint/resume support at shard granularity."""
        ids = [r.id for r in records]
        seqs = [str(r.seq) for r in records]

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

            shard_embs: list[np.ndarray] = []
            shard_out_ids: list[str] = []
            for b in range(0, len(shard_ids), self.batch_size):
                b_ids = shard_ids[b : b + self.batch_size]
                b_seqs = shard_seqs[b : b + self.batch_size]
                emb = self.embed_sequences(b_seqs)
                shard_embs.append(emb)
                shard_out_ids.extend(b_ids)

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
