"""Per-env episode-return statistics tracker.

Replaces the legacy "per-step mean over worlds, averaged over time"
metric with a true **cross-env episode-return** statistic:

  * accumulate each metric term into a per-env running total
  * on episode completion (``done_mask[w] == 1``), the running total of
    world ``w`` IS that term's episode return for that env — push it
    into a population of completed-episode returns and zero the
    running total
  * at end of iter / eval window, summarise the population with
    cross-env mean / variance / std / count

All accumulators live on GPU (zero-copy torch views of the warp metric
buffers + scalar GPU tensors for sum / sumsq / n). One CPU sync per
iter (or per eval) at :meth:`stats` time keeps the rollout sync-minimal.
"""
from __future__ import annotations

import torch


class RewardTermReturnTracker:
    """Per-term episode-return statistics (cross-env mean/var of returns).

    **Batched / vectorised** internals: all K metric terms share a single
    ``(K, NWORLD)`` running buffer + ``(K,)`` sum/sumsq scalar buffers.
    Per-step update collapses ~5K launches (per-key loop in the legacy
    impl) to ~K + 6 launches — at K=27 that's ~5x fewer GPU launches per
    rollout step, which is the dominant inner-loop cost on small ops.

    Usage::

        tracker = RewardTermReturnTracker(h.metrics, h.NWORLD, h.torch_device)
        tracker.reset()                                         # at episode batch start
        for _ in range(rollout_length):
            obs, rew, done, info = h.step(action)
            tracker.update(h.metrics, info['done_mask'])        # sync-free
        out = tracker.stats()    # one CPU sync — {term: {mean, std, n}}
    """

    def __init__(self, metrics, n_world: int, device):
        self.keys    = list(metrics.keys())
        self.K       = len(self.keys)
        self.n_world = int(n_world)
        self.device  = device
        # Stacked (K, N) running per-env totals + per-step staging.
        self._running = torch.zeros(self.K, self.n_world, device=device)
        self._step    = torch.empty(self.K, self.n_world, device=device)
        # Pre-built row views — used for the K source-tensor copies.
        self._step_rows = [self._step[i] for i in range(self.K)]
        # Cross-episode scalar accumulators (one entry per term).
        self._sum   = torch.zeros(self.K, device=device)
        self._sumsq = torch.zeros(self.K, device=device)
        # Total completed-episode count (shared across all terms).
        self._n     = torch.zeros((), device=device, dtype=torch.long)

    def reset(self) -> None:
        self._running.zero_()
        self._sum.zero_()
        self._sumsq.zero_()
        self._n.zero_()

    def update(self, metrics, done_mask: torch.Tensor) -> None:
        """Sync-free per-step update — batched across all K terms.

        Stage all metric ``(NWORLD,)`` tensors into ``self._step`` (K rows),
        then run vectorised accumulation / sum / sumsq / zero-on-done in
        one sweep. ``copy_`` handles int→float conversion for ``bonus_*``
        metrics in the same launch.
        """
        # K copies (handles int → float conversion in-place into the row view).
        for i, src in enumerate(self.keys):
            self._step_rows[i].copy_(metrics[src])
        # Batched ops on the (K, N) buffer.
        self._running.add_(self._step)
        mask = done_mask.float()                                 # (N,)
        completed = self._running * mask                          # (K, N) broadcast
        self._sum   += completed.sum(dim=1)                       # (K,)
        self._sumsq += (completed * completed).sum(dim=1)         # (K,)
        self._running.mul_(1.0 - mask)
        self._n += done_mask.sum()

    def flush_all(self) -> None:
        """Flush EVERY world's running total as ONE completed episode.

        For eval-style use where there is NO per-world reset: accumulate the
        full window via ``update(metrics, done_mask=0)`` each step, then call
        this ONCE at the end. Each world then contributes exactly one
        episode-return (its full-window metric sum) → ``n == n_world``.

        Avoids the over-count produced when ``update`` is fed the real
        ``done_mask`` in a no-reset eval: a persistent failure (object dropped /
        drifted, never re-posed) re-flags ``done`` every subsequent step, so the
        same world flushes dozens of fragmented 1-step segments and ``n``
        balloons far past the world count.
        """
        self._sum   += self._running.sum(dim=1)
        self._sumsq += (self._running * self._running).sum(dim=1)
        self._n     += self.n_world
        self._running.zero_()

    def stats(self) -> dict:
        """One CPU sync (3 transfers: ``_n``, ``_sum``, ``_sumsq`` — all
        small). Returns ``{term: {mean, std, n}}`` over all episodes
        that have completed since the last :meth:`reset` (``var`` is computed
        internally only to derive ``std`` — it is not reported).

        ``n == 0`` (no episode completed yet) → all zeros.
        """
        n      = float(self._n.item())
        sums   = self._sum.cpu().tolist()
        sumsqs = self._sumsq.cpu().tolist()
        out = {}
        if n > 0:
            inv_n = 1.0 / n
            for i, k in enumerate(self.keys):
                mean = sums[i] * inv_n
                var  = max(sumsqs[i] * inv_n - mean * mean, 0.0)   # local only → std
                out[k] = dict(mean=mean, std=var ** 0.5, n=int(n))
        else:
            for k in self.keys:
                out[k] = dict(mean=0.0, std=0.0, n=0)
        return out
