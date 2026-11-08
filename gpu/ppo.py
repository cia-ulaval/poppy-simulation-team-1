"""PPO de rsl_rl, aligné sur la mise à jour PPO de Stable-Baselines3, en graphe CUDA.

La perte est déjà celle de SB3 quand on règle rsl_rl comme SB3 (avantages
normalisés par mini-lot, perte de valeur non bornée de poids 0,5, entropie
nulle, taux fixe). Trois écarts restent, que la configuration ne couvre pas,
et que :class:`SB3PPO` corrige :

* SB3 borne la norme du gradient **de tous les paramètres ensemble**
  (acteur, critique et ``log_std``) ; rsl_rl borne l'acteur et le critique
  séparément. Quand la perte de valeur domine, SB3 réduit aussi le pas de
  la politique ; rsl_rl non.
* SB3 crée Adam avec ``eps = 1e-5`` ; rsl_rl garde la valeur de torch, 1e-8.
* SB3 rebat les transitions à **chaque** époque ; rsl_rl tire une seule
  permutation et la reprend à toutes les époques.

Débit. Le témoin A fait 20 époques de 16 mini-lots de 256 : 780 000 pas de
gradient pour 10 M de pas d'environnement. Sur de si petits lots, un pas
coûte ~11 ms sur la RTX 3050, presque tout en lancements de noyaux. Capturé
dans un graphe CUDA et rejoué, il tombe à ~0,6 ms. Le calcul est le même
(Adam « fused » au lieu de « foreach », mêmes formules) ; seule la façon de
le lancer change.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import chain

import torch
from rsl_rl.algorithms import PPO

_SB3_ADAM_EPS = 1e-5
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class _GlobalClipAdam(torch.optim.Adam):
    """Adam qui borne la norme globale du gradient avant chaque pas (chemin sans graphe)."""

    def __init__(
        self, params: Iterable[torch.nn.Parameter], lr: float, max_grad_norm: float
    ) -> None:
        params = list(params)
        super().__init__(params, lr=lr, eps=_SB3_ADAM_EPS)
        self._clipped = params
        self._max_grad_norm = max_grad_norm

    def step(self, closure=None):  # noqa: ANN001, ANN201 - signature de torch
        torch.nn.utils.clip_grad_norm_(self._clipped, self._max_grad_norm)
        return super().step(closure)


class SB3PPO(PPO):
    """PPO de rsl_rl, avec l'optimiseur, le bornage et le brassage de SB3.

    Sans symétrie et sur GPU, la mise à jour d'un mini-lot est capturée
    une fois dans un graphe CUDA puis rejouée ; sinon, c'est le
    ``PPO.update`` de rsl_rl, avec l'optimiseur de SB3.
    """

    def __init__(self, *args, use_cuda_graph: bool = True, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        if kwargs.get("optimizer", "adam") != "adam":
            raise ValueError("SB3PPO reproduit Adam seulement.")
        self._sb3_max_grad_norm = self.max_grad_norm
        self._params = list(chain(self.actor.parameters(), self.critic.parameters()))
        self._graph_ok = (
            use_cuda_graph and self.symmetry is None and self.rnd is None
            and str(self.device).startswith("cuda") and self.schedule == "fixed"
            and self.normalize_advantage_per_mini_batch
            and not self.use_clipped_value_loss
            and not self.actor.obs_normalization and not self.critic.obs_normalization
        )
        if self._graph_ok:
            self.optimizer = torch.optim.Adam(
                self._params, lr=self.learning_rate, eps=_SB3_ADAM_EPS,
                fused=True, capturable=True,
            )
        else:
            self.optimizer = _GlobalClipAdam(
                self._params, lr=self.learning_rate, max_grad_norm=self.max_grad_norm
            )
            # Les deux clip_grad_norm_ séparés de PPO.update deviennent sans effet.
            self.max_grad_norm = float("inf")
        self._graph: torch.cuda.CUDAGraph | None = None

    # ── Mise à jour en graphe CUDA ───────────────────────────────────────

    def _log_std(self) -> torch.Tensor:
        dist = self.actor.distribution
        lo, hi = dist.log_std_range
        return dist.log_std_param.clamp(lo, hi)

    def _minibatch_step(self) -> None:
        """Un pas de PPO sur le mini-lot ``self._idx`` (corps du graphe)."""
        b = self._buffers
        idx = self._idx
        obs = b["obs"][idx]
        act = b["actions"][idx]
        adv = b["advantages"][idx]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mean = self.actor.mlp(obs)
        log_std = self._log_std()
        log_prob = (-0.5 * ((act - mean) * torch.exp(-log_std)) ** 2
                    - log_std - _LOG_SQRT_2PI).sum(-1)
        ratio = torch.exp(log_prob - b["old_log_prob"][idx])
        surrogate = torch.max(
            -adv * ratio, -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        ).mean()
        value = self.critic.mlp(obs).squeeze(-1)
        value_loss = (b["returns"][idx] - value).pow(2).mean()
        entropy = (0.5 + _LOG_SQRT_2PI + log_std).sum()
        loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, self._sb3_max_grad_norm, foreach=True)
        self.optimizer.step()
        b["sum_value"] += value_loss.detach()
        b["sum_surrogate"] += surrogate.detach()
        b["sum_entropy"] += entropy.detach()

    def _capture(self, batch: int, mini: int) -> None:
        """Capture le pas de mini-lot, sans modifier les poids ni l'état d'Adam."""
        dev = self.device
        dim = self.storage.observations.flatten(0, 1)[next(iter(self.actor.obs_groups))].shape[-1]
        nact = self.storage.actions.shape[-1]
        self._buffers = {
            "obs": torch.zeros(batch, dim, device=dev),
            "actions": torch.zeros(batch, nact, device=dev),
            "advantages": torch.zeros(batch, device=dev),
            "returns": torch.zeros(batch, device=dev),
            "old_log_prob": torch.zeros(batch, device=dev),
            "sum_value": torch.zeros((), device=dev),
            "sum_surrogate": torch.zeros((), device=dev),
            "sum_entropy": torch.zeros((), device=dev),
        }
        self._idx = torch.arange(mini, device=dev)
        saved = [p.detach().clone() for p in self._params]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # échauffement : alloue gradients et état d'Adam
                self._minibatch_step()
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._minibatch_step()
        # L'échauffement a fait trois vrais pas : on revient à l'état initial.
        with torch.no_grad():
            for p, value in zip(self._params, saved, strict=True):
                p.copy_(value)
            for state in self.optimizer.state.values():
                for tensor in state.values():
                    tensor.zero_()

    def update(self) -> dict[str, float]:
        """Époques × mini-lots, rebattus à chaque époque comme SB3."""
        if not self._graph_ok:
            return super().update()
        st = self.storage
        group = next(iter(self.actor.obs_groups))
        batch = st.num_envs * st.num_transitions_per_env
        mini = batch // self.num_mini_batches
        if self._graph is None:
            self._capture(batch, mini)
        b = self._buffers
        b["obs"].copy_(st.observations.flatten(0, 1)[group])
        b["actions"].copy_(st.actions.flatten(0, 1))
        b["advantages"].copy_(st.advantages.flatten(0, 1).squeeze(-1))
        b["returns"].copy_(st.returns.flatten(0, 1).squeeze(-1))
        b["old_log_prob"].copy_(st.actions_log_prob.flatten(0, 1).squeeze(-1))
        for key in ("sum_value", "sum_surrogate", "sum_entropy"):
            b[key].zero_()
        for _ in range(self.num_learning_epochs):
            order = torch.randperm(batch, device=self.device)
            for i in range(self.num_mini_batches):
                self._idx.copy_(order[i * mini:(i + 1) * mini])
                self._graph.replay()
        n = self.num_learning_epochs * self.num_mini_batches
        st.clear()
        return {
            "value": b["sum_value"].item() / n,
            "surrogate": b["sum_surrogate"].item() / n,
            "entropy": b["sum_entropy"].item() / n,
        }
