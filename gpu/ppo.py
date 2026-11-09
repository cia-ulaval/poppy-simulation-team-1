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

Deux options de SB3 et du dépôt, utilisées par H-Placo, sont reproduites
telles quelles :

* ``target_kl`` : avant chaque pas, SB3 estime la divergence
  ``mean((r − 1) − log r)`` sur le mini-lot ; au-delà de ``1,5 × target_kl``,
  il **ne fait pas** ce pas et arrête la mise à jour (toutes époques).
* ``symmetry_loss_weight`` (``src/training/mirror_ppo.py``) : **après** la
  mise à jour PPO, une époque de plus sur le même lot, rebattu, avec la
  seule perte ``poids · mean((μ(s) − M_a(μ(M_s(s))))²)``. Dans SB3, seuls
  les poids de la moyenne de l'acteur reçoivent un gradient : ``log_std``
  et le critique n'en ont pas, Adam les saute, ni leurs moments ni leur
  compteur n'avancent, et le bornage de norme ne porte que sur l'acteur.
  D'où deux Adam ici : un pour le réseau de la moyenne, un pour le reste ;
  ensemble, ils font exactement l'Adam de SB3 (Adam est par paramètre).
  Ce n'est **pas** la perte miroir de rsl_rl, qui s'ajoute dans la perte PPO
  à chaque pas : elle n'est pas utilisée.

Débit. A et H-Placo font 20 époques de 16 mini-lots de 256 : 780 000 pas
de gradient pour 10 M de pas d'environnement. Sur de si petits lots, un pas
coûte ~11 ms sur la RTX 3050, presque tout en lancements de noyaux. Capturé
dans un graphe CUDA et rejoué, il tombe à ~0,6 ms. Le calcul est le même
(Adam « fused » au lieu de « foreach », mêmes formules) ; seule la façon de
le lancer change. Avec ``target_kl``, le pas est coupé en deux graphes (gradient,
puis Adam) et la divergence est lue entre les deux : une synchronisation
par mini-lot, le prix de l'arrêt anticipé.
"""

from __future__ import annotations

import math
from itertools import chain

import torch
from rsl_rl.algorithms import PPO

_SB3_ADAM_EPS = 1e-5
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class _AdamPair:
    """Deux Adam vus comme un seul par rsl_rl (sauvegarde, reprise, taux)."""

    def __init__(self, first: torch.optim.Adam, second: torch.optim.Adam) -> None:
        self.parts = (first, second)

    @property
    def param_groups(self) -> list[dict]:
        return [g for opt in self.parts for g in opt.param_groups]

    @property
    def state(self) -> dict:
        return {k: v for opt in self.parts for k, v in opt.state.items()}

    def zero_grad(self, set_to_none: bool = False) -> None:
        for opt in self.parts:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.parts:
            opt.step()

    def state_dict(self) -> dict:
        return {"mean_net": self.parts[0].state_dict(), "rest": self.parts[1].state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.parts[0].load_state_dict(state["mean_net"])
        self.parts[1].load_state_dict(state["rest"])


class SB3PPO(PPO):
    """PPO de rsl_rl, avec l'optimiseur, le bornage, le brassage, ``target_kl``
    et la perte miroir de SB3 / ``MirrorPPO``.

    Sur GPU, chaque morceau de la mise à jour d'un mini-lot (gradient PPO,
    pas d'Adam, pas de symétrie) est capturé une fois dans un graphe CUDA
    puis rejoué ; ailleurs, les mêmes fonctions s'exécutent normalement.
    """

    def __init__(  # noqa: ANN002, ANN003 - arguments de rsl_rl
        self, *args, use_cuda_graph: bool = True, target_kl: float | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if kwargs.get("optimizer", "adam") != "adam":
            raise ValueError("SB3PPO reproduit Adam seulement.")
        if (self.symmetry is not None or self.rnd is not None or self.schedule != "fixed"
                or not self.normalize_advantage_per_mini_batch or self.use_clipped_value_loss
                or self.actor.obs_normalization or self.critic.obs_normalization):
            raise NotImplementedError(
                "SB3PPO suppose les réglages de gpu.train.runner_cfg (taux fixe, avantages "
                "normalisés par mini-lot, valeur non bornée, pas de symétrie rsl_rl ni de RND).")
        self.target_kl = None if target_kl is None else float(target_kl)
        self._sb3_max_grad_norm = self.max_grad_norm
        self._params = list(chain(self.actor.parameters(), self.critic.parameters()))
        self._mean_params = list(self.actor.mlp.parameters())
        mean_ids = {id(p) for p in self._mean_params}
        self._rest_params = [p for p in self._params if id(p) not in mean_ids]
        self._use_graph = use_cuda_graph and str(self.device).startswith("cuda")
        options = {"lr": self.learning_rate, "eps": _SB3_ADAM_EPS}
        if self._use_graph:
            options.update(fused=True, capturable=True)
        self.optimizer = _AdamPair(
            torch.optim.Adam(self._mean_params, **options),
            torch.optim.Adam(self._rest_params, **options),
        )
        self.max_grad_norm = float("inf")  # bornage global fait ici, pas par rsl_rl
        self._mirror: dict | None = None
        self._sym_weight = 0.0
        self._graphs: dict[str, torch.cuda.CUDAGraph] | None = None
        self._buffers: dict[str, torch.Tensor] | None = None

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str) -> SB3PPO:  # noqa: ANN001
        """``PPO.construct_algorithm``, puis le miroir de ``MirrorPPO`` s'il est demandé."""
        weight = float(cfg["algorithm"].pop("sb3_symmetry_weight", 0.0))
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        if weight > 0.0:
            alg.set_mirror(env, weight)
        return alg

    def set_mirror(self, env, weight: float) -> None:  # noqa: ANN001 - VecNormalizeWrapper
        """Active la perte de symétrie de ``MirrorPPO``, de poids ``weight``.

        Les tables sont celles de ``src.environments.mirror.build_mirror_spec``
        (voir ``gpu/symmetry.py``) ; les statistiques de normalisation sont
        relues sur ``env`` à chaque mise à jour, comme ``MirrorPPO`` relit
        ``VecNormalize``.
        """
        from gpu.symmetry import mirror_tables

        if self._graphs is not None:
            raise RuntimeError("set_mirror doit précéder la première mise à jour.")
        self._mirror = {"env": env, "tables": mirror_tables(env)}
        self._sym_weight = float(weight)

    # ── Les trois morceaux d'un mini-lot ─────────────────────────────────

    def _log_std(self) -> torch.Tensor:
        dist = self.actor.distribution
        lo, hi = dist.log_std_range
        return dist.log_std_param.clamp(lo, hi)

    def _ppo_grad(self) -> None:
        """Perte PPO de SB3 sur le mini-lot ``self._idx``, gradient et divergence."""
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
        log_ratio = log_prob - b["old_log_prob"][idx]
        ratio = torch.exp(log_ratio)
        surrogate = torch.max(
            -adv * ratio, -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        ).mean()
        value = self.critic.mlp(obs).squeeze(-1)
        value_loss = (b["returns"][idx] - value).pow(2).mean()
        entropy = (0.5 + _LOG_SQRT_2PI + log_std).sum()
        loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        with torch.no_grad():
            b["kl"].copy_(((ratio - 1.0) - log_ratio).mean())
            b["sum_value"] += value_loss
            b["sum_surrogate"] += surrogate
            b["sum_entropy"] += entropy

    def _ppo_step(self) -> None:
        """Bornage global de la norme du gradient, puis Adam sur tous les paramètres."""
        torch.nn.utils.clip_grad_norm_(self._params, self._sb3_max_grad_norm, foreach=True)
        self.optimizer.step()

    def _minibatch_step(self) -> None:
        """Un pas de PPO complet sur ``self._idx``, sans arrêt anticipé."""
        self._ppo_grad()
        self._ppo_step()

    def _mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """``MirrorPPO._mirror_obs`` : dénormaliser, refléter, renormaliser, borner."""
        m = self._mirror["tables"]
        b = self._buffers
        if not self._mirror["env"].norm_obs:
            return obs[:, m.obs_perm] * m.obs_sign
        raw = obs * b["obs_std"] + b["obs_mean"]
        mirrored = raw[:, m.obs_perm] * m.obs_sign
        clip = self._mirror["env"].clip_obs
        return torch.clamp((mirrored - b["obs_mean"]) / b["obs_std"], -clip, clip)

    def _mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        """``MirrorPPO._mirror_action`` : par les angles cibles, courses asymétriques."""
        m = self._mirror["tables"]
        target = m.joint_init + torch.where(action >= 0, action * m.range_up, action * m.range_down)
        delta = target[:, m.joint_perm] * m.joint_sign - m.joint_init
        return torch.where(delta >= 0, delta / m.range_up, delta / m.range_down)

    def _sym_step(self) -> None:
        """Un pas de la perte de symétrie de ``MirrorPPO.train`` sur ``self._idx``."""
        b = self._buffers
        obs = b["obs"][self._idx]
        mean = self.actor.mlp(obs)
        mirrored_mean = self.actor.mlp(self._mirror_obs(obs))
        loss = self._sym_weight * ((mean - self._mirror_action(mirrored_mean)) ** 2).mean()
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._mean_params, self._sb3_max_grad_norm, foreach=True)
        self.optimizer.parts[0].step()
        with torch.no_grad():
            b["sum_symmetry"] += loss

    # ── Capture et mise à jour ───────────────────────────────────────────

    def _allocate(self, batch: int, mini: int) -> None:
        dev = self.device
        dim = self.storage.observations.flatten(0, 1)[next(iter(self.actor.obs_groups))].shape[-1]
        nact = self.storage.actions.shape[-1]
        self._buffers = {
            "obs": torch.zeros(batch, dim, device=dev),
            "actions": torch.zeros(batch, nact, device=dev),
            "advantages": torch.zeros(batch, device=dev),
            "returns": torch.zeros(batch, device=dev),
            "old_log_prob": torch.zeros(batch, device=dev),
            "obs_mean": torch.zeros(dim, device=dev),
            "obs_std": torch.ones(dim, device=dev),
            "kl": torch.zeros((), device=dev),
            "sum_value": torch.zeros((), device=dev),
            "sum_surrogate": torch.zeros((), device=dev),
            "sum_entropy": torch.zeros((), device=dev),
            "sum_symmetry": torch.zeros((), device=dev),
        }
        self._idx = torch.arange(mini, device=dev)

    def _capture(self) -> None:
        """Capture les morceaux du pas, sans modifier les poids ni l'état d'Adam."""
        bodies = {"grad": self._ppo_grad, "step": self._ppo_step}
        if self._mirror is not None:
            bodies["sym"] = self._sym_step
        saved = [p.detach().clone() for p in self._params]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # échauffement : alloue gradients et état d'Adam
                for body in bodies.values():
                    body()
        torch.cuda.current_stream().wait_stream(stream)
        self._graphs = {}
        for name, body in bodies.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            self._graphs[name] = graph
        # L'échauffement a fait de vrais pas : on revient à l'état initial.
        with torch.no_grad():
            for p, value in zip(self._params, saved, strict=True):
                p.copy_(value)
            for state in self.optimizer.state.values():
                for tensor in state.values():
                    tensor.zero_()

    def _run(self, name: str) -> None:
        if self._graphs is not None:
            self._graphs[name].replay()
        else:
            {"grad": self._ppo_grad, "step": self._ppo_step, "sym": self._sym_step}[name]()

    def update(self) -> dict[str, float]:
        """Époques × mini-lots rebattus à chaque époque, arrêt anticipé, puis symétrie."""
        st = self.storage
        group = next(iter(self.actor.obs_groups))
        batch = st.num_envs * st.num_transitions_per_env
        mini = batch // self.num_mini_batches
        if self._buffers is None:
            self._allocate(batch, mini)
            if self._use_graph:
                self._capture()
        b = self._buffers
        b["obs"].copy_(st.observations.flatten(0, 1)[group])
        b["actions"].copy_(st.actions.flatten(0, 1))
        b["advantages"].copy_(st.advantages.flatten(0, 1).squeeze(-1))
        b["returns"].copy_(st.returns.flatten(0, 1).squeeze(-1))
        b["old_log_prob"].copy_(st.actions_log_prob.flatten(0, 1).squeeze(-1))
        if self._mirror is not None and self._mirror["env"].norm_obs:
            env = self._mirror["env"]
            b["obs_mean"].copy_(env.obs_rms.mean.to(torch.float32))
            b["obs_std"].copy_(torch.sqrt(env.obs_rms.var + env.epsilon).to(torch.float32))
        for key in ("sum_value", "sum_surrogate", "sum_entropy", "sum_symmetry"):
            b[key].zero_()

        n_grad = 0
        stopped_at = None
        limit = None if self.target_kl is None else 1.5 * self.target_kl
        for epoch in range(self.num_learning_epochs):
            order = torch.randperm(batch, device=self.device)
            for i in range(self.num_mini_batches):
                self._idx.copy_(order[i * mini:(i + 1) * mini])
                self._run("grad")
                n_grad += 1
                if limit is not None and b["kl"].item() > limit:
                    stopped_at = epoch
                    break
                self._run("step")
            if stopped_at is not None:
                break

        if self._mirror is not None:
            order = torch.randperm(batch, device=self.device)
            for i in range(self.num_mini_batches):
                self._idx.copy_(order[i * mini:(i + 1) * mini])
                self._run("sym")

        st.clear()
        out = {
            "value": b["sum_value"].item() / n_grad,
            "surrogate": b["sum_surrogate"].item() / n_grad,
            "entropy": b["sum_entropy"].item() / n_grad,
            "approx_kl_last": b["kl"].item(),
            "epochs_done": float(self.num_learning_epochs if stopped_at is None else stopped_at),
        }
        if self._mirror is not None:
            out["symmetry"] = b["sum_symmetry"].item() / self.num_mini_batches
        return out
