"""PPO avec perte de symétrie miroir (proposition 2 de la spec, contre la boiterie).

Après chaque mise à jour PPO, quelques pas de gradient supplémentaires sur le
même lot d'expériences rapprochent l'action moyenne de la politique pour un
état ``s`` du reflet de son action pour l'état reflété ``M_s(s)`` :

    L_sym = poids · ‖ μ(s) − M_a(μ(M_s(s))) ‖²

C'est la « mirror symmetry loss » de Yu, Turk et Liu (2018). Elle n'ajoute
aucune dimension à l'observation : la politique reste chargeable par tout le
reste du dépôt comme un PPO ordinaire.

Les observations vues par la politique sont **normalisées** par
``VecNormalize`` ; le reflet porte sur l'observation brute. On dénormalise
donc, on reflète, puis on renormalise avec les mêmes statistiques.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO

from src.environments.mirror import MirrorSpec


class MirrorPPO(PPO):
    """PPO dont la politique est poussée vers la symétrie gauche ↔ droite.

    ``set_mirror`` doit être appelé avant ``learn`` ; sans lui, c'est un PPO
    ordinaire. Les tables ne sont pas enregistrées dans le ``.zip`` : elles se
    recalculent sur l'environnement.
    """

    _mirror: dict | None = None
    symmetry_weight: float = 0.0

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_mirror"]

    def set_mirror(self, spec: MirrorSpec, weight: float) -> None:
        """Active la perte de symétrie.

        Args:
            spec: Tables miroir de l'environnement.
            weight: Poids de la perte. 0 la désactive.
        """
        def tensor(values: np.ndarray) -> th.Tensor:
            return th.as_tensor(values, dtype=th.float32, device=self.device)

        self.symmetry_weight = float(weight)
        self._mirror = {
            "obs_perm": th.as_tensor(spec.obs_perm, device=self.device),
            "obs_sign": tensor(spec.obs_sign),
            "joint_perm": th.as_tensor(spec.joint_perm, device=self.device),
            "joint_sign": tensor(spec.joint_sign),
            "init": tensor(spec.joint_init),
            "up": tensor(spec.range_up),
            "down": tensor(spec.range_down),
        }

    def _mirror_obs(self, obs: th.Tensor) -> th.Tensor:
        m = self._mirror
        norm = self.get_vec_normalize_env()
        if norm is None or not norm.norm_obs:
            return obs[:, m["obs_perm"]] * m["obs_sign"]
        mean = th.as_tensor(norm.obs_rms.mean, dtype=th.float32, device=self.device)
        std = th.sqrt(th.as_tensor(norm.obs_rms.var, dtype=th.float32,
                                   device=self.device) + norm.epsilon)
        raw = obs * std + mean
        mirrored = raw[:, m["obs_perm"]] * m["obs_sign"]
        return th.clamp((mirrored - mean) / std, -norm.clip_obs, norm.clip_obs)

    def _mirror_action(self, action: th.Tensor) -> th.Tensor:
        m = self._mirror
        target = m["init"] + th.where(action >= 0, action * m["up"], action * m["down"])
        mirrored = target[:, m["joint_perm"]] * m["joint_sign"]
        delta = mirrored - m["init"]
        return th.where(delta >= 0, delta / m["up"], delta / m["down"])

    def train(self) -> None:
        super().train()
        if self._mirror is None or self.symmetry_weight <= 0.0:
            return

        self.policy.set_training_mode(True)
        losses = []
        for rollout in self.rollout_buffer.get(self.batch_size):
            obs = rollout.observations
            mean = self.policy.get_distribution(obs).distribution.mean
            mirrored_mean = self.policy.get_distribution(
                self._mirror_obs(obs)
            ).distribution.mean
            loss = self.symmetry_weight * th.mean(
                (mean - self._mirror_action(mirrored_mean)) ** 2
            )
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
        self.logger.record("train/symmetry_loss", float(np.mean(losses)))
