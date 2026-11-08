"""``VecNormalize`` de Stable-Baselines3, recopié en torch pour l'enveloppe rsl_rl.

Le témoin A normalise observations **et** récompenses
(``normalize_obs``, ``normalize_reward``, ``clip_obs: 10``, ``gamma: 0.99``).
rsl_rl sait normaliser les observations, mais autrement : dans le réseau,
avec d'autres statistiques, et en renormalisant les observations stockées au
moment de la mise à jour. Il ne normalise pas les récompenses. Pour ne pas
changer ce qu'on apprend, on reproduit donc ``VecNormalize`` à l'identique
dans l'enveloppe de l'environnement :

* moyenne et variance courantes de Chan et al. (``RunningMeanStd``), en
  float64, initialisées à (0, 1, 1e-4) ;
* observation ``clip((x − μ) / sqrt(σ² + 1e-8), ±10)`` ;
* récompense divisée par l'écart type du **retour actualisé** courant,
  bornée à ±10, retour remis à zéro en fin d'épisode.

Les statistiques s'enregistrent avec chaque checkpoint et s'exportent en
``vec_normalize.pkl`` pour l'évaluation CPU (``gpu/export_sb3.py``).
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

OBS_GROUP = "policy"


class RunningMeanStd:
    """Moyenne et variance courantes, mises à jour par lots (comme SB3).

    Attributes:
        mean: Moyenne, forme ``shape``, float64.
        var: Variance (de population), forme ``shape``, float64.
        count: Nombre d'échantillons vus, initialisé à 1e-4.
    """

    def __init__(self, shape: tuple[int, ...], device: torch.device | str) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = 1e-4

    def update(self, batch: torch.Tensor) -> None:
        """Ajoute un lot ``(n, *shape)``, comme ``RunningMeanStd.update`` de SB3."""
        batch = batch.to(torch.float64)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = (self.var * self.count + batch_var * batch_count
              + delta ** 2 * self.count * batch_count / total)
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def state_dict(self) -> dict:
        """État sérialisable (tenseurs CPU)."""
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        """Recharge un état écrit par :meth:`state_dict`."""
        self.mean = state["mean"].to(self.mean.device, torch.float64)
        self.var = state["var"].to(self.var.device, torch.float64)
        self.count = float(state["count"])


class VecNormalizeWrapper(RslRlVecEnvWrapper):
    """Enveloppe rsl_rl qui normalise comme ``VecNormalize``.

    Args:
        env: L'environnement mjlab.
        norm_obs: Normalise les observations.
        norm_reward: Normalise les récompenses.
        clip_obs: Bornage des observations normalisées.
        gamma: Actualisation du retour utilisé pour la normalisation des
            récompenses.
        clip_reward: Bornage des récompenses normalisées.
        epsilon: Terme ajouté à la variance.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        gamma: float = 0.99,
        clip_reward: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True
        self.obs_rms: RunningMeanStd | None = None
        self.ret_rms: RunningMeanStd | None = None
        self.returns: torch.Tensor | None = None
        self._raw_obs: torch.Tensor | None = None
        # L'enveloppe de mjlab remet l'environnement à zéro sans passer par
        # self.reset() : on recommence, pour initialiser les statistiques.
        super().__init__(env)
        self.reset()

    def _lazy_init(self, raw: torch.Tensor) -> None:
        if self.obs_rms is None:
            self.obs_rms = RunningMeanStd(tuple(raw.shape[1:]), raw.device)
            self.ret_rms = RunningMeanStd((), raw.device)
            self.returns = torch.zeros(raw.shape[0], dtype=torch.float64, device=raw.device)

    def normalize_obs(self, raw: torch.Tensor) -> torch.Tensor:
        """Observation normalisée et bornée, en float32."""
        if not self.norm_obs:
            return raw
        assert self.obs_rms is not None
        std = torch.sqrt(self.obs_rms.var + self.epsilon)
        out = torch.clamp((raw.to(torch.float64) - self.obs_rms.mean) / std,
                          -self.clip_obs, self.clip_obs)
        return out.to(torch.float32)

    def _wrap(self, obs_dict: dict, update: bool) -> TensorDict:
        raw = obs_dict[OBS_GROUP]
        self._lazy_init(raw)
        if update and self.training and self.norm_obs:
            self.obs_rms.update(raw)
        self._raw_obs = raw
        return TensorDict({OBS_GROUP: self.normalize_obs(raw)}, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        obs = self._wrap(obs_dict, update=True)
        assert self.returns is not None
        self.returns.zero_()
        return obs, extras

    def get_observations(self) -> TensorDict:
        assert self._raw_obs is not None
        return TensorDict({OBS_GROUP: self.normalize_obs(self._raw_obs)},
                          batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = self._wrap(obs_dict, update=True)
        dones = (terminated | truncated).to(torch.long)
        extras["time_outs"] = truncated
        extras["raw_reward"] = rew
        if self.norm_reward:
            assert self.returns is not None and self.ret_rms is not None
            if self.training:
                self.returns = self.returns * self.gamma + rew.to(torch.float64)
                self.ret_rms.update(self.returns)
            rew = torch.clamp(
                rew.to(torch.float64) / torch.sqrt(self.ret_rms.var + self.epsilon),
                -self.clip_reward, self.clip_reward,
            ).to(torch.float32)
            self.returns[dones.bool()] = 0.0
        return obs, rew, dones, extras

    def state_dict(self) -> dict:
        """Statistiques de normalisation, à ranger avec le checkpoint."""
        assert self.obs_rms is not None and self.ret_rms is not None
        return {
            "obs_rms": self.obs_rms.state_dict(),
            "ret_rms": self.ret_rms.state_dict(),
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "norm_obs": self.norm_obs,
            "norm_reward": self.norm_reward,
        }

    def load_state_dict(self, state: dict) -> None:
        """Recharge les statistiques d'un checkpoint."""
        assert self.obs_rms is not None and self.ret_rms is not None
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.ret_rms.load_state_dict(state["ret_rms"])
