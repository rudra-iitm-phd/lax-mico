import jax.numpy as jnp
from flax import nnx, struct
from utils.models import EnsembleCritic, SACGaussianActor, Scalar
from utils.metric_models import EnsembleStateMetric, EnsembleStateActionMetric, MinStateActiontoStateMetric


class Models(nnx.Module):
    def __init__(
        self,
        critic:EnsembleCritic,
        target_critic:EnsembleCritic,
        actor:SACGaussianActor,
        state_metric:EnsembleStateMetric,
        target_state_metric:EnsembleStateMetric,
        state_action_metric:EnsembleStateActionMetric,
        target_state_action_metric:EnsembleStateActionMetric,
        min_state_action_to_state_metric:MinStateActiontoStateMetric,
        target_state_action_to_state_metric:MinStateActiontoStateMetric,
        log_alpha:Scalar,
    ):
        self.critic = critic
        self.target_critic = target_critic
        self.actor = actor
        self.log_alpha = log_alpha
        self.state_metric = state_metric
        self.target_state_metric = target_state_metric
        self.state_action_metric = state_action_metric
        self.target_state_action_metric = target_state_action_metric
        self.min_state_action_to_state_metric = min_state_action_to_state_metric
        self.target_state_action_to_state_metric = target_state_action_to_state_metric


class Optimizers(nnx.Module):
    def __init__(
        self,
        critic,
        actor,
        log_alpha,
        state_metric,
        state_action_metric,
        min_state_action_to_state_metric,
    ):
        self.critic = critic
        self.actor = actor
        self.log_alpha = log_alpha
        self.state_metric = state_metric
        self.state_action_metric = state_action_metric
        self.min_state_action_to_state_metric = min_state_action_to_state_metric


class TrainingState(nnx.Module):
    def __init__(self, models: Models, optimizers: Optimizers):
        self.models = models
        self.optimizers = optimizers


@struct.dataclass
class CriticAux:
    loss: jnp.ndarray
    q1_mean: jnp.ndarray
    q2_mean: jnp.ndarray


@struct.dataclass
class ActorAux:
    loss: jnp.ndarray


@struct.dataclass
class AgentAux:
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    q1_mean: float = 0.0
    q2_mean: float = 0.0
    alpha_loss: float = 0.0
    alpha: float = 0.0
    log_pi_mean: float = 0.0


@struct.dataclass
class MetricAux:
    state_metric_loss: float = 0.0
    state_action_metric_loss: float = 0.0
    state_action_to_state_metric_loss: float = 0.0
    self_state_distance: float = 0.0
    cross_state_distance: float = 0.0
    self_state_action_distance: float = 0.0
    cross_state_action_distance: float = 0.0
    self_state_action_to_state_distance : float = 0.0
    cross_state_action_to_state_distance : float = 0.0
    self_state_asymmetry_avg: float = 0.0
    cross_state_asymmetry_avg: float = 0.0
    self_state_action_asymmetry_avg: float = 0.0
    cross_state_action_asymmetry_avg: float = 0.0
    act_rep_loss: float = 0.0

    h_lambda_diff_self:float = 0.0
    h_lambda_diff_cross:float=0.0
