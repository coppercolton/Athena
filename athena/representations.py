"""Learned visual representations with protected reusable reasoning heads.

Athena v0.8 removes the hand-authored four-number interface from its newest
learning path.  The system receives raw 8x8 sensor grids, learns a compressed
latent representation by reconstructing observations, and then learns modular
binary operators on top of that representation.

Representation updates are treated as dangerous because every downstream
operator depends on them.  Candidate encoder/decoder weights train in
isolation, each retained operator is re-consolidated from protected replay, and
the complete candidate is promoted only when reconstruction and every operator
pass independent regression gates.  Failed candidates are discarded.

This is a compact, falsifiable research substrate rather than a claim of
general vision or AGI.  The observations are synthetic sensor grids and the
operator signal is supervised, but the latent features themselves are learned
from pixels rather than supplied as object coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


VisualRule = Literal["horizontal_order", "vertical_order", "far_apart"]


@dataclass(frozen=True)
class RawObservation:
    """One bounded raw sensor observation with no hand-authored features."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("raw observation cannot be empty")
        if any(not math.isfinite(float(value)) for value in self.values):
            raise ValueError("raw observation values must be finite")
        if any(float(value) < 0.0 or float(value) > 1.0 for value in self.values):
            raise ValueError("raw observation values must be in [0, 1]")


@dataclass(frozen=True)
class GroundedExample:
    """An outcome label attached to a raw observation."""

    observation: RawObservation
    target: int

    def __post_init__(self) -> None:
        if self.target not in (0, 1):
            raise ValueError("grounded target must be 0 or 1")


@dataclass(frozen=True)
class RepresentationConfig:
    """Bounds representation learning, transfer, replay, and promotion."""

    sensor_dim: int = 128
    latent_dim: int = 16
    head_hidden_dim: int = 24
    representation_learning_rate: float = 0.018
    operator_learning_rate: float = 0.025
    representation_epochs: int = 700
    operator_epochs: int = 650
    l2: float = 1e-4
    observation_replay_capacity: int = 768
    operator_replay_capacity: int = 256
    operator_probe_capacity: int = 256
    minimum_representation_validation: int = 64
    minimum_operator_validation: int = 32
    maximum_reconstruction_loss: float = 0.035
    reconstruction_regression_tolerance: float = 0.002
    operator_validation_threshold: float = 0.90
    operator_regression_tolerance: float = 0.02
    minimum_latent_variance: float = 1e-4
    seed: int = 37

    def __post_init__(self) -> None:
        if self.sensor_dim < 4:
            raise ValueError("sensor_dim must be >= 4")
        if not 1 < self.latent_dim < self.sensor_dim:
            raise ValueError("latent_dim must be between 1 and sensor_dim")
        if self.head_hidden_dim < 2:
            raise ValueError("head_hidden_dim must be >= 2")
        if self.representation_learning_rate <= 0.0:
            raise ValueError("representation_learning_rate must be > 0")
        if self.operator_learning_rate <= 0.0:
            raise ValueError("operator_learning_rate must be > 0")
        if self.representation_epochs < 1 or self.operator_epochs < 1:
            raise ValueError("training epochs must be >= 1")
        if self.l2 < 0.0:
            raise ValueError("l2 must be >= 0")
        if self.observation_replay_capacity < self.minimum_representation_validation:
            raise ValueError("observation replay cannot cover validation minimum")
        if self.operator_replay_capacity < self.minimum_operator_validation:
            raise ValueError("operator replay cannot cover validation minimum")
        if self.operator_probe_capacity < self.minimum_operator_validation:
            raise ValueError("operator probes cannot cover validation minimum")
        if self.minimum_representation_validation < 1:
            raise ValueError("minimum representation validation must be >= 1")
        if self.minimum_operator_validation < 1:
            raise ValueError("minimum operator validation must be >= 1")
        if self.maximum_reconstruction_loss <= 0.0:
            raise ValueError("maximum reconstruction loss must be > 0")
        if self.reconstruction_regression_tolerance < 0.0:
            raise ValueError("reconstruction regression tolerance must be >= 0")
        if not 0.5 < self.operator_validation_threshold <= 1.0:
            raise ValueError("operator validation threshold must be in (0.5, 1]")
        if not 0.0 <= self.operator_regression_tolerance < 0.5:
            raise ValueError("operator regression tolerance must be in [0, 0.5)")
        if self.minimum_latent_variance <= 0.0:
            raise ValueError("minimum latent variance must be > 0")


@dataclass(frozen=True)
class RepresentationState:
    """Inspectable state of the retained sensor representation."""

    version: int
    sensor_dim: int
    latent_dim: int
    observations_seen: int
    replay_observations: int
    validation_loss: float
    latent_variance: float
    operator_count: int
    checksum: str


@dataclass(frozen=True)
class RepresentationReport:
    """Promotion evidence for one candidate representation update."""

    promoted: bool
    recruited: bool
    rolled_back: bool
    version: int
    before_loss: float
    candidate_loss: float
    protected_loss_before: float
    protected_loss_after: float
    latent_variance: float
    weight_delta: float
    training_observations: int
    validation_observations: int
    replay_observations: int
    protected_operators: int
    minimum_operator_accuracy: float
    checksum_before: str
    checksum_after: str
    reason: str


@dataclass(frozen=True)
class ReasoningOperator:
    """One reusable decision head grounded in the shared latent state."""

    name: str
    version: int
    examples_seen: int
    replay_examples: int
    probe_examples: int
    validation_accuracy: float
    representation_version: int
    checksum: str


@dataclass(frozen=True)
class OperatorReport:
    """Promotion and transfer evidence for one reasoning operator."""

    name: str
    promoted: bool
    recruited: bool
    rolled_back: bool
    version: int
    before_accuracy: float
    candidate_accuracy: float
    regression_accuracy: float
    untrained_representation_accuracy: float
    head_weight_delta: float
    representation_checksum: str
    training_examples: int
    validation_examples: int
    replay_examples: int
    reason: str


class _Adam:
    def __init__(self, parameters: Sequence[np.ndarray], learning_rate: float) -> None:
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.first = [np.zeros_like(item) for item in self.parameters]
        self.second = [np.zeros_like(item) for item in self.parameters]
        self.step = 0

    def update(self, gradients: Sequence[np.ndarray]) -> None:
        self.step += 1
        for index, (parameter, gradient) in enumerate(
            zip(self.parameters, gradients)
        ):
            np.clip(gradient, -5.0, 5.0, out=gradient)
            self.first[index] = 0.9 * self.first[index] + 0.1 * gradient
            self.second[index] = 0.999 * self.second[index] + 0.001 * np.square(
                gradient
            )
            first = self.first[index] / (1.0 - 0.9**self.step)
            second = self.second[index] / (1.0 - 0.999**self.step)
            parameter -= self.learning_rate * first / (np.sqrt(second) + 1e-8)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class _Autoencoder:
    """A nonlinear bottleneck learned only from raw sensor reconstruction."""

    def __init__(
        self, config: RepresentationConfig, rng: np.random.Generator
    ) -> None:
        self.config = config
        encoder_scale = math.sqrt(2.0 / (config.sensor_dim + config.latent_dim))
        decoder_scale = math.sqrt(2.0 / (config.latent_dim + config.sensor_dim))
        self.encoder_weights = rng.normal(
            0.0, encoder_scale, (config.sensor_dim, config.latent_dim)
        )
        self.encoder_bias = np.zeros(config.latent_dim, dtype=float)
        self.decoder_weights = rng.normal(
            0.0, decoder_scale, (config.latent_dim, config.sensor_dim)
        )
        self.decoder_bias = np.zeros(config.sensor_dim, dtype=float)

    def copy(self) -> "_Autoencoder":
        duplicate = object.__new__(_Autoencoder)
        duplicate.config = self.config
        duplicate.encoder_weights = self.encoder_weights.copy()
        duplicate.encoder_bias = self.encoder_bias.copy()
        duplicate.decoder_weights = self.decoder_weights.copy()
        duplicate.decoder_bias = self.decoder_bias.copy()
        return duplicate

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.encoder_weights,
            self.encoder_bias,
            self.decoder_weights,
            self.decoder_bias,
        )

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for item in self.arrays():
            digest.update(np.asarray(item, dtype="<f8").tobytes())
        return digest.hexdigest()[:16]

    def distance(self, other: "_Autoencoder") -> float:
        return math.sqrt(
            sum(
                float(np.sum(np.square(left - right)))
                for left, right in zip(self.arrays(), other.arrays())
            )
        )

    def encode(self, sensors: np.ndarray) -> np.ndarray:
        return np.tanh(sensors @ self.encoder_weights + self.encoder_bias)

    def reconstruct(self, sensors: np.ndarray) -> np.ndarray:
        latent = self.encode(sensors)
        return _sigmoid(latent @ self.decoder_weights + self.decoder_bias)

    def loss(self, sensors: np.ndarray) -> float:
        reconstruction = self.reconstruct(sensors)
        weights = 1.0 + 4.0 * sensors
        return float(np.mean(weights * np.square(reconstruction - sensors)))

    def latent_variance(self, sensors: np.ndarray) -> float:
        return float(np.mean(np.var(self.encode(sensors), axis=0)))

    def train(self, sensors: np.ndarray, *, epochs: int) -> None:
        parameters = list(self.arrays())
        optimizer = _Adam(
            parameters, self.config.representation_learning_rate
        )
        normalizer = float(sensors.shape[0] * sensors.shape[1])
        corruption_rng = np.random.default_rng(self.config.seed + sensors.shape[0])
        for _ in range(epochs):
            # Masked reconstruction prevents the network from merely copying
            # individual pixels.  It must encode recurring spatial structure
            # that can be reused by later operators.
            keep = corruption_rng.random(sensors.shape) >= 0.20
            corrupted = sensors * keep
            latent = self.encode(corrupted)
            reconstruction = _sigmoid(
                latent @ self.decoder_weights + self.decoder_bias
            )
            weights = 1.0 + 4.0 * sensors
            reconstruction_error = (
                2.0
                * weights
                * (reconstruction - sensors)
                * reconstruction
                * (1.0 - reconstruction)
                / normalizer
            )
            grad_decoder_weights = (
                latent.T @ reconstruction_error
                + self.config.l2 * self.decoder_weights
            )
            grad_decoder_bias = np.sum(reconstruction_error, axis=0)
            latent_error = (
                reconstruction_error @ self.decoder_weights.T
            ) * (1.0 - np.square(latent))
            grad_encoder_weights = (
                corrupted.T @ latent_error + self.config.l2 * self.encoder_weights
            )
            grad_encoder_bias = np.sum(latent_error, axis=0)
            optimizer.update(
                (
                    grad_encoder_weights,
                    grad_encoder_bias,
                    grad_decoder_weights,
                    grad_decoder_bias,
                )
            )


class _OperatorHead:
    """A modular reasoning operator over the learned latent state."""

    def __init__(
        self, config: RepresentationConfig, rng: np.random.Generator
    ) -> None:
        self.config = config
        hidden_scale = math.sqrt(
            2.0 / (config.latent_dim + config.head_hidden_dim)
        )
        output_scale = math.sqrt(2.0 / (config.head_hidden_dim + 1))
        self.hidden_weights = rng.normal(
            0.0,
            hidden_scale,
            (config.latent_dim, config.head_hidden_dim),
        )
        self.hidden_bias = np.zeros(config.head_hidden_dim, dtype=float)
        self.output_weights = rng.normal(
            0.0, output_scale, (config.head_hidden_dim, 1)
        )
        self.output_bias = np.zeros(1, dtype=float)

    def copy(self) -> "_OperatorHead":
        duplicate = object.__new__(_OperatorHead)
        duplicate.config = self.config
        duplicate.hidden_weights = self.hidden_weights.copy()
        duplicate.hidden_bias = self.hidden_bias.copy()
        duplicate.output_weights = self.output_weights.copy()
        duplicate.output_bias = self.output_bias.copy()
        return duplicate

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
        )

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for item in self.arrays():
            digest.update(np.asarray(item, dtype="<f8").tobytes())
        return digest.hexdigest()[:16]

    def distance(self, other: "_OperatorHead") -> float:
        return math.sqrt(
            sum(
                float(np.sum(np.square(left - right)))
                for left, right in zip(self.arrays(), other.arrays())
            )
        )

    def probabilities(self, latent: np.ndarray) -> np.ndarray:
        hidden = np.tanh(latent @ self.hidden_weights + self.hidden_bias)
        return _sigmoid(hidden @ self.output_weights + self.output_bias).reshape(-1)

    def train(self, latent: np.ndarray, targets: np.ndarray, *, epochs: int) -> None:
        parameters = list(self.arrays())
        optimizer = _Adam(parameters, self.config.operator_learning_rate)
        target_column = targets.reshape(-1, 1)
        for _ in range(epochs):
            hidden = np.tanh(latent @ self.hidden_weights + self.hidden_bias)
            probabilities = _sigmoid(
                hidden @ self.output_weights + self.output_bias
            )
            output_error = (probabilities - target_column) / len(targets)
            grad_output_weights = (
                hidden.T @ output_error + self.config.l2 * self.output_weights
            )
            grad_output_bias = np.sum(output_error, axis=0)
            hidden_error = (
                output_error @ self.output_weights.T
            ) * (1.0 - np.square(hidden))
            grad_hidden_weights = (
                latent.T @ hidden_error + self.config.l2 * self.hidden_weights
            )
            grad_hidden_bias = np.sum(hidden_error, axis=0)
            optimizer.update(
                (
                    grad_hidden_weights,
                    grad_hidden_bias,
                    grad_output_weights,
                    grad_output_bias,
                )
            )


@dataclass
class _RepresentationRecord:
    model: _Autoencoder
    version: int
    observations_seen: int
    replay: tuple[RawObservation, ...]
    validation_loss: float
    latent_variance: float


@dataclass
class _OperatorRecord:
    head: _OperatorHead
    version: int
    examples_seen: int
    replay: tuple[GroundedExample, ...]
    probes: tuple[GroundedExample, ...]
    validation_accuracy: float
    representation_version: int


def _observation_matrix(
    observations: Sequence[RawObservation], sensor_dim: int
) -> np.ndarray:
    if not observations:
        raise ValueError("at least one raw observation is required")
    if any(len(item.values) != sensor_dim for item in observations):
        raise ValueError(f"every raw observation must contain {sensor_dim} values")
    return np.asarray([item.values for item in observations], dtype=float)


def _example_matrix(
    examples: Sequence[GroundedExample], sensor_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("at least one grounded example is required")
    observations = tuple(item.observation for item in examples)
    sensors = _observation_matrix(observations, sensor_dim)
    targets = np.asarray([item.target for item in examples], dtype=float)
    return sensors, targets


def _sample_across_lifetime(items: Sequence, capacity: int) -> tuple:
    if len(items) <= capacity:
        return tuple(items)
    indices = np.linspace(0, len(items) - 1, capacity, dtype=int)
    return tuple(items[int(index)] for index in indices)


def _merge_observations(
    existing: Sequence[RawObservation],
    additions: Sequence[RawObservation],
    capacity: int,
) -> tuple[RawObservation, ...]:
    unique: dict[tuple[float, ...], RawObservation] = {}
    for item in tuple(existing) + tuple(additions):
        unique[item.values] = item
    return _sample_across_lifetime(tuple(unique.values()), capacity)


def _merge_examples(
    existing: Sequence[GroundedExample],
    additions: Sequence[GroundedExample],
    capacity: int,
) -> tuple[GroundedExample, ...]:
    unique: dict[tuple[tuple[float, ...], int], GroundedExample] = {}
    for item in tuple(existing) + tuple(additions):
        unique[(item.observation.values, item.target)] = item
    return _sample_across_lifetime(tuple(unique.values()), capacity)


def _operator_accuracy(
    encoder: _Autoencoder,
    head: _OperatorHead,
    examples: Sequence[GroundedExample],
    sensor_dim: int,
) -> float:
    sensors, targets = _example_matrix(examples, sensor_dim)
    predictions = head.probabilities(encoder.encode(sensors)) >= 0.5
    return float(np.mean(predictions == targets.astype(bool)))


class GroundedRepresentationSystem:
    """Protected shared representation with reusable modular operators."""

    CHECKPOINT_VERSION = 1

    def __init__(self, config: RepresentationConfig | None = None) -> None:
        self.config = config or RepresentationConfig()
        self._representation: _RepresentationRecord | None = None
        self._operators: dict[str, _OperatorRecord] = {}

    def __len__(self) -> int:
        return len(self._operators)

    def _seed_for(self, purpose: str) -> int:
        digest = hashlib.sha256(
            f"{self.config.seed}:{purpose}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _new_representation(self) -> _Autoencoder:
        return _Autoencoder(
            self.config,
            np.random.default_rng(self._seed_for("shared-representation")),
        )

    def _new_head(self, name: str) -> _OperatorHead:
        return _OperatorHead(
            self.config,
            np.random.default_rng(self._seed_for(f"operator:{name}")),
        )

    @staticmethod
    def _clean_name(name: str) -> str:
        clean = name.strip()
        if not clean:
            raise ValueError("operator name is required")
        if len(clean) > 120:
            raise ValueError("operator name must be at most 120 characters")
        return clean

    def representation(self) -> RepresentationState:
        if self._representation is None:
            raise RuntimeError("Athena has not learned a representation yet")
        record = self._representation
        return RepresentationState(
            version=record.version,
            sensor_dim=self.config.sensor_dim,
            latent_dim=self.config.latent_dim,
            observations_seen=record.observations_seen,
            replay_observations=len(record.replay),
            validation_loss=record.validation_loss,
            latent_variance=record.latent_variance,
            operator_count=len(self._operators),
            checksum=record.model.checksum(),
        )

    def operators(self) -> tuple[ReasoningOperator, ...]:
        return tuple(self.get_operator(name) for name in sorted(self._operators))

    def get_operator(self, name: str) -> ReasoningOperator:
        clean = self._clean_name(name)
        try:
            record = self._operators[clean]
        except KeyError as exc:
            raise KeyError(f"unknown reasoning operator: {clean}") from exc
        return ReasoningOperator(
            name=clean,
            version=record.version,
            examples_seen=record.examples_seen,
            replay_examples=len(record.replay),
            probe_examples=len(record.probes),
            validation_accuracy=record.validation_accuracy,
            representation_version=record.representation_version,
            checksum=record.head.checksum(),
        )

    def encode(self, observation: RawObservation) -> tuple[float, ...]:
        if self._representation is None:
            raise RuntimeError("Athena has not learned a representation yet")
        sensors = _observation_matrix((observation,), self.config.sensor_dim)
        return tuple(float(item) for item in self._representation.model.encode(sensors)[0])

    def reconstruct(self, observation: RawObservation) -> tuple[float, ...]:
        if self._representation is None:
            raise RuntimeError("Athena has not learned a representation yet")
        sensors = _observation_matrix((observation,), self.config.sensor_dim)
        reconstruction = self._representation.model.reconstruct(sensors)[0]
        return tuple(float(item) for item in reconstruction)

    def learn_representation(
        self,
        training_observations: Sequence[RawObservation],
        validation_observations: Sequence[RawObservation],
        *,
        epochs: int | None = None,
    ) -> RepresentationReport:
        """Train a candidate encoder and protect every dependent operator."""

        training = tuple(training_observations)
        validation = tuple(validation_observations)
        _observation_matrix(training, self.config.sensor_dim)
        validation_sensors = _observation_matrix(validation, self.config.sensor_dim)
        if len(validation) < self.config.minimum_representation_validation:
            raise ValueError(
                "needs at least "
                f"{self.config.minimum_representation_validation} held-out observations"
            )
        training_epochs = (
            self.config.representation_epochs if epochs is None else int(epochs)
        )
        if training_epochs < 1:
            raise ValueError("epochs must be >= 1")

        existing = self._representation
        recruited = existing is None
        starting = self._new_representation() if recruited else existing.model
        candidate = starting.copy()
        checksum_before = starting.checksum()
        before_loss = starting.loss(validation_sensors)
        old_replay = () if existing is None else existing.replay
        old_sensors = (
            None
            if not old_replay
            else _observation_matrix(old_replay, self.config.sensor_dim)
        )
        protected_loss_before = (
            before_loss if old_sensors is None else starting.loss(old_sensors)
        )
        training_replay = _merge_observations(
            old_replay,
            training,
            self.config.observation_replay_capacity,
        )
        candidate.train(
            _observation_matrix(training_replay, self.config.sensor_dim),
            epochs=training_epochs,
        )
        candidate_loss = candidate.loss(validation_sensors)
        protected_loss_after = (
            candidate_loss if old_sensors is None else candidate.loss(old_sensors)
        )
        latent_variance = candidate.latent_variance(validation_sensors)

        candidate_heads: dict[str, _OperatorHead] = {}
        operator_accuracies: dict[str, float] = {}
        operators_pass = True
        for name, record in self._operators.items():
            head = record.head.copy()
            replay_sensors, replay_targets = _example_matrix(
                record.replay, self.config.sensor_dim
            )
            head.train(
                candidate.encode(replay_sensors),
                replay_targets,
                epochs=self.config.operator_epochs,
            )
            previous_accuracy = _operator_accuracy(
                starting,
                record.head,
                record.probes,
                self.config.sensor_dim,
            )
            accuracy = _operator_accuracy(
                candidate,
                head,
                record.probes,
                self.config.sensor_dim,
            )
            candidate_heads[name] = head
            operator_accuracies[name] = accuracy
            required = max(
                self.config.operator_validation_threshold,
                previous_accuracy - self.config.operator_regression_tolerance,
            )
            operators_pass = operators_pass and accuracy >= required

        reconstruction_pass = (
            candidate_loss <= self.config.maximum_reconstruction_loss
            and candidate_loss < before_loss
        )
        replay_pass = protected_loss_after <= (
            protected_loss_before + self.config.reconstruction_regression_tolerance
        )
        variance_pass = latent_variance >= self.config.minimum_latent_variance
        promoted = reconstruction_pass and replay_pass and variance_pass and operators_pass
        weight_delta = candidate.distance(starting)
        minimum_operator_accuracy = (
            min(operator_accuracies.values()) if operator_accuracies else 1.0
        )

        if promoted:
            protected_replay = _merge_observations(
                old_replay,
                training + validation,
                self.config.observation_replay_capacity,
            )
            version = 1 if existing is None else existing.version + 1
            observations_seen = len(training) + (
                0 if existing is None else existing.observations_seen
            )
            self._representation = _RepresentationRecord(
                model=candidate,
                version=version,
                observations_seen=observations_seen,
                replay=protected_replay,
                validation_loss=candidate_loss,
                latent_variance=latent_variance,
            )
            for name, head in candidate_heads.items():
                old = self._operators[name]
                self._operators[name] = _OperatorRecord(
                    head=head,
                    version=old.version + 1,
                    examples_seen=old.examples_seen,
                    replay=old.replay,
                    probes=old.probes,
                    validation_accuracy=operator_accuracies[name],
                    representation_version=version,
                )
            checksum_after = candidate.checksum()
            reason = (
                "learned and grounded a new latent sensor representation"
                if recruited
                else "improved the shared representation without regressing operators"
            )
            replay_observations = len(protected_replay)
        else:
            version = 0 if existing is None else existing.version
            checksum_after = checksum_before
            replay_observations = 0 if existing is None else len(existing.replay)
            if not reconstruction_pass:
                reason = "candidate failed held-out reconstruction improvement"
            elif not replay_pass:
                reason = "candidate would forget protected observations"
            elif not variance_pass:
                reason = "candidate latent representation collapsed"
            else:
                reason = "candidate would regress a protected reasoning operator"

        return RepresentationReport(
            promoted=promoted,
            recruited=recruited,
            rolled_back=not promoted,
            version=version,
            before_loss=before_loss,
            candidate_loss=candidate_loss,
            protected_loss_before=protected_loss_before,
            protected_loss_after=protected_loss_after,
            latent_variance=latent_variance,
            weight_delta=weight_delta,
            training_observations=len(training),
            validation_observations=len(validation),
            replay_observations=replay_observations,
            protected_operators=len(self._operators),
            minimum_operator_accuracy=minimum_operator_accuracy,
            checksum_before=checksum_before,
            checksum_after=checksum_after,
            reason=reason,
        )

    def _untrained_representation_accuracy(
        self,
        name: str,
        training: Sequence[GroundedExample],
        validation: Sequence[GroundedExample],
        *,
        epochs: int,
    ) -> float:
        random_encoder = self._new_representation()
        # Use the exact same head initialization as the learned-encoder path so
        # the only experimental difference is the representation.
        head = self._new_head(name)
        sensors, targets = _example_matrix(training, self.config.sensor_dim)
        head.train(random_encoder.encode(sensors), targets, epochs=epochs)
        return _operator_accuracy(
            random_encoder, head, validation, self.config.sensor_dim
        )

    def learn_operator(
        self,
        name: str,
        training_examples: Sequence[GroundedExample],
        validation_examples: Sequence[GroundedExample],
        *,
        epochs: int | None = None,
    ) -> OperatorReport:
        """Learn a reusable operator while keeping the representation frozen."""

        if self._representation is None:
            raise RuntimeError("learn a representation before learning an operator")
        clean = self._clean_name(name)
        training = tuple(training_examples)
        validation = tuple(validation_examples)
        training_sensors, training_targets = _example_matrix(
            training, self.config.sensor_dim
        )
        _example_matrix(validation, self.config.sensor_dim)
        if len(validation) < self.config.minimum_operator_validation:
            raise ValueError(
                "needs at least "
                f"{self.config.minimum_operator_validation} held-out operator cases"
            )
        training_epochs = self.config.operator_epochs if epochs is None else int(epochs)
        if training_epochs < 1:
            raise ValueError("epochs must be >= 1")

        encoder = self._representation.model
        representation_checksum = encoder.checksum()
        existing = self._operators.get(clean)
        recruited = existing is None
        starting = self._new_head(clean) if recruited else existing.head
        candidate = starting.copy()
        before_accuracy = _operator_accuracy(
            encoder, starting, validation, self.config.sensor_dim
        )
        old_replay = () if existing is None else existing.replay
        replay = _merge_examples(
            old_replay,
            training,
            self.config.operator_replay_capacity,
        )
        replay_sensors, replay_targets = _example_matrix(
            replay, self.config.sensor_dim
        )
        candidate.train(
            encoder.encode(replay_sensors), replay_targets, epochs=training_epochs
        )
        candidate_accuracy = _operator_accuracy(
            encoder, candidate, validation, self.config.sensor_dim
        )
        regression_accuracy = (
            1.0
            if not old_replay
            else _operator_accuracy(
                encoder, candidate, old_replay, self.config.sensor_dim
            )
        )
        baseline_accuracy = self._untrained_representation_accuracy(
            clean,
            training,
            validation,
            epochs=training_epochs,
        )
        passed_new = (
            candidate_accuracy >= self.config.operator_validation_threshold
            and candidate_accuracy >= before_accuracy
        )
        passed_regression = regression_accuracy >= (
            1.0 - self.config.operator_regression_tolerance
        )
        promoted = passed_new and passed_regression

        if promoted:
            probes = _merge_examples(
                () if existing is None else existing.probes,
                validation,
                self.config.operator_probe_capacity,
            )
            version = 1 if existing is None else existing.version + 1
            examples_seen = len(training) + (
                0 if existing is None else existing.examples_seen
            )
            self._operators[clean] = _OperatorRecord(
                head=candidate,
                version=version,
                examples_seen=examples_seen,
                replay=replay,
                probes=probes,
                validation_accuracy=candidate_accuracy,
                representation_version=self._representation.version,
            )
            reason = (
                "grounded a new reasoning operator in the learned representation"
                if recruited
                else "updated and replay-verified the reasoning operator"
            )
            retained_replay_examples = len(replay)
        else:
            version = 0 if existing is None else existing.version
            reason = (
                "candidate failed held-out operator verification"
                if not passed_new
                else "candidate would regress protected operator cases"
            )
            retained_replay_examples = (
                0 if existing is None else len(existing.replay)
            )

        return OperatorReport(
            name=clean,
            promoted=promoted,
            recruited=recruited,
            rolled_back=not promoted,
            version=version,
            before_accuracy=before_accuracy,
            candidate_accuracy=candidate_accuracy,
            regression_accuracy=regression_accuracy,
            untrained_representation_accuracy=baseline_accuracy,
            head_weight_delta=candidate.distance(starting),
            representation_checksum=representation_checksum,
            training_examples=len(training),
            validation_examples=len(validation),
            replay_examples=retained_replay_examples,
            reason=reason,
        )

    def predict_probability(self, name: str, observation: RawObservation) -> float:
        if self._representation is None:
            raise RuntimeError("Athena has not learned a representation yet")
        record = self._operators[self._clean_name(name)]
        sensors = _observation_matrix((observation,), self.config.sensor_dim)
        latent = self._representation.model.encode(sensors)
        return float(record.head.probabilities(latent)[0])

    def predict(self, name: str, observation: RawObservation) -> int:
        return int(self.predict_probability(name, observation) >= 0.5)

    def evaluate(
        self, name: str, examples: Sequence[GroundedExample]
    ) -> float:
        if self._representation is None:
            raise RuntimeError("Athena has not learned a representation yet")
        record = self._operators[self._clean_name(name)]
        return _operator_accuracy(
            self._representation.model,
            record.head,
            tuple(examples),
            self.config.sensor_dim,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        representation_state = None
        if self._representation is not None:
            record = self._representation
            for suffix, value in zip(
                ("encoder_weights", "encoder_bias", "decoder_weights", "decoder_bias"),
                record.model.arrays(),
            ):
                arrays[f"representation_{suffix}"] = value
            representation_state = {
                "version": record.version,
                "observations_seen": record.observations_seen,
                "validation_loss": record.validation_loss,
                "latent_variance": record.latent_variance,
                "replay": [asdict(item) for item in record.replay],
            }

        operators = []
        for index, name in enumerate(sorted(self._operators)):
            record = self._operators[name]
            for suffix, value in zip(
                ("hidden_weights", "hidden_bias", "output_weights", "output_bias"),
                record.head.arrays(),
            ):
                arrays[f"operator_{index}_{suffix}"] = value
            operators.append(
                {
                    "name": name,
                    "version": record.version,
                    "examples_seen": record.examples_seen,
                    "validation_accuracy": record.validation_accuracy,
                    "representation_version": record.representation_version,
                    "replay": [asdict(item) for item in record.replay],
                    "probes": [asdict(item) for item in record.probes],
                }
            )

        metadata = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "representation": representation_state,
            "operators": operators,
        }
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata)),
                **arrays,
            )
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GroundedRepresentationSystem":
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            if metadata.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
                raise ValueError("unsupported representation checkpoint version")
            system = cls(RepresentationConfig(**metadata["config"]))
            representation_state = metadata.get("representation")
            if representation_state is not None:
                model = system._new_representation()
                model.encoder_weights = checkpoint[
                    "representation_encoder_weights"
                ].copy()
                model.encoder_bias = checkpoint["representation_encoder_bias"].copy()
                model.decoder_weights = checkpoint[
                    "representation_decoder_weights"
                ].copy()
                model.decoder_bias = checkpoint["representation_decoder_bias"].copy()
                replay = tuple(
                    RawObservation(tuple(float(value) for value in item["values"]))
                    for item in representation_state["replay"]
                )
                system._representation = _RepresentationRecord(
                    model=model,
                    version=int(representation_state["version"]),
                    observations_seen=int(representation_state["observations_seen"]),
                    replay=replay,
                    validation_loss=float(representation_state["validation_loss"]),
                    latent_variance=float(representation_state["latent_variance"]),
                )

            for index, state in enumerate(metadata.get("operators", [])):
                name = str(state["name"])
                head = system._new_head(name)
                head.hidden_weights = checkpoint[
                    f"operator_{index}_hidden_weights"
                ].copy()
                head.hidden_bias = checkpoint[f"operator_{index}_hidden_bias"].copy()
                head.output_weights = checkpoint[
                    f"operator_{index}_output_weights"
                ].copy()
                head.output_bias = checkpoint[f"operator_{index}_output_bias"].copy()

                def restore_examples(key: str) -> tuple[GroundedExample, ...]:
                    return tuple(
                        GroundedExample(
                            RawObservation(
                                tuple(
                                    float(value)
                                    for value in item["observation"]["values"]
                                )
                            ),
                            int(item["target"]),
                        )
                        for item in state[key]
                    )

                system._operators[name] = _OperatorRecord(
                    head=head,
                    version=int(state["version"]),
                    examples_seen=int(state["examples_seen"]),
                    replay=restore_examples("replay"),
                    probes=restore_examples("probes"),
                    validation_accuracy=float(state["validation_accuracy"]),
                    representation_version=int(state["representation_version"]),
                )
            return system


def _visual_scene(
    rng: np.random.Generator,
    *,
    noise: float,
    brightness: float,
) -> tuple[RawObservation, tuple[float, float, float, float]]:
    if noise < 0.0:
        raise ValueError("noise must be >= 0")
    if not 0.25 <= brightness <= 1.25:
        raise ValueError("brightness must be in [0.25, 1.25]")
    while True:
        bright_row, bright_column = rng.uniform(0.8, 6.2, 2)
        dim_row, dim_column = rng.uniform(0.8, 6.2, 2)
        if math.hypot(bright_row - dim_row, bright_column - dim_column) >= 1.8:
            break
    rows, columns = np.mgrid[0:8, 0:8]
    bright_blob = np.exp(
        -(
            np.square(rows - bright_row) + np.square(columns - bright_column)
        )
        / (2.0 * 0.62**2)
    )
    dim_blob = np.exp(
        -(
            np.square(rows - dim_row) + np.square(columns - dim_column)
        )
        / (2.0 * 0.62**2)
    )
    # Two raw sensor channels make object identity observable in the same way
    # that colour channels do, while withholding the coordinates and relation.
    # Athena still receives 128 pixel intensities rather than engineered
    # position, distance, or ordering features.
    image = brightness * np.stack((0.95 * bright_blob, 0.78 * dim_blob))
    if noise:
        image += rng.normal(0.0, noise, image.shape)
    image = np.clip(image, 0.0, 1.0)
    observation = RawObservation(tuple(float(value) for value in image.reshape(-1)))
    return observation, (bright_row, bright_column, dim_row, dim_column)


def make_visual_observations(
    count: int,
    seed: int,
    *,
    noise: float = 0.025,
    brightness: float = 1.0,
) -> tuple[RawObservation, ...]:
    """Generate raw sensor grids without exposing their scene coordinates."""

    if count < 1:
        raise ValueError("count must be >= 1")
    rng = np.random.default_rng(seed)
    return tuple(
        _visual_scene(rng, noise=noise, brightness=brightness)[0]
        for _ in range(count)
    )


def make_visual_cases(
    rule: VisualRule,
    count: int,
    seed: int,
    *,
    noise: float = 0.025,
    brightness: float = 1.0,
) -> tuple[GroundedExample, ...]:
    """Create balanced labels while returning pixels—not coordinates—to Athena."""

    if rule not in ("horizontal_order", "vertical_order", "far_apart"):
        raise ValueError(f"unknown visual rule: {rule}")
    if count < 2:
        raise ValueError("count must be >= 2")
    rng = np.random.default_rng(seed)
    quotas = {0: count // 2, 1: count - count // 2}
    cases: list[GroundedExample] = []
    while quotas[0] or quotas[1]:
        observation, coordinates = _visual_scene(
            rng, noise=noise, brightness=brightness
        )
        bright_row, bright_column, dim_row, dim_column = coordinates
        if rule == "horizontal_order":
            margin = bright_column - dim_column
            if abs(margin) < 0.8:
                continue
            target = int(margin > 0.0)
        elif rule == "vertical_order":
            margin = dim_row - bright_row
            if abs(margin) < 0.8:
                continue
            target = int(margin > 0.0)
        else:
            distance = math.hypot(
                bright_row - dim_row, bright_column - dim_column
            )
            if abs(distance - 4.0) < 0.45:
                continue
            target = int(distance > 4.0)
        if quotas[target] == 0:
            continue
        quotas[target] -= 1
        cases.append(GroundedExample(observation, target))
    rng.shuffle(cases)
    return tuple(cases)
