"""Fail-safe hot loading for combat policies.

The injection/campaign process stays resident.  A changed policy source is
compiled into a fresh module; a syntax/runtime loading failure leaves the last
known-good policy active.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path

from .policy_api import POLICY_API_VERSION, CombatPolicy, PolicyDecision, PolicyObservation


class HotReloadPolicy:
    def __init__(self, path: Path, *, check_interval_frames: int = 30) -> None:
        if check_interval_frames <= 0:
            raise ValueError("check interval must be positive")
        self.path = path.resolve()
        self.check_interval_frames = check_interval_frames
        self.policy: CombatPolicy | None = None
        self.digest: str | None = None
        self.generation = 0
        self.reloads = 0
        self.reload_failures = 0
        self.last_error: str | None = None
        self.failed_digest: str | None = None
        self._rollback: tuple[CombatPolicy, str | None] | None = None
        self.maybe_reload(0, force=True)

    def _load(self, source: bytes, digest: str) -> CombatPolicy:
        self.generation += 1
        module_name = f"th105.policies._hot_{self.path.stem}_{self.generation}"
        module = types.ModuleType(module_name)
        module.__file__ = str(self.path)
        module.__package__ = "th105.policies"
        exec(compile(source, str(self.path), "exec"), module.__dict__)
        factory = getattr(module, "create_policy", None)
        if not callable(factory):
            raise TypeError("policy plugin must export create_policy()")
        policy = factory()
        if getattr(policy, "api_version", None) != POLICY_API_VERSION:
            raise RuntimeError(
                f"policy API mismatch: expected {POLICY_API_VERSION}, "
                f"got {getattr(policy, 'api_version', None)!r}"
            )
        if not callable(getattr(policy, "decide", None)):
            raise TypeError("policy object must implement decide(observation)")
        old = self.policy
        if old is not None:
            export = getattr(old, "export_state", None)
            restore = getattr(policy, "import_state", None)
            if callable(export) and callable(restore):
                restore(export())
        return policy

    def maybe_reload(self, frame: int, *, force: bool = False) -> bool:
        if not force and frame % self.check_interval_frames:
            return False
        try:
            source = self.path.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if not force and digest == self.digest:
                return False
            if not force and digest == self.failed_digest:
                return False
            policy = self._load(source, digest)
        except Exception as exc:
            self.reload_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if "digest" in locals():
                self.failed_digest = digest
            if self.policy is None:
                raise RuntimeError(
                    f"unable to load initial policy plugin {self.path}: {self.last_error}"
                ) from exc
            return False
        old = self.policy
        old_digest = self.digest
        self.policy = policy
        self.digest = digest
        self._rollback = (old, old_digest) if old is not None else None
        self.reloads += 1
        self.failed_digest = None
        self.last_error = None
        return True

    def decide(self, observation: PolicyObservation) -> PolicyDecision:
        self.maybe_reload(observation.frame)
        assert self.policy is not None
        try:
            decision = self.policy.decide(observation)
        except Exception as exc:
            self.reload_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failed_digest = self.digest
            if self._rollback is not None:
                self.policy, self.digest = self._rollback
                self._rollback = None
                try:
                    return self.policy.decide(observation)
                except Exception as fallback_exc:
                    self.last_error += (
                        f"; fallback {type(fallback_exc).__name__}: {fallback_exc}"
                    )
            # An initially loaded policy can contain a delayed branch failure
            # and has no prior generation. Keep the controller/injection alive
            # with a conservative spatial guard until the source digest changes.
            keys: set[str] = set()
            try:
                me, enemy = observation.state.p1, observation.state.p2
                keys.add("left" if me.x < enemy.x else "right")
            except (AttributeError, TypeError):
                pass
            return PolicyDecision(frozenset(keys), "policy-error-guard")
        return decision

    def status(self) -> dict[str, int | str | None | object]:
        try:
            policy_metrics = self.policy.metrics() if self.policy is not None else {}
        except Exception as exc:
            policy_metrics = {"metrics_error": f"{type(exc).__name__}: {exc}"}
        return {
            "path": str(self.path),
            "name": getattr(self.policy, "name", None),
            "generation": self.generation,
            "reloads": self.reloads,
            "reload_failures": self.reload_failures,
            "last_error": self.last_error,
            "sha256": self.digest,
            "metrics": policy_metrics,
        }
