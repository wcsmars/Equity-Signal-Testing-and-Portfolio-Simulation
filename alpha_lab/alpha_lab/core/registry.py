"""Tiny name -> class registry so configs can instantiate components by name."""

from __future__ import annotations

from typing import Callable, TypeVar

from alpha_lab.core.errors import ConfigError

T = TypeVar("T")


class Registry:
    """Maps string names to classes. One instance per component kind.

    Usage:
        FEATURES = Registry("feature")

        @FEATURES.register("momentum")
        class Momentum(Feature): ...

        FEATURES.create("momentum", window=252, skip=21)
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            if name in self._items:
                raise ConfigError(f"{self.kind} '{name}' registered twice")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type:
        try:
            return self._items[name]
        except KeyError:
            raise ConfigError(
                f"unknown {self.kind} '{name}'; available: {self.names()}"
            ) from None

    def create(self, name: str, **params):
        try:
            return self.get(name)(**params)
        except TypeError as exc:
            raise ConfigError(f"bad params for {self.kind} '{name}': {exc}") from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
