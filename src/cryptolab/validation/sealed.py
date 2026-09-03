"""The sealed test-period token (SPEC.md §10.1).

The test period may be read **once, ever, per strategy family**. That rule is enforced in code:
`data.store` refuses to serve test-period rows without a valid token, and tokens are issued only by
`validation.registry`, which records the consumption in the append-only trial registry.

This module holds only the token *type* and its verification, so that `data.store` and
`validation.registry` can share it without an import cycle.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import ClassVar, Final

_TOKEN_BYTES: Final[int] = 16


class SealedPeriodError(PermissionError):
    """Raised on any attempt to read the sealed test period without a valid, unspent token."""


@dataclass(frozen=True, slots=True)
class TestPeriodToken:
    """A one-time authorisation to read sealed test-period data.

    Instances are only meaningful when issued by `validation.registry.TrialRegistry.issue_test_token`,
    which is what binds `strategy_family` to a row in the registry. Constructing one by hand does not
    grant access: `store` verifies the digest against the registry that issued it.
    """

    # Not a pytest class, despite the name.
    __test__: ClassVar[bool] = False

    strategy_family: str
    nonce: str
    digest: str

    @staticmethod
    def mint(strategy_family: str, secret: bytes) -> TestPeriodToken:
        """Create a token bound to `strategy_family` under the registry's `secret`."""
        nonce = os.urandom(_TOKEN_BYTES).hex()
        return TestPeriodToken(
            strategy_family=strategy_family,
            nonce=nonce,
            digest=_digest(strategy_family, nonce, secret),
        )

    def verify(self, secret: bytes) -> bool:
        """True if this token was minted under `secret` for its own strategy family."""
        return hmac.compare_digest(self.digest, _digest(self.strategy_family, self.nonce, secret))


def _digest(strategy_family: str, nonce: str, secret: bytes) -> str:
    return hmac.new(secret, f"{strategy_family}:{nonce}".encode(), hashlib.sha256).hexdigest()
