#!/usr/bin/env python3
from collections.abc import Generator
from contextlib import contextmanager

NUMBERS = "0123456789"


@contextmanager
def NoneContext() -> Generator[None, None, None]:  # pragma: no cover
    yield


def NoneGenerator() -> Generator[None, None, None]:  # pragma: no cover
    while True:
        yield None
