"""Fraud scenario identifiers persisted in generated transactions."""

from enum import IntEnum


class Scenario(IntEnum):
    LEGITIMATE = 0
    COMPROMISED_TERMINAL = 1
    STOLEN_CARD = 2
    CARD_TESTING = 3
    IMPOSSIBLE_TRAVEL = 4
