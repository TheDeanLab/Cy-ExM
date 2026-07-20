"""Propagate the portable scheduler firewall into child Python processes."""

from __future__ import annotations

import os

if os.environ.get("CYEXM_PORTABLE_SCHEDULER_FIREWALL") == "1":
    from _scheduler_firewall import install_process_guards

    install_process_guards(setattr)
