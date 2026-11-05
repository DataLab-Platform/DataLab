# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Registered features agree with the Sigima contracts of their functions.

Sigima operation contracts are the only source of patterns: a ``register_*``
call that disagrees with the contract of its function is an error.
"""

from __future__ import annotations

from sigima.proc.contracts import contract_for_function

from datalab.gui.processor.geometry_postprocess import unwrap_computation_function
from datalab.tests.features.common.provenance_unit_test import app

#: A 1-to-n command runs one 1-to-1 invocation per parameter set.
_CONTRACT_PATTERN = {"1_to_n": "1_to_1"}


def test_registered_features_match_their_contract() -> None:
    """Pattern and X-alignment behaviour of every built-in feature."""
    checked = 0
    problems = []
    with app() as win:
        for panel in (win.signalpanel, win.imagepanel):
            for feature in panel.processor.computing_registry.values():
                if feature.function is None or feature.plugin_origin is not None:
                    continue
                function = unwrap_computation_function(feature.function)
                contract = contract_for_function(function)
                name = f"{panel.PANEL_STR_ID}:{feature.feature_id}"
                if contract is None:
                    problems.append(f"{name}: no Sigima contract")
                    continue
                checked += 1
                expected = _CONTRACT_PATTERN.get(feature.pattern, feature.pattern)
                if contract.pattern != expected:
                    problems.append(
                        f"{name}: registered {feature.pattern}, "
                        f"contract {contract.pattern}"
                    )
                multi_signal = contract.pattern in ("2_to_1", "n_to_1") and (
                    contract.inputs[0].kind == "signal"
                )
                # Without a declared policy, the inputs are used as they are: the
                # legacy X-array check must not interpolate them.
                if (
                    multi_signal
                    and contract.x_alignment is None
                    and not feature.skip_xarray_compat
                ):
                    problems.append(
                        f"{name}: no X-alignment policy declared, but the legacy "
                        "X-array check is enabled"
                    )
    assert not problems, "\n".join(problems)
    assert checked > 150
