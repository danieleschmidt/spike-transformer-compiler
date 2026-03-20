"""SynapseEncoder — encodes synaptic weights for event-driven (spike-based) processing."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .graph import Edge


class SynapseEncoder:
    """Encodes weight matrices into sparse spike-event representations."""

    def encode_weights(self, weight_matrix: np.ndarray) -> Dict:
        """
        Encode a weight matrix into a CSR-like sparse format.

        Returns a dict with:
          - non_zero_count (int)
          - compressed_rows (list of int)
          - compressed_cols (list of int)
          - compressed_vals (list of float)
        """
        rows, cols = np.nonzero(weight_matrix)
        vals = weight_matrix[rows, cols]
        return {
            "non_zero_count": int(len(vals)),
            "compressed_rows": rows.tolist(),
            "compressed_cols": cols.tolist(),
            "compressed_vals": vals.tolist(),
        }

    def spike_event(
        self,
        pre_neuron: int,
        post_neuron: int,
        weight: float,
        timestamp: int,
    ) -> Dict:
        """
        Create a single spike event dict.

        Returns::

            {
                "pre_neuron": int,
                "post_neuron": int,
                "weight": float,
                "timestamp": int,
            }
        """
        return {
            "pre_neuron": pre_neuron,
            "post_neuron": post_neuron,
            "weight": float(weight),
            "timestamp": timestamp,
        }

    def encode_layer(self, edge: Edge, timestamp_base: int) -> List[Dict]:
        """
        Encode all non-zero synaptic weights of an edge as spike events.

        Each non-zero weight entry becomes one spike event. Timestamp for
        each event is ``timestamp_base + pre_neuron_index``.

        Returns a list of spike event dicts.
        """
        if edge.weight_matrix is None:
            return []

        encoded = self.encode_weights(edge.weight_matrix)
        events: List[Dict] = []
        for pre, post, val in zip(
            encoded["compressed_rows"],
            encoded["compressed_cols"],
            encoded["compressed_vals"],
        ):
            events.append(
                self.spike_event(
                    pre_neuron=pre,
                    post_neuron=post,
                    weight=val,
                    timestamp=timestamp_base + pre,
                )
            )
        return events
