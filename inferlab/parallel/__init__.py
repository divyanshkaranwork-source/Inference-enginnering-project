"""5D parallelism, simulated: DP, TP (+SP), PP, CP (ring / Ulysses), EP.

Every strategy runs on a ``SimCluster`` of virtual GPUs, is checked against a
single-device computation, and reports the communication it would need.
"""

from inferlab.parallel.cluster import SimCluster
from inferlab.parallel.data_parallel import simulate_dp
from inferlab.parallel.expert_parallel import MoELayer, ep_forward
from inferlab.parallel.pipeline_parallel import pp_forward
from inferlab.parallel.planner import Plan, evaluate, recommend
from inferlab.parallel.sequence_parallel import ring_attention, ulysses_attention
from inferlab.parallel.tensor_parallel import tp_forward

__all__ = [
    "MoELayer",
    "Plan",
    "SimCluster",
    "ep_forward",
    "evaluate",
    "pp_forward",
    "recommend",
    "ring_attention",
    "simulate_dp",
    "tp_forward",
    "ulysses_attention",
]
