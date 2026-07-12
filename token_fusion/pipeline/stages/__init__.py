"""All Fusion Pipeline stages.

NOTES:
 - Abbrev and TokenOpt stages exist but are NOT in the default pipeline.
   They save <5 tokens on typical content — irrelevant at modern context windows.
   Include explicitly if you need micro-optimization for high-volume short prompts.
 - Neurosyntax NEVER strips comments or docstrings. Comments carry intent.
"""

from .quantum_lock import QuantumLockStage
from .cortex import CortexStage
from .photon import PhotonStage
from .rle import RLEStage
from .semantic_dedup import SemanticDedupStage
from .ionizer import IonizerStage
from .log_crunch import LogCrunchStage
from .search_crunch import SearchCrunchStage
from .diff_crunch import DiffCrunchStage
from .structural_collapse import StructuralCollapseStage
from .neurosyntax import NeurosyntaxStage
from .nexus import NexusStage
from .token_opt import TokenOptStage
from .abbrev import AbbrevStage

__all__ = [
    "QuantumLockStage",
    "CortexStage",
    "PhotonStage",
    "RLEStage",
    "SemanticDedupStage",
    "IonizerStage",
    "LogCrunchStage",
    "SearchCrunchStage",
    "DiffCrunchStage",
    "StructuralCollapseStage",
    "NeurosyntaxStage",
    "NexusStage",
    "TokenOptStage",
    "AbbrevStage",
]

# Default pipeline: stages that provide meaningful token reduction.
# Abbrev and TokenOpt excluded — they save <5 tokens on typical content.
DEFAULT_STAGES = [
    QuantumLockStage,
    CortexStage,
    PhotonStage,
    RLEStage,
    SemanticDedupStage,
    IonizerStage,
    LogCrunchStage,
    SearchCrunchStage,
    DiffCrunchStage,
    StructuralCollapseStage,
    NeurosyntaxStage,
    NexusStage,
]

# All stages including micro-optimizations
ALL_STAGES = [
    QuantumLockStage,
    CortexStage,
    PhotonStage,
    RLEStage,
    SemanticDedupStage,
    IonizerStage,
    LogCrunchStage,
    SearchCrunchStage,
    DiffCrunchStage,
    StructuralCollapseStage,
    NeurosyntaxStage,
    NexusStage,
    TokenOptStage,
    AbbrevStage,
]
