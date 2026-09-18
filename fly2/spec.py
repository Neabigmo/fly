"""fly2: the frozen specification of the whole-brain experiment.

Everything that defines the experiment lives here, so that a run cannot drift from the plan
and so the pre-flight and the trainer read the same numbers.  Nothing in this module
imports the previous codebase: ``fly2`` is a fresh implementation, and the only place the
old project is touched is the one-off data build that parses the public MaleCNS release
(``scripts/40_fly2_build.py``), because parsing a dataset correctly is not the thing that
was wrong before.

What is different from the previous attempt, and why
----------------------------------------------------
The previous substrates never contained a learning circuit.  Measured on the MaleCNS v1.0
annotations: ``core`` was the columnar optic lobe plus visual projection neurons, and
``full`` was the optic lobe plus its one-hop targets -- 89,397 optic-lobe intrinsic cells,
9,200 visual projection neurons, 563 centrifugal and **three** central-brain cells.  Neither
contained a single Kenyon cell, MBON or dopamine neuron.  Asking a substrate with no
associative learning centre to acquire structure beyond its training distribution is not an
experiment about a brain.

The whole-brain substrate below does contain them, joined from the neurotransmitter file's
cell-type vocabulary (11,751 types): 4,064 Kenyon cells, 97 MBONs, 24 dopaminergic PPL
neurons, 37 octopaminergic, 2,150 central-complex cells, 2,028 lateral-horn cells and 311
AOTU/tubercle cells, all within four hops of the retina and three of the read-out.

The learning rule follows the anatomy instead of the other way round: Kenyon cells are the
plastic population and the dopaminergic/octopaminergic cells are the third factor, which is
what the mushroom body is for.  Backpropagation-through-time is kept as an upper-bound
control, not as the default.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Substrate
# --------------------------------------------------------------------------- #
BRAIN = {
    #: neurons are the annotated ones that carry at least one edge in the released
    #: connectome; isolated cells cannot participate in a recurrence
    "keep_isolated": False,
    "w_scale": 0.5,
    #: per-neuron row normalisation.  The whole brain has a maximum in-degree of 11,530
    #: against a mean of 123, and the previous project already found that global
    #: normalisation diverges on circuits this heterogeneous.
    "normalization": "row",
    #: signs come from the transmitter identity of the presynaptic cell, applied *after*
    #: normalisation so |W| is unchanged and only the sign flips
    "signed": True,
    "unknown_sign": +1,
}

#: Membrane leak per superclass: ``h <- (1 - alpha) h + alpha * phi(drive)``.
#: Optic lobe is fast and feed-forward-ish, the central brain integrates; the numbers are
#: the only knob here that is not measured, so they are stated as a prior and swept once in
#: the pre-flight rather than tuned silently.
ALPHA_BY_SUPERCLASS: dict[str, float] = {
    "ol_sensory": 0.35,
    "ol_intrinsic": 0.30,
    "visual_projection": 0.20,
    "visual_centrifugal": 0.20,
    "cb_sensory": 0.15,
    "cb_intrinsic": 0.05,
    "cb_motor": 0.10,
    "ascending_neuron": 0.15,
    "descending_neuron": 0.10,
    "vnc_sensory": 0.20,
    "vnc_intrinsic": 0.08,
    "vnc_motor": 0.10,
    "__unassigned__": 0.10,
}
ALPHA_DEFAULT = 0.10

#: Populations, identified by annotation superclass and by cell-type vocabulary.
POOLS = {
    "retina": {"kind": "superclass", "value": "ol_sensory", "input": "lamina"},
    "odor": {"kind": "celltype_prefix", "value": ("orn",)},
    "readout_visual": {"kind": "superclass", "value": "visual_projection"},
    "readout_mb": {"kind": "celltype_prefix", "value": ("mbon",)},
    "kenyon": {"kind": "celltype_prefix", "value": ("kc",)},
    "dopamine": {"kind": "celltype_prefix", "value": ("ppl", "dan")},
    "octopamine": {"kind": "celltype_prefix", "value": ("oa-", "vum", "vpm")},
}

READOUT_POOL = "readout_visual"
READOUT_WINDOW = 2

# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #
WORLD = {
    "image_size": 32,
    #: one "moment" of the world = one retinal frame + one odour sample, T steps long
    #: recurrent steps per moment.  The visual path is two hops and the mushroom-body loop
    #: is three, so twelve steps is ample; the cost is linear in this number and a 32-step
    #: unroll bought nothing but time.
    "steps": 16,
    "batch": 16,
    #: dot stimuli reuse the geometry the pre-flight already validated for 1..7 items
    "n_min": 1,
    "n_max": 7,
    "radius_min": 1.13,
    "radius_max": 3.70,
    "area_target": 35.6,
    "min_gap": 1.5,
    "ring_radius": 9.5,
    #: odours: a fixed random binding matrix from receptor channels to ORNs
    "n_odour_channels": 24,
    "odour_sparsity": 0.25,
}

# --------------------------------------------------------------------------- #
# Education
# --------------------------------------------------------------------------- #
LEARN = {
    "objective": "predict_next",     # self-supervised: predict the next frame
    "optimizer": "AdamW",
    "schedule": "constant",
    "lr_gain": 3e-3,
    "lr_readout": 3e-3,
    "wd_readout": 1e-4,
    "lr_neuromod": 0.0,             # third factor is not trained in stage 0
    "grad_clip": 1.0,
    "truncate": 16,                 # steps of BPTT
    "plasticity": "bptt",           # bptt | three_factor
}

#: Pre-flight gates.  A run may not start unless every one of these holds.
GATES = {
    "h_peak_max": 50.0,             # bounded activity over the full sequence
    "dead_fraction_max": 0.5,
    "grad_finite": True,
    "pilot_updates": 2_000,
    "pilot_loss_drop": 0.05,        # the prediction loss must fall by this fraction
    "pilot_seconds_max": 3_600,
}
