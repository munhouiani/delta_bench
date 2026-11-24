"""Open models on Delta-World, scored on acting and on predicting tool outputs.

    config    config.yaml
    serving   vLLM's flags for a model
    harness   the vf-eval command that runs a model on a split
    layout    which tokens of a rendered conversation are tool output
    acting    pass@1 from traces
    bpb       bits per byte from log-probs
    files     the data directory
    report    the tables and the chart
"""
