"""Execute 15 returned actions before replanning with an H16 model."""
_base_ = './droid.py'
themis = dict(evaluation=dict(execute_horizon=15))
