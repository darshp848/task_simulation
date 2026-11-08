"""Hill Climb harness: a supervisor for Dream-1 long-horizon runs.

It never touches a motor. It only drives Dream's station agent over HTTP
(see DESIGN.md), so every cap, clamp and home move in Dream's control loop stays in force.
"""

__version__ = "0.2.0"
