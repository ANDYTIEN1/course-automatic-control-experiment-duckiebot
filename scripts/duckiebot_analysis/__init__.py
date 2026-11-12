# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""duckiebot_analysis

Library code of the wheel-video analysis scripts in scripts/. A script run as
``python scripts/<name>.py`` finds this package because Python puts the
script's folder on the import path.

Subpackages group code by its role in the analysis pipeline:
    tracking/       video tracking of the wheel marker, hub, and sync LED
    data/           frame tables, folder layout, robot logs
    analysis/       synchronisation, processing, quality, model fits
    models/         motor models
    visualization/  plotting and figure generation
"""

__version__ = "0.1.0"
