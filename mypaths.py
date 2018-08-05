# -*- coding: utf-8 -*-
"""
Paths to data
"""
import os
from pathlib import Path

# Root of the current repository
curdir = Path('.').absolute().parent
sample_dir = curdir/'data'

# External data directories
# igp_data_dir = Path('/media')/os.getenv('USER')/'Elements'/'IGP'/'data'
igp_data_dir = Path('~/IGP/data').expanduser()
wpk_dir = igp_data_dir/'weatherpacks'
nmea_dir = igp_data_dir/'nmea_logs'
masin_dir = igp_data_dir/'masin'
dundee_dir = igp_data_dir/'dundee'
ostia_dir = igp_data_dir/'ostia'
amsr2_dir = igp_data_dir/'amsr2'

# Output directories
plotdir = curdir/'figures'
