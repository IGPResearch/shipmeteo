# -*- coding: utf-8 -*-
"""Parse NMEA logs from Alliance."""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import pynmea2
import xarray as xr
try:
    # If tqdm is installed
    try:
        # Check if it's Jupyter Notebook
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str.lower() and False:
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
    except NameError:
        from tqdm import tqdm
    from functools import partial
    pbar = partial(tqdm, leave=False, disable=DISABLE_TQDM)  # noqa
except ImportError:
    def pbar(obj, **tqdm_kw):
        """Empty progress bar."""
        return obj

# TODO: docstrings!!!


class AllianceComposite:
    TSTART = datetime(1970, 1, 1)

    def __init__(self, fname, date):
        assert isinstance(fname, Path), 'fname should be a Path object!'
        self.fname = fname
        self.date = date
        self.time_range = (pd.date_range(start=date,
                                         freq='S',
                                         periods=86400)
                           .to_series()
                           .to_frame(name='time'))

        self.data_d = dict()

    def read(self, msg_req_list):
        for msg_req in msg_req_list:
            self.data_d[msg_req['talker']] = dict()
            for fld in msg_req['fields']:
                self.data_d[msg_req['talker']][fld[0]] = []
        with self.fname.open('r') as f:
            for line in tqdm(f.readlines()):
                try:
                    msg = pynmea2.NMEASentence.parse(line)
                    for msg_req in msg_req_list:
                        if isinstance(msg, getattr(pynmea2.talker,
                                                   msg_req['talker'])):
                            for fld in msg_req['fields']:
                                assert isinstance(fld, tuple), 'Each field must be tuple!'
                                value = getattr(msg, fld[0])
                                if len(fld) == 2:
                                    # if the tuple contains two elements, assume the second one
                                    # is a function to convert the field value
                                    value = fld[1](value)
                                self.data_d[msg_req['talker']][fld[0]].append(value)

                except pynmea2.ParseError:
                    pass

        # Convert dictionaries of lists to dataframes and merge them together
        # using the time_range dataframe of 86400 seconds
        self.df = self.time_range
        for val in self.data_d.values():
            msg_df = pd.DataFrame(data=val)
            msg_df.rename(dict(datetime_str='datetime'), axis=1, inplace=True)
            msg_df['datetime'] = (msg_df['datetime']
                                  .astype(int)
                                  .apply(lambda x: self.TSTART + timedelta(seconds=x)))
            msg_df = msg_df.drop_duplicates('datetime').set_index('datetime')

            self.df = pd.merge(self.df, msg_df,
                               how='outer', left_index=True, right_index=True)

    def clean_up(self, drop_time=True,
                 mask_invalid_wind=True, mask_relative_wind=True,
                 convert_to_uv=True, convert_water_speed=True):
        """Clean up the dataframe."""
        if drop_time:
            self.df.drop('time', axis=1, inplace=True)
            self.df.index.rename('time', inplace=True)
        if mask_invalid_wind:
            self.df.wind_angle.values[self.df.status != 'A'] = np.nan
            self.df.wind_speed.values[self.df.status != 'A'] = np.nan
        if mask_relative_wind:
            self.df.wind_angle.values[self.df.reference != 'T'] = np.nan
            self.df.wind_speed.values[self.df.reference != 'T'] = np.nan
        if convert_to_uv:
            import metpy.calc as mcalc  # noqa
            import metpy.units as metunits  # noqa
            kt2mps = metunits.units('knots').to('m/s')
            u, v = mcalc.get_wind_components(self.df.wind_speed.values * kt2mps,
                                             self.df.wind_angle.values * metunits.units('degrees'))
            self.df['u_nmea'], self.df['v_nmea'] = u, v
            self.df.drop(labels=['wind_speed', 'wind_angle', 'reference', 'status'],
                         axis=1, inplace=True)
        if convert_water_speed:
            self.df.water_speed_knots *= kt2mps.m
            self.df.rename(dict(water_speed_knots='water_speed'),
                           axis=1, inplace=True)

    def process(self, msg_req_list, **kwargs):
        self.read(msg_req_list)
        self.clean_up(**kwargs)

    def average_over_time(self, freq):
        new_time = pd.date_range(start=self.date,
                                 end=self.date+timedelta(hours=23, minutes=59, seconds=59),
                                 freq=freq)
        tstep = new_time[1] - new_time[0]
        new_time += tstep
        return (self.df.groupby(np.arange(len(self.df)) // tstep.total_seconds())
                .mean()
                .set_index(new_time))
