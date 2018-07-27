# -*- coding: utf-8 -*-
"""Parse NMEA logs from Alliance."""
from datetime import datetime, timedelta
import metpy.calc as mcalc
import metpy.units as metunits
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
        if 'zmqshell' in ipy_str.lower():
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
    except NameError:
        from tqdm import tqdm
    from functools import partial
    pbar = partial(tqdm, leave=False)
except ImportError:
    def pbar(obj, **tqdm_kw):
        """Empty progress bar."""
        return obj

# TODO: docstrings!!!


class AllianceComposite:
    """
    Class for processing Alliance NMEA logs

    Useful for combining several messages together, aligning data along time axis,
    averaging over time, and saving to netCDF files.

    Typical workflow:
    >>> ac = AllianceComposite(Path('/path/to/file.log'), datetime.datetime(2018, 3, 1))
    >>> ac.process(msg_list)
    >>> averaged_dataset = ac.average_over_time(freq='1H')
    >>> ac.to_netcdf(averaged_dataset, path='myfile.nc')
    """
    TSTART = datetime(1970, 1, 1)

    def __init__(self, fname, date):
        """
        Initialise the AllianceComposite object

        Arguments
        ---------
        fname: pathlib.Path
            Path to the log file to process
        date: datetime.datetime
            Log file date
        """
        assert isinstance(fname, Path), 'fname should be a Path object!'
        self.fname = fname
        self.date = date
        # Assume 1 second frequency of the log files
        # TODO: make it flexible
        self.time_range = (pd.date_range(start=date,
                                         freq='S',
                                         periods=86400)
                           .to_series()
                           .to_frame(name='time'))
        self.data_d = dict()

    def read(self, msg_req_list):
        """
        Read the log file and store results as `.ds` attribute (xarray.Dataset).

        Arguments
        ---------
        msg_req_list: list
            List of dictionaries with fields to extract from NMEA messages
        """
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
        df = self.time_range
        for val in self.data_d.values():
            msg_df = pd.DataFrame(data=val)
            msg_df.rename(dict(datetime_str='datetime'), axis=1, inplace=True)
            msg_df['datetime'] = (msg_df['datetime']
                                  .astype(int)
                                  .apply(lambda x: self.TSTART + timedelta(seconds=x)))
            msg_df = msg_df.drop_duplicates('datetime').set_index('datetime')

            df = pd.merge(df, msg_df,
                          how='outer', left_index=True, right_index=True)
            self.ds = df.to_xarray()

    def clean_up(self, drop_time=True,
                 mask_invalid_wind=True, mask_relative_wind=True,
                 convert_to_uv=True, convert_to_mps=True):
        """
        Clean up the dataset and add essential attributes.

        Arguments
        ---------
        drop_time: bool
            Remove additional time variable (and leave only the index)
        mask_invalid_wind: bool
            Mask out wind speed and wind angle values if $INMWV Status is not "A"
        mask_relative_wind: bool
            Mask out wind speed and wind angle values if $INMWV Reference is not "T"
        convert_to_uv: bool
            Convert wind speed and wind angle to u- and v-components
        convert_to_mps: bool
            Convert units of wind speed and water speed from knots to m/s
        """
        self.ds.longitude.attrs['units'] = 'degrees_east'
        self.ds.latitude.attrs['units'] = 'degrees_north'

        if drop_time:
            self.ds = self.ds.drop(labels=['time'])
            self.ds.rename(dict(index='time'), inplace=True)
        if mask_invalid_wind:
            self.ds.wind_angle.values[self.ds.status != 'A'] = np.nan
            self.ds.wind_speed.values[self.ds.status != 'A'] = np.nan
            self.ds = self.ds.drop(labels=['status'])
        if mask_relative_wind:
            self.ds.wind_angle.values[self.ds.reference != 'T'] = np.nan
            self.ds.wind_speed.values[self.ds.reference != 'T'] = np.nan
            self.ds = self.ds.drop(labels=['reference'])
        if convert_to_mps:
            kt2mps = metunits.units('knots').to('m/s')
            self.ds['wind_speed'] *= kt2mps
            self.ds['water_speed_knots'] *= kt2mps
            self.ds.rename(dict(water_speed_knots='water_speed'), inplace=True)
        else:
            kt2mps = metunits.units('knots')

        if convert_to_uv:
            u, v = mcalc.get_wind_components(self.ds.wind_speed.values * kt2mps,
                                             self.ds.wind_angle.values * metunits.units('degrees'))
            self.ds = self.ds.drop(labels=['wind_speed', 'wind_angle'])
            self.ds = self.ds.assign(u=xr.Variable(dims='time', data=u,
                                                   attrs=dict(units='m s**-1',
                                                              long_name='U component of wind',
                                                              short_name='eastward_wind')),
                                     v=xr.Variable(dims='time', data=v,
                                                   attrs=dict(units='m s**-1',
                                                              long_name='V component of wind',
                                                              short_name='northward_wind')))

    def process(self, msg_req_list, **kwargs):
        """Shortcut for read() and clean_up() methods."""
        self.read(msg_req_list)
        self.clean_up(**kwargs)

    def average_over_time(self, freq, mark='end'):
        """
        Average the dataset over constant periods of time

        Arguments
        ---------
        freq: string or pandas.DateOffset
            Size of time chunks. E.g. 10T is 10 minutes
        mark: string, optional
            Time index mark. Can be one "start" or "end", e.g. the start 
            or the end of time chunks.

        Returns
        -------
        ave_ds: xarray.Dataset
            Dataset of averaged data
        """
        # create time index with the given frequency
        new_time = pd.date_range(start=self.date,
                                 end=self.date+timedelta(hours=23, minutes=59, seconds=59),
                                 freq=freq)
        if mark == 'end':
            tstep = new_time[1] - new_time[0]
            new_time += tstep
        # TODO: add "middle" option

        # save attributes before averaging
        _attrs = {k: self.ds[k].attrs for k in self.ds.data_vars}

        # average over time chunks
        ave_ds = (self.ds.groupby(xr.IndexVariable(dims='time',
                                                   data=np.arange(len(self.ds.time)) // tstep.total_seconds()))
                  .mean())

        # reset time index
        ave_ds['time'] = new_time

        # after groupby operation, the attributes are lost, so the saved are used
        for k in self.ds.data_vars:
            ave_ds[k].attrs.update(_attrs[k])

        return ave_ds

    @classmethod
    def to_netcdf(cls, ds, path, encoding=None, **kwargs):
        """Save xarray dataset to netCDF file and ensure that calendar uses the same start date."""
        if encoding is None:
            encoding = dict(time=dict(units=f'seconds since {cls.TSTART}',
                                      calendar='gregorian'))
        ds.to_netcdf(path=path, encoding=encoding, **kwargs)
