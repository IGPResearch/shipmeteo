# coding: utf-8
"""
Convert .mat file with CTD data to netCDF format
"""
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from scipy.io import loadmat

import mypaths


# Define functions to convert matlab datetimes to python
@np.vectorize
def matlab2datetime_dt(x):
    return np.datetime64((datetime.fromordinal(int(x))
                          + timedelta(days=x % 1)
                          - timedelta(days=366))
                         .strftime('%Y-%m-%dT%H:%M:%S.%f'))


def matlab2datetime_np(matlab_ordinal_time_arr):
    origin = np.datetime64('0000-01-01', 'D') - np.timedelta64(1, 'D')
    delta = np.timedelta64(1, 'us') * 86400e6
    return matlab_ordinal_time_arr * delta + origin


# Full path to the matlab file
fname = mypaths.igp_data_dir / 'ALL0118_uctd' / 'uctd_1second.mat'

# Load matlab structure from the file and select the only group 'sec'
ctd_mat = loadmat(fname)['sec'][0]


# Convert array of matlab time to numpy datetime objects
time_arr = matlab2datetime_np(ctd_mat['time'][0][0])

# Check if two functions give same results
# time_arr2 = matlab2datetime_dt(ctd_mat['time'][0][0])
# print(abs((time_arr - time_arr2)).max())
# print(4 * 1e-6 / 86400)

# Create an xarray dataset
field_names = [i for i in ctd_mat.dtype.fields.keys() if i != 'time']
# print(field_names)


vrbl_attrs = {
    't1': {
        'name': 'sbe45_thermosalinograph_temperature',
        'attrs': {
            'standard_name': 'sea_water_temperature',
            'long_name': 'SBE45 Micro TSG Thermosalinograph Temperature',
            'units': 'degree_celsius'
        }
    },
    't2': {
        'name': 'sbe38_bow_temperature',
        'attrs': {
            'standard_name': 'sea_water_temperature',
            'long_name': 'SBE38 bow temperature',
            'units': 'degree_celsius'
        }
    },
    'c1': {
        'name': 'sbe45_thermosalinograph_conductivity',
        'attrs': {
            'standard_name': 'sea_water_electrical_conductivity',
            'long_name': 'SBE45 Micro TSG Thermosalinograph Conductivity',
            'units': 'S m-1'
        }
    },
    's1': {
        'name': 'sbe45_thermosalinograph_salinity',
        'attrs': {
            'standard_name': 'sea_water_salinity',
            'long_name': 'SBE45 Micro TSG Thermosalinograph Salinity',
            'units': 'g kg-1'
        }
    },
    'sv': {
        'name': 'sv',
        'attrs': {
            'standard_name': 'unknown',
        }
    },
    'lon': {
        'name': 'longitude',
        'attrs': {
            'standard_name': 'longitude',
            'long_name': 'NMEA longitude',
            'units': 'degree_east'
        }
    },
    'lat': {
        'name': 'latitude',
        'attrs': {
            'standard_name': 'latitude',
            'long_name': 'NMEA latitude',
            'units': 'degree_north'
        }
    },
}


# TODO: add license

# In[14]:


global_attrs = {
    'title': 'Temperature, Salinity and Conductivity measurements from ALL0118 Underway CTD Survey',
    'summary': """Shipboard underway CTD data were collected for the majority of the ALL0118 survey.
A shipboard dedicated computer running SeaSave (Seabird Electronics software) recorded ocean temperature and salinity through a water intake located at 2.5 m depth on the bow of the ship.
An SBE38 temperature sensor measured water temperature close to the seawater intake.
Further down the seawater intake line an SBE45 Micro TSG Thermosalinograph measured conductivity and temperature.
Throughout the cruise, the SBE45 temperature consistently measured approx. 0.7 K warmer than the SBE38 temperature sensor closer to the bow intake.
As indicated in the SBE XML configuration files saved with the raw data, no calibration coefficients were applied to the sensors.
As such, a post-cruise calibration of the underway CTD temperature and salinity using the calibrated CTD station data will be assessed.""",
    'keywords': 'ctd,alliance,sea temperature,salinity,conductivity,igp',
    'conventions': 'CF-1.7,ACDD-1.3',
    'history': f'Originally created at 2018-05-02 21:33:32 GMT by Leah McRaven (Woods Hole Oceanographic Institution); Converted to netCDF at {datetime.utcnow():%Y-%m-%d %H:%M:%S} GMT by Denis Sergeev (University of East Anglia)',
    'source': 'SBE38 temperature sensor, SBE45 micro TSG thermosalinograph, NMEA on board NRV Alliance',
    'processing_level': 'no calibration performed',
    'comment':  """Despite efforts to keep the underway system continuously logging, there are periods throughout the cruise where there is no available data.
This was primarily the case during periods of rough weather as bubbles can disrupt the flow through sensors, causing them
to shut off.

The bow temperature recording from the SBE38 is considered the closest value to the true ocean temperature.""",
    'date_created': f'{datetime.utcnow():%Y-%m-%d %H:%M:%S} GMT',
    'institution': 'University of East Anglia, Woods Hole Oceanographic Institution',
    'creator_name': 'Denis Sergeev',
    'creator_email': 'd.sergeev@uea.ac.uk',
    'creator_institution': 'University of East Anglia',
    'project': 'The Iceland Greenland Seas Project (IGP)',
    'time_coverage_start': f'{time_arr[0].astype(datetime):%Y-%m-%dT%H:%M:%SZ}',
    'time_coverage_end': f'{time_arr[-1].astype(datetime):%Y-%m-%dT%H:%M:%SZ}',
    'time_coverage_resolution': 's',
    'coverage_content_type': 'physicalMeasurement',
}


# Common time coordinate
coord_info = dict(dims=('time',), coords=dict(time=xr.IndexVariable(dims='time',
                                                                    data=time_arr,
                                                                    attrs=dict(standard_name='time'))))


# Create xarray dataset and append dataarrays to it
ds = xr.Dataset(attrs=global_attrs)
for field_name in field_names:
    ds = ds.assign(**{vrbl_attrs[field_name]['name']: xr.DataArray(ctd_mat[field_name][0][0], **coord_info, **vrbl_attrs[field_name])})


# Save the dataset to a netCDF file
ds.to_netcdf(mypaths.igp_data_dir / 'igp_all0118_uctd.nc',
             encoding=dict(time=dict(units=f'seconds since 1970-01-01T00:00:00.0',
                                     calendar='gregorian')))
