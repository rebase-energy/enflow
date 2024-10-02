import numpy as np
import pandas as pd
import pvlib

from pvlib import pvsystem, modelchain, location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']


def PV_AC_Output(
        PV_rated_power = 250,
        Inverter_AC_power=None,
        surface_tilt = 30,
        surface_azimuth = 180,
        latitude = 59.3293,
        longitude = 18.0686,
        start_day = '2021-01-01',
        end_day = '2021-01-02',
        GHI=None,
        DNI=None,
        DHI=None,
        clear_sky_condition = False
):
    start_date = pd.Timestamp(start_day)  # Start date and time
    end_date = pd.Timestamp(end_day)    # End date and time
    time_vector = pd.date_range(start=start_date, end=end_date, freq='h')
    time_vector=time_vector[:-1]

    if Inverter_AC_power is None:
        Inverter_AC_power=PV_rated_power

    module_parameters = {
        'pdc0': PV_rated_power, 
        'gamma_pdc': -0.004
    }

    inverter_parameters = {
        'pdc0': Inverter_AC_power,
        'eta_inv_nom': 0.96
    }

    pvwatts_system = pvsystem.PVSystem(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        module_parameters=module_parameters,
        inverter_parameters=inverter_parameters,
        temperature_model_parameters=temperature_model_parameters)


    loc = location.Location(latitude, longitude)
    if clear_sky_condition:
        clearsky = loc.get_clearsky(times=time_vector)
        GHI = clearsky['ghi']
        DNI = clearsky['dni']
        DHI = clearsky['dhi']
    else:
        if GHI is None:
            print('Provide at least GHI values or select clear sky conditions')
        else:
            solar_position = loc.get_solarposition(times=time_vector)
            if DNI is None:
                DNI = pvlib.irradiance.disc(GHI, solar_position['zenith'].values, time_vector)
                DNI = DNI['dni'].values
            if DHI is None:
                DHI = GHI - DNI * np.cos(np.radians(solar_position['zenith'].values))

    mc = modelchain.ModelChain(pvwatts_system, loc, aoi_model='no_loss', spectral_model='no_loss')


    weather = pd.DataFrame(np.column_stack((GHI, DNI, DHI)),
                        columns=['ghi', "dni", "dhi"],
                        index=time_vector)

    mc.run_model(weather)

    output_power = mc.results.ac

    return output_power.values


def PV_AC_Output_OLD(
        PV_rated_power = 250,
        Inverter_AC_power=None,
        surface_tilt = 30,
        surface_azimuth = 180,
        latitude = 59.3293,
        longitude = 18.0686,
        GHI=None,
        DNI=None,
        DHI=None,
        start_day = '2021-01-01',
        end_day = '2021-01-02',
        tz = 'Europe/Stockholm',
        clear_sky_condition = True
):
    start_date = pd.Timestamp(start_day, tz=tz)  # Start date and time
    end_date = pd.Timestamp(end_day, tz=tz)    # End date and time
    time_vector = pd.date_range(start=start_date, end=end_date, freq='h', tz=tz)
    time_vector=time_vector[:-1]

    if Inverter_AC_power is None:
        Inverter_AC_power=PV_rated_power

    module_parameters = {
        'pdc0': PV_rated_power, 
        'gamma_pdc': -0.004
    }

    inverter_parameters = {
        'pdc0': Inverter_AC_power,
        'eta_inv_nom': 0.96
    }

    pvwatts_system = pvsystem.PVSystem(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        module_parameters=module_parameters,
        inverter_parameters=inverter_parameters,
        temperature_model_parameters=temperature_model_parameters)


    loc = location.Location(latitude, longitude)
    if clear_sky_condition:
        clearsky = loc.get_clearsky(times=time_vector)
        #clearsky = clearsky[:-1]
        GHI = clearsky['ghi']
        DNI = clearsky['dni']
        DHI = clearsky['dhi']

    mc = modelchain.ModelChain(pvwatts_system, loc, aoi_model='no_loss', spectral_model='no_loss')


    weather = pd.DataFrame(np.column_stack((GHI, DNI, DHI)),
                        columns=['ghi', "dni", "dhi"],
                        index=time_vector)

    mc.run_model(weather)

    output_power = mc.results.ac

    return output_power.values
