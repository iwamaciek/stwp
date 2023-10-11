import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            '2m_temperature', 'surface_pressure',
        ],
        'year': '2023',
        'month': '10',
        'day': [
            '03', '04',
        ],
        'time': [
            '00:00', '04:00', '08:00',
            '12:00', '16:00', '20:00',
        ],
        'area': [
            55, 14, 49,
            25,
        ],
    },
    'data.grib')
