"""Project-wide constants: paths, data source IDs and the frozen train/validation/test split.

Change split dates only through a decision recorded in PLAN.md.
"""
from pathlib import Path

# paths, resolved from this file so code runs from any working directory
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'

# Analyze Boston (CKAN) endpoints and resource IDs, verified 2026-11-10
CKAN_API = 'https://data.boston.gov/api/3/action'
INSPECTIONS_RESOURCE = '4582bec6-2b4f-4f9e-bc55-cbaa73117f4c'
LICENSES_RESOURCE = 'f1e13724-284d-478c-b8bc-ef042aa5b70b'
SERVICE_311_RESOURCES = {
    2011: '94b499d9-712a-4d2a-b790-7ceec5c9c4b1',
    2012: '382e10d9-1864-40ba-bef6-4eea3c75463c',
    2013: '407c5cd0-f764-4a41-adf8-054ff535049e',
    2014: 'bdae89c8-d4ce-40e9-a6e1-a5203953a2e0',
    2015: 'c9509ab4-6f6d-4b97-979a-0cf2a10c922b',
    2016: 'b7ea6b1b-3ca4-4c5b-9713-6dc1db52379a',
    2017: '30022137-709d-465e-baae-ca155b51927d',
    2018: '2be28d90-3a90-4af1-a3f6-f28c1e25880a',
    2019: 'ea2e4696-4a2d-429c-9807-d02eb92e0222',
    2020: '6ff6a6fd-3141-4440-a880-6f60a37fe789',
    2021: 'f53ebccd-bc61-49f9-83db-625f209c95f5',
    2022: '81a7b022-f8fc-4da5-80e4-b160058ca207',
    2023: 'e6013a93-1321-4f2a-bf91-8d8a02f1e62f',
    2024: 'dff4d804-5031-443a-8409-8344efd0e5c8',
    2025: '9d7c2214-4709-478a-a2e8-fb2020a5bb94',
    2026: '1a0b420d-99f1-4887-9851-990b2a5a6e17',
}
# 311 case types used for the complaint-leakage flag and the neighborhood features
SERVICE_311_TYPES = [
    'Unsanitary Conditions - Food',
    'Unsanitary Conditions - Establishment',
    'Unsanitary Conditions - Employees',
    'Food Alert - Confirmed',
    'Food Alert - Unconfirmed',
    'Rodent Activity',
]

# NOAA GHCN-Daily, Boston Logan airport
WEATHER_URL = ('https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/'
               'access/USW00014739.csv')

# temporal split (cycle-opening inspection dates, local time, inclusive).
# Labels start in 2020: Boston switched to the 2013 FDA Food Code during 2019, and only from 2020 on
# does viol_level '***' mean exactly a Priority (P) violation (notebook 01). Older visits still feed
# history features.
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-12-31'
VALID_START = '2024-01-01'
VALID_END = '2024-12-31'
TEST_START = '2025-01-01'
TEST_END = '2026-10-31'
PANDEMIC = ('2020-03-01', '2021-06-30')   # flagged, not dropped; sensitivity run without it

SEED = 15071
