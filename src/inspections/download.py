"""Download a dated snapshot of the raw data from Analyze Boston.

    python -m inspections.download               # inspections + 311 (food & rodent types) + weather + census + earnings
    python -m inspections.download --only inspections
    python -m inspections.download --only 311 --years 2023 2024

Writes into data/raw/ and records each file's snapshot date, row count and SHA-256 in
data/raw/manifest.json, so a result can always be traced to the exact data it used.
Standard library only, so it runs before the heavier dependencies are installed.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

from inspections import config

PAGE = 30000   # CKAN datastore_search_sql caps rows per call


def api(action, **params):
    url = f'{config.CKAN_API}/{action}?{urllib.parse.urlencode(params)}'
    with urllib.request.urlopen(url, timeout=300) as r:
        body = json.load(r)
    if not body.get('success'):
        raise RuntimeError(f'{action} failed: {body.get("error")}')
    return body['result']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record(manifest, name, path, rows, source):
    manifest[name] = {'file': path.name, 'rows': rows, 'sha256': sha256(path),
                      'downloaded_utc': dt.datetime.now(dt.UTC).isoformat(timespec='seconds'),
                      'source': source}
    print(f'  {path.name}: {rows:,} rows')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def signed_s3_url(url):
    """Follow the portal's redirect to S3 ourselves.

    The portal redirects to a pre-signed S3 link written with an explicit ':443'. urllib then sends
    'Host: s3.amazonaws.com:443', which no longer matches the signed host, and S3 answers 403.
    Dropping the default port fixes the signature.
    """
    try:
        urllib.request.build_opener(_NoRedirect).open(url)
        return url                       # no redirect: the URL is directly downloadable
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        return e.headers['Location'].replace('.amazonaws.com:443/', '.amazonaws.com/')


def download_inspections(manifest, stamp):
    """The full CSV; its download URL changes daily, so look it up through the API."""
    url = api('resource_show', id=config.INSPECTIONS_RESOURCE)['url']
    api_total = api('datastore_search', resource_id=config.INSPECTIONS_RESOURCE, limit=0)['total']
    out = config.RAW / f'food_inspections_{stamp}.csv'
    print(f'inspections <- {url}')
    urllib.request.urlretrieve(signed_s3_url(url), out)
    with open(out, newline='', encoding='utf-8') as f:
        rows = sum(1 for _ in csv.reader(f)) - 1
    record(manifest, 'inspections', out, rows, url)
    manifest['inspections']['api_total'] = api_total   # independent count to detect truncated downloads


def download_311(manifest, stamp, years):
    """Only the food and rodent case types, pulled page by page with SQL."""
    types = ', '.join("'" + t.replace("'", "''") + "'" for t in config.SERVICE_311_TYPES)
    out = config.RAW / f'311_food_rodent_{stamp}.csv'
    cols = ['case_enquiry_id', 'open_dt', 'closed_dt', 'type', 'case_status',
            'neighborhood', 'latitude', 'longitude']
    rows = 0
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        for year in years:
            resource = config.SERVICE_311_RESOURCES[year]
            loaded = int(api('datastore_search_sql', sql=f'SELECT count(*) n FROM "{resource}"')['records'][0]['n'])
            before = rows
            if loaded:
                offset = 0
                while True:
                    sql = (f'SELECT {", ".join(cols)} FROM "{resource}" WHERE type IN ({types}) '
                           f'ORDER BY _id LIMIT {PAGE} OFFSET {offset}')
                    batch = api('datastore_search_sql', sql=sql)['records']
                    writer.writerows(batch)
                    rows += len(batch)
                    offset += PAGE
                    if len(batch) < PAGE:
                        break
            else:
                # some years (2011, 2014) are missing from the query database but the file exists:
                # stream the full CSV and keep only the case types we need
                url = signed_s3_url(api('resource_show', id=resource)['url'])
                with urllib.request.urlopen(url, timeout=600) as r:
                    lines = (line.decode('utf-8', errors='replace') for line in r)
                    for rec in csv.DictReader(lines):
                        if rec.get('type') in config.SERVICE_311_TYPES:
                            writer.writerow(rec)
                            rows += 1
            source = 'sql' if loaded else 'full file'
            print(f'  311 {year}: {rows - before:,} rows ({source})')
    record(manifest, '311_food_rodent', out, rows, f'{config.CKAN_API} SQL, types={config.SERVICE_311_TYPES}')


def download_weather(manifest, stamp):
    """Daily weather at Boston Logan from NOAA GHCN-Daily (public file, no token needed)."""
    out = config.RAW / f'weather_logan_{stamp}.csv'
    print(f'weather <- {config.WEATHER_URL}')
    urllib.request.urlretrieve(config.WEATHER_URL, out)
    with open(out, newline='', encoding='utf-8') as f:
        rows = sum(1 for _ in csv.reader(f)) - 1
    record(manifest, 'weather', out, rows, config.WEATHER_URL)


def download_census(manifest, stamp):
    """Median household income by ZIP code tabulation area (ACS 5-year); keeps only the ZCTA rows of the national file."""
    out = config.RAW / f'census_income_zcta_{stamp}.csv'
    print(f'census <- {config.ACS_INCOME_URL}')
    rows = 0
    with urllib.request.urlopen(config.ACS_INCOME_URL, timeout=600) as r, open(out, 'w', newline='', encoding='utf-8') as f:
        lines = (line.decode('utf-8') for line in r)
        reader = csv.DictReader(lines, delimiter='|')
        writer = csv.writer(f)
        writer.writerow(['zcta', 'median_household_income', 'moe'])
        for rec in reader:
            if rec['GEO_ID'].startswith('860Z200US'):
                writer.writerow([rec['GEO_ID'][-5:], rec['B19013_E001'], rec['B19013_M001']])
                rows += 1
    record(manifest, 'census_income', out, rows, config.ACS_INCOME_URL)


def download_earnings(manifest, stamp):
    """Pay of Inspectional Services staff from the city's Employee Earnings Report: title and pay only, no names (R18)."""
    out = config.RAW / f'earnings_isd_{config.EARNINGS_YEAR}_{stamp}.csv'
    cols = ['DEPARTMENT_NAME', 'TITLE', 'REGULAR', 'OVERTIME', 'TOTAL GROSS']
    sql = (f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{config.EARNINGS_RESOURCE}" '
           "WHERE \"DEPARTMENT_NAME\" ILIKE '%inspectional%'")
    records = api('datastore_search_sql', sql=sql)['records']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    record(manifest, 'earnings_isd', out, len(records), f'{config.CKAN_API} SQL, resource {config.EARNINGS_RESOURCE}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', choices=['inspections', '311', 'weather', 'census', 'earnings'])
    parser.add_argument('--years', type=int, nargs='+', default=sorted(config.SERVICE_311_RESOURCES))
    args = parser.parse_args()

    config.RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = config.RAW / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stamp = dt.date.today().strftime('%Y%m%d')

    if args.only in (None, 'inspections'):
        download_inspections(manifest, stamp)
    if args.only in (None, '311'):
        download_311(manifest, stamp, args.years)
    if args.only in (None, 'weather'):
        download_weather(manifest, stamp)
    if args.only in (None, 'census'):
        download_census(manifest, stamp)
    if args.only in (None, 'earnings'):
        download_earnings(manifest, stamp)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'manifest -> {manifest_path}')


if __name__ == '__main__':
    main()
