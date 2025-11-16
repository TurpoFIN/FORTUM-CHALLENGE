## Fingrid Open Data API (Physical Cross-Border Flows)

This module integrates Fingrid's Open Data API to fetch commercial transfer time-series on Finnish interconnectors.

### Datasets (measured physical flows)
- FI-EE measured flow (dataset 55) — Transmission of electricity FI–EE (15 min)
- FI-NO4 measured flow (dataset 57) — Transmission of electricity FI–NO (15 min)
- FI-SE1 measured flow (dataset 60) — Transmission of electricity FI–SE1 (15 min)
- FI-SE3 measured flow (dataset 61) — Transmission of electricity FI–SE3 (15 min)

Notes:
- Positive values denote export from Finland (+), negative values import to Finland (−)
- Resolution: 15-minute for all four datasets (older history may be hourly)

### Authentication
Get an API key from Fingrid developer portal and export it as:

```bash
export FINGRID_API_KEY="your_api_key_here"
```

Alternatively, pass `api_key` directly to `FingridApiClient`.

### Quick Start
See `fingrid_api/quick_start.py`:

```bash
python -m fingrid_api.quick_start --variable 55 --hours 24 --csv out_fi_ee_physical.csv
```

### Programmatic Usage

```python
from fingrid_api import FingridApiClient, get_fi_ee_transfer

client = FingridApiClient()  # reads FINGRID_API_KEY from env (or hardcoded)
df = get_fi_ee_transfer(client)  # last 48h by default (physical FI–EE)
print(df.head())
```

### Troubleshooting

#### Connection Issues

If you encounter connection errors (`getaddrinfo failed` or similar), test connectivity:

```bash
python -m fingrid_api.test_connectivity
```

#### Browser Works But Python Doesn't?

If you can access `data.fingrid.fi` in your browser but Python can't, your browser is likely using DNS over HTTPS (DoH) which bypasses system DNS.

**Quick Fix - Use IP Address in hosts file:**

1. Get the IP address from your browser:
   - Open Developer Tools (F12) → Network tab
   - Visit `https://data.fingrid.fi` 
   - Check the request details for the IP address

2. Run the helper script:
   ```bash
   python -m fingrid_api.fix_dns_issue
   ```

3. Or manually add to hosts file:
   - Open `C:\Windows\System32\drivers\etc\hosts` as Administrator
   - Add: `[IP_ADDRESS]    api.fingrid.fi`
   - Save and try again

**Other Solutions:**

1. **DNS Resolution Failure**: Change your DNS servers to 8.8.8.8 (Google) or 1.1.1.1 (Cloudflare)
2. **Firewall/Proxy**: Check if firewall or antivirus is blocking connections
3. **Network Configuration**: If behind a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` environment variables
4. **SSL Certificate Issues**: If in a corporate environment with proxy, try:
   ```python
   client = FingridApiClient(verify_ssl=False)  # Only for testing!
   ```

#### Programmatic Connectivity Test

```python
from fingrid_api import FingridApiClient

client = FingridApiClient()
results = client.test_connectivity()
print(results)
```

### Important Notes

#### Date Filtering
The API may not respect `start_time` and `end_time` parameters server-side. The client performs client-side filtering to ensure only data within the requested date range is returned. If the API doesn't have data for the requested dates, an empty DataFrame will be returned.

#### Data Availability
The API typically returns recent data. Historical data availability may vary by dataset. Check the [dataset pages](https://data.fingrid.fi) for information about data availability and update frequencies.

### License
Data license per Fingrid: Creative Commons Attribution (CC BY 4.0). Respect terms on each dataset page.


