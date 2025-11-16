import os
import datetime as dt
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


ISOTime = Union[str, dt.datetime]


class FingridApiClient:
    """
    Thin client for Fingrid Open Data API.
    - Auth via API key header 'x-api-key'
    - Time is UTC, ISO 8601
    - Base endpoint for dataset data: /datasets/{id}/data
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://data.fingrid.fi/api",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("FINGRID_API_KEY") or ""
        self.timeout_seconds = timeout_seconds
        self.verify_ssl = verify_ssl

        session = requests.Session()
        # Use system proxy settings (requests does this by default via environment variables)
        # Also check for Windows proxy settings
        proxies = {}
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            value = os.getenv(key)
            if value:
                proxies[key.lower()] = value
        if proxies:
            session.proxies.update(proxies)
        
        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        self._session = session

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    @staticmethod
    def _iso8601(t: ISOTime) -> str:
        if isinstance(t, str):
            return t
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc).isoformat()

    def build_events_url(self, variable_id: int) -> str:
        return f"{self.base_url}/datasets/{variable_id}/data"

    def test_connectivity(self) -> Dict[str, Any]:
        """
        Test connectivity to the Fingrid API.
        Returns a dictionary with test results.
        """
        import socket
        results = {
            "base_url": self.base_url,
            "dns_resolution": None,
            "connection_test": None,
            "error": None,
        }
        
        # Extract hostname from base_url
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            
            # Test DNS resolution
            try:
                ip_address = socket.gethostbyname(hostname)
                results["dns_resolution"] = f"Success: {hostname} -> {ip_address}"
            except socket.gaierror as e:
                results["dns_resolution"] = f"Failed: {str(e)}"
                results["error"] = (
                    "DNS resolution failed. This usually means:\n"
                    "  1. No internet connection\n"
                    "  2. DNS server issues (try using 8.8.8.8 or 1.1.1.1)\n"
                    "  3. Firewall/proxy blocking DNS queries\n"
                    "  4. VPN or network configuration problems"
                )
                return results
            
            # Test connection
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((hostname, port))
                sock.close()
                if result == 0:
                    results["connection_test"] = f"Success: Can connect to {hostname}:{port}"
                else:
                    results["connection_test"] = f"Failed: Cannot connect to {hostname}:{port}"
                    results["error"] = "Connection test failed. Check firewall/proxy settings."
            except Exception as e:
                results["connection_test"] = f"Error: {str(e)}"
                results["error"] = f"Connection test error: {str(e)}"
                
        except Exception as e:
            results["error"] = f"Test setup error: {str(e)}"
        
        return results

    def fetch_variable_events(
        self,
        variable_id: int,
        start_time: ISOTime,
        end_time: ISOTime,
        max_pages: Optional[int] = None,
        per_page: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch raw dataset events as JSON list with keys: 'value', 'startTime', 'endTime'
        Time arguments can be ISO strings or datetime; interpreted as UTC.
        Returns a list of event dictionaries from the API's 'data' field.
        
        Args:
            variable_id: Dataset ID
            start_time: Start time (ISO string or datetime)
            end_time: End time (ISO string or datetime)
            max_pages: Maximum number of pages to fetch (None = all pages)
            per_page: Number of items per page (default: 1000, max recommended)
        """
        url = self.build_events_url(variable_id)
        all_data: List[Dict[str, Any]] = []
        current_page = 1
        
        try:
            while True:
                params = {
                    # Fingrid new API expects camelCase parameter names
                    "startTime": self._iso8601(start_time),
                    "endTime": self._iso8601(end_time),
                    "page": current_page,
                    "pageSize": per_page,
                }
                
                resp = self._session.get(
                    url, 
                    headers=self._headers(), 
                    params=params, 
                    timeout=self.timeout_seconds,
                    verify=self.verify_ssl
                )
                resp.raise_for_status()
                response_data: Dict[str, Any] = resp.json()
                
                # Extract data from response
                if isinstance(response_data, dict) and "data" in response_data:
                    page_data: List[Dict[str, Any]] = response_data["data"]
                    all_data.extend(page_data)
                    
                    # Check pagination
                    pagination = response_data.get("pagination", {})
                    total_pages = pagination.get("lastPage", 1)
                    has_next = pagination.get("nextPage") is not None
                    
                    # Stop if no more pages or reached max_pages
                    if not has_next or (max_pages and current_page >= max_pages):
                        break
                    
                    current_page += 1
                elif isinstance(response_data, list):
                    # Fallback: if API returns list directly
                    all_data.extend(response_data)
                    break
                else:
                    break
                    
            return all_data
        except requests.exceptions.ConnectionError as e:
            # Run connectivity diagnostics
            diagnostics = self.test_connectivity()
            error_msg = (
                f"Failed to connect to Fingrid API at {self.base_url}.\n"
                f"Error: {str(e)}\n\n"
            )
            if diagnostics.get("error"):
                error_msg += f"Diagnostics:\n{diagnostics['error']}\n\n"
            if diagnostics.get("dns_resolution"):
                error_msg += f"DNS Test: {diagnostics['dns_resolution']}\n"
            if diagnostics.get("connection_test"):
                error_msg += f"Connection Test: {diagnostics['connection_test']}\n"
            error_msg += (
                f"\nRequest URL: {url}\n"
                f"Parameters: startTime={params['startTime']}, endTime={params['endTime']}\n\n"
                "Troubleshooting steps:\n"
                "  1. Check your internet connection\n"
                "  2. Try changing DNS servers (8.8.8.8 or 1.1.1.1)\n"
                "  3. Check firewall/antivirus settings\n"
                "  4. If behind a proxy, configure HTTP_PROXY/HTTPS_PROXY environment variables"
            )
            raise requests.exceptions.ConnectionError(error_msg) from e
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"Request to Fingrid API failed.\n"
                f"URL: {url}\n"
                f"Error: {str(e)}\n"
                f"Parameters: startTime={params['startTime']}, endTime={params['endTime']}"
            )
            raise requests.exceptions.RequestException(error_msg) from e

    def fetch_dataframe(
        self,
        variable_id: int,
        start_time: ISOTime,
        end_time: ISOTime,
        tz: str = "UTC",
        max_pages: Optional[int] = None,
        per_page: int = 1000,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: ['value'] and DatetimeIndex at the event start time.
        
        Note: The API may not respect date parameters server-side, so filtering is performed
        client-side. If the API doesn't have data for the requested date range, an empty
        DataFrame will be returned.
        
        Args:
            variable_id: Dataset ID
            start_time: Start time (ISO string or datetime)
            end_time: End time (ISO string or datetime)
            tz: Timezone for output index (default: UTC)
            max_pages: Maximum number of pages to fetch (None = all pages)
            per_page: Number of items per page (default: 1000)
        """
        events = self.fetch_variable_events(variable_id, start_time, end_time, max_pages=max_pages, per_page=per_page)
        if not events:
            return pd.DataFrame(columns=["value"]).astype({"value": "float64"})
        
        # API uses camelCase: startTime, endTime
        # Convert to a list of dicts with consistent keys
        processed_events = []
        for event in events:
            processed_events.append({
                "value": event.get("value"),
                "start_time": event.get("startTime") or event.get("start_time"),
                "end_time": event.get("endTime") or event.get("end_time"),
            })
        
        df = pd.DataFrame(processed_events)
        # Prefer 'start_time' as index; Fingrid times are ISO8601
        idx = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df = df.assign(value=pd.to_numeric(df["value"], errors="coerce"))
        df.index = idx
        
        # Filter by date range (API doesn't seem to respect date parameters, so filter client-side)
        original_len = len(df)
        if start_time or end_time:
            start_dt = pd.to_datetime(self._iso8601(start_time), utc=True) if start_time else None
            end_dt = pd.to_datetime(self._iso8601(end_time), utc=True) if end_time else None
            
            if start_dt is not None:
                df = df[df.index >= start_dt]
            if end_dt is not None:
                df = df[df.index <= end_dt]
            
            # Warn if filtering removed all data
            if original_len > 0 and len(df) == 0:
                import warnings
                data_start = idx.min() if len(idx) > 0 else None
                data_end = idx.max() if len(idx) > 0 else None
                warnings.warn(
                    f"No data found in requested range ({start_dt} to {end_dt}). "
                    f"API returned data from {data_start} to {data_end}. "
                    f"Try adjusting your date range to match available data.",
                    UserWarning
                )
        
        df = df[["value"]].sort_index()
        if tz and tz.upper() != "UTC":
            df = df.tz_convert(tz)
        return df


