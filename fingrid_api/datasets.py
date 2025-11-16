import datetime as dt
from typing import Optional

import pandas as pd

from .api_client import FingridApiClient

# Dataset IDs for measured physical cross-border flows (15-min unless noted)
# Positive = export from Finland, Negative = import to Finland
# Sources:
# - Transmission of electricity between Finland and Estonia - measured every 15 minutes (id: 55)
# - Transmission of electricity between Finland and Norway - measured every 15 minutes (id: 57)
# - Transmission of electricity between Finland and Northern Sweden (SE1) - measured every 15 minutes (id: 60)
# - Transmission of electricity between Finland and Central Sweden (SE3) - measured every 15 minutes (id: 61)
VAR_ID_FI_EE = 55    # FI-EE physical (measured) flow
VAR_ID_FI_NO4 = 57   # FI-NO (NO4) physical (measured) flow
VAR_ID_FI_SE1 = 60   # FI-SE1 physical (measured) flow
VAR_ID_FI_SE3 = 61   # FI-SE3 physical (measured) flow


def _default_window(hours: int = 48) -> tuple[dt.datetime, dt.datetime]:
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = end - dt.timedelta(hours=hours)
    return start, end


def get_fi_ee_transfer(
    client: FingridApiClient,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    if start_time is None or end_time is None:
        start_time, end_time = _default_window()
    return client.fetch_dataframe(VAR_ID_FI_EE, start_time, end_time, tz=tz)


def get_fi_no4_transfer(
    client: FingridApiClient,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    if start_time is None or end_time is None:
        start_time, end_time = _default_window()
    return client.fetch_dataframe(VAR_ID_FI_NO4, start_time, end_time, tz=tz)


def get_fi_se1_transfer(
    client: FingridApiClient,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    if start_time is None or end_time is None:
        start_time, end_time = _default_window()
    return client.fetch_dataframe(VAR_ID_FI_SE1, start_time, end_time, tz=tz)


def get_fi_se3_transfer(
    client: FingridApiClient,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    if start_time is None or end_time is None:
        start_time, end_time = _default_window()
    return client.fetch_dataframe(VAR_ID_FI_SE3, start_time, end_time, tz=tz)


