"""
FMI Open Data API Client
Base client for making requests to the Finnish Meteorological Institute's Open Data WFS API
"""

import requests
from typing import Dict, Any, Optional, List
import time
import xml.etree.ElementTree as ET
from datetime import datetime

from .config import (
    FMI_WFS_BASE_URL,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    STORED_QUERIES
)


class FMIClient:
    """
    Client for interacting with the FMI Open Data WFS API.
    
    The FMI API provides access to weather observations, forecasts, and other
    meteorological data from Finland through OGC Web Feature Service (WFS).
    
    No authentication required - the API is completely open.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the FMI API client.
        
        Args:
            base_url: Base URL for the WFS service (optional)
        """
        self.base_url = base_url or FMI_WFS_BASE_URL
        self.session = requests.Session()
        
    def _make_request(
        self,
        params: Dict[str, Any],
        timeout: int = DEFAULT_TIMEOUT,
        retry_count: int = 0
    ) -> ET.Element:
        """
        Make a GET request to the FMI WFS API.
        
        Args:
            params: Query parameters
            timeout: Request timeout in seconds
            retry_count: Current retry attempt number
            
        Returns:
            XML ElementTree root element
            
        Raises:
            requests.exceptions.RequestException: If request fails after retries
        """
        try:
            response = self.session.get(self.base_url, params=params, timeout=timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Check for WFS exceptions
            if 'ExceptionReport' in root.tag:
                exception_text = root.find('.//{http://www.opengis.net/ows/1.1}ExceptionText')
                if exception_text is not None:
                    raise ValueError(f"FMI API Error: {exception_text.text}")
                else:
                    raise ValueError("FMI API returned an exception")
            
            return root
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and retry_count < MAX_RETRIES:
                # Rate limited, wait and retry
                wait_time = 2 ** retry_count
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                return self._make_request(params, timeout, retry_count + 1)
            else:
                print(f"HTTP Error: {e.response.status_code}")
                print(f"Response: {e.response.text[:500]}")
                raise
                
        except requests.exceptions.RequestException as e:
            if retry_count < MAX_RETRIES:
                wait_time = 2 ** retry_count
                print(f"Request failed. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                return self._make_request(params, timeout, retry_count + 1)
            else:
                raise
    
    def get_capabilities(self) -> ET.Element:
        """
        Get WFS service capabilities.
        
        Returns:
            XML root element with capabilities information
        """
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetCapabilities'
        }
        return self._make_request(params)
    
    def get_feature(
        self,
        stored_query_id: str,
        **kwargs
    ) -> ET.Element:
        """
        Get features using a stored query.
        
        Args:
            stored_query_id: ID of the stored query to use
            **kwargs: Additional parameters (place, parameters, starttime, endtime, etc.)
            
        Returns:
            XML root element with feature data
            
        Example:
            >>> client = FMIClient()
            >>> root = client.get_feature(
            ...     'fmi::observations::weather::timevaluepair',
            ...     place='Helsinki',
            ...     starttime='2023-01-01T00:00:00Z',
            ...     endtime='2023-01-02T00:00:00Z'
            ... )
        """
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'getFeature',
            'storedquery_id': stored_query_id,
        }
        
        # Add additional parameters
        params.update(kwargs)
        
        return self._make_request(params)
    
    def get_observations(
        self,
        place: Optional[str] = None,
        fmisid: Optional[str] = None,
        bbox: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        timestep: Optional[int] = None
    ) -> ET.Element:
        """
        Get weather observations.
        
        Args:
            place: Place name (e.g., 'Helsinki')
            fmisid: Station ID(s), comma-separated
            bbox: Bounding box (minLon,minLat,maxLon,maxLat)
            start_time: Start time in ISO format (e.g., '2023-01-01T00:00:00Z')
            end_time: End time in ISO format
            parameters: List of parameters to retrieve
            timestep: Time step in minutes
            
        Returns:
            XML root element with observation data
        """
        kwargs = {}
        
        if place:
            kwargs['place'] = place
        if fmisid:
            kwargs['fmisid'] = fmisid
        if bbox:
            kwargs['bbox'] = bbox
        if start_time:
            kwargs['starttime'] = start_time
        if end_time:
            kwargs['endtime'] = end_time
        if parameters:
            kwargs['parameters'] = ','.join(parameters)
        if timestep:
            kwargs['timestep'] = str(timestep)
        
        return self.get_feature(
            STORED_QUERIES['observations_realtime'],
            **kwargs
        )
    
    def get_forecast(
        self,
        place: Optional[str] = None,
        latlon: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        timestep: Optional[int] = None
    ) -> ET.Element:
        """
        Get weather forecast.
        
        Args:
            place: Place name (e.g., 'Helsinki')
            latlon: Latitude,longitude (e.g., '60.17,24.94')
            start_time: Start time in ISO format
            end_time: End time in ISO format
            parameters: List of parameters to retrieve
            timestep: Time step in minutes
            
        Returns:
            XML root element with forecast data
        """
        kwargs = {}
        
        if place:
            kwargs['place'] = place
        if latlon:
            kwargs['latlon'] = latlon
        if start_time:
            kwargs['starttime'] = start_time
        if end_time:
            kwargs['endtime'] = end_time
        if parameters:
            kwargs['parameters'] = ','.join(parameters)
        if timestep:
            kwargs['timestep'] = str(timestep)
        
        # Use edited forecast by default (official FMI forecast)
        # Falls back to HARMONIE if edited not available
        stored_query = STORED_QUERIES.get('forecast_edited', STORED_QUERIES['forecast_harmonie'])
        
        return self.get_feature(
            stored_query,
            **kwargs
        )
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


