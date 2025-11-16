"""
Temporary test script to try connecting with SSL verification disabled.
WARNING: This is only for testing in corporate environments with proxy/firewall.
"""
import datetime as dt
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

from .api_client import FingridApiClient

def main():
    print("Testing Fingrid API with SSL verification disabled...")
    print("=" * 60)
    
    # Try with SSL verification disabled (for corporate proxy environments)
    client = FingridApiClient(verify_ssl=False)
    
    # Test connectivity first
    print("\n1. Testing connectivity...")
    results = client.test_connectivity()
    print(f"   DNS Resolution: {results.get('dns_resolution', 'Not tested')}")
    print(f"   Connection Test: {results.get('connection_test', 'Not tested')}")
    
    if results.get('error') and 'DNS resolution failed' in results.get('error', ''):
        print("\n[ERROR] DNS resolution still failing.")
        print("The issue is DNS, not SSL. Try:")
        print("  1. Check if api.fingrid.fi resolves in your browser")
        print("  2. Check Windows hosts file: C:\\Windows\\System32\\drivers\\etc\\hosts")
        print("  3. Try using a different DNS server (8.8.8.8 or 1.1.1.1)")
        return
    
    # Try to fetch data
    print("\n2. Attempting to fetch data...")
    try:
        end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        start = end - dt.timedelta(hours=24)
        
        df = client.fetch_dataframe(140, start, end)  # FI-EE transfer
        print(f"   [SUCCESS] Retrieved {len(df)} rows")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"\n   First few rows:")
        print(df.head())
    except Exception as e:
        print(f"   [ERROR] Failed to fetch data: {e}")
        print("\n   This might be:")
        print("   - DNS resolution issue (api.fingrid.fi not resolving)")
        print("   - Network/firewall blocking")
        print("   - Missing API key (check FINGRID_API_KEY environment variable)")

if __name__ == "__main__":
    main()

