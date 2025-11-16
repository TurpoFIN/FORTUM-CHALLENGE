"""
Diagnostic script to test connectivity to Fingrid API.
Run this to troubleshoot connection issues.
"""
import sys
from .api_client import FingridApiClient


def main():
    print("Testing connectivity to Fingrid API...")
    print("=" * 60)
    
    client = FingridApiClient()
    results = client.test_connectivity()
    
    print(f"Base URL: {results['base_url']}")
    print(f"Expected: https://data.fingrid.fi/api")
    print(f"\nDNS Resolution: {results.get('dns_resolution', 'Not tested')}")
    print(f"Connection Test: {results.get('connection_test', 'Not tested')}")
    
    if results.get('error'):
        print(f"\n[ERROR] Error detected:")
        print(results['error'])
        print("\n" + "=" * 60)
        print("Troubleshooting suggestions:")
        print("1. Check your internet connection")
        print("2. Try changing DNS servers:")
        print("   - Windows: Network Settings > Change adapter options >")
        print("     Right-click your connection > Properties > IPv4 >")
        print("     Use: 8.8.8.8 (Google) or 1.1.1.1 (Cloudflare)")
        print("3. Check if firewall/antivirus is blocking the connection")
        print("4. If behind a proxy, set HTTP_PROXY/HTTPS_PROXY environment variables")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Connectivity tests passed!")
        print("The API endpoint should be reachable.")
        sys.exit(0)


if __name__ == "__main__":
    main()

