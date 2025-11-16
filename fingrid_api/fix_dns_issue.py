"""
Helper script to diagnose and work around DNS issues with Fingrid API.
If your browser can access api.fingrid.fi but Python can't, this helps.
"""
import socket
import sys

def get_ip_from_browser():
    """
    Instructions for getting IP address from browser.
    """
    print("=" * 60)
    print("HOW TO GET THE IP ADDRESS FROM YOUR BROWSER:")
    print("=" * 60)
    print("\n1. Open your browser and visit: https://data.fingrid.fi")
    print("2. Open Developer Tools (F12)")
    print("3. Go to Network tab")
    print("4. Refresh the page")
    print("5. Click on any request to data.fingrid.fi")
    print("6. Look for 'Remote Address' or check the request details")
    print("7. Note the IP address (e.g., 123.45.67.89)")
    print("\nAlternatively:")
    print("- In Chrome: Right-click page > Inspect > Network tab")
    print("- Look for the IP in the request headers or connection info")
    print("\n" + "=" * 60)

def test_dns_resolution():
    """Test DNS resolution"""
    print("\nTesting DNS resolution...")
    try:
        ip = socket.gethostbyname('data.fingrid.fi')
        print(f"[OK] DNS resolution successful: data.fingrid.fi -> {ip}")
        return ip
    except socket.gaierror as e:
        print(f"[FAIL] DNS resolution failed: {e}")
        return None

def test_with_ip(ip_address):
    """Test if we can connect using IP address directly"""
    print(f"\nTesting connection to IP: {ip_address}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip_address, 443))
        sock.close()
        if result == 0:
            print(f"[OK] Can connect to {ip_address}:443")
            return True
        else:
            print(f"[FAIL] Cannot connect to {ip_address}:443")
            return False
    except Exception as e:
        print(f"[FAIL] Connection test error: {e}")
        return False

def create_hosts_file_entry(ip_address):
    """Instructions for adding hosts file entry"""
    hosts_file = r"C:\Windows\System32\drivers\etc\hosts"
    print("\n" + "=" * 60)
    print("SOLUTION: Add to Windows hosts file")
    print("=" * 60)
    print(f"\n1. Open Notepad as Administrator")
    print(f"2. Open file: {hosts_file}")
    print(f"3. Add this line at the end:")
    print(f"   {ip_address}    data.fingrid.fi")
    print(f"4. Save the file")
    print(f"5. Try running your script again")
    print("\n" + "=" * 60)

def main():
    print("Fingrid API DNS Resolution Helper")
    print("=" * 60)
    
    # Test current DNS
    ip = test_dns_resolution()
    
    if ip:
        # DNS works!
        print("\n[OK] DNS resolution is working!")
        print("The issue might be SSL/proxy related.")
        print("Try running: py -m fingrid_api.test_with_ssl_bypass")
        return
    
    # DNS doesn't work
    print("\n[FAIL] DNS resolution is NOT working.")
    print("\nSince your browser can access it, your browser is likely using")
    print("DNS over HTTPS (DoH) or a different DNS resolver.")
    print("\nSOLUTIONS:")
    print("\nOption 1: Get IP from browser and add to hosts file")
    get_ip_from_browser()
    
    print("\nOption 2: Change Windows DNS settings")
    print("1. Open Network Settings")
    print("2. Change adapter options")
    print("3. Right-click your connection > Properties")
    print("4. IPv4 > Properties")
    print("5. Use DNS: 8.8.8.8 (Google) or 1.1.1.1 (Cloudflare)")
    print("6. Click OK and restart")
    
    print("\nOption 3: Use a different DNS resolver in Python")
    print("This requires installing dnspython: pip install dnspython")
    
    # If user provides IP, test it
    if len(sys.argv) > 1:
        ip_from_user = sys.argv[1]
        print(f"\nTesting provided IP: {ip_from_user}")
        if test_with_ip(ip_from_user):
            create_hosts_file_entry(ip_from_user)
        else:
            print("[FAIL] Cannot connect to that IP address")

if __name__ == "__main__":
    main()

