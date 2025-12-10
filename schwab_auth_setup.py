#!/usr/bin/env python3
"""
Schwab API Initial Authentication Setup

Run this script ONCE to get your initial tokens.
After that, the app will automatically refresh them.

Usage:
    python schwab_auth_setup.py

Requirements:
    pip install schwab-py

You need:
    - SCHWAB_APP_KEY: Your app key from developer.schwab.com
    - SCHWAB_APP_SECRET: Your app secret from developer.schwab.com
"""

import os
import json
import webbrowser
import http.server
import socketserver
import urllib.parse
import base64
import requests
from datetime import datetime, timedelta

# Configuration
APP_KEY = os.environ.get("SCHWAB_APP_KEY", "")
APP_SECRET = os.environ.get("SCHWAB_APP_SECRET", "")
CALLBACK_URL = "https://127.0.0.1:8182"
TOKEN_PATH = "schwab_tokens.json"


def get_authorization_url():
    """Generate the authorization URL"""
    params = {
        "client_id": APP_KEY,
        "redirect_uri": CALLBACK_URL,
        "response_type": "code",
    }
    base_url = "https://api.schwabapi.com/v1/oauth/authorize"
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(auth_code):
    """Exchange authorization code for access and refresh tokens"""
    auth_string = f"{APP_KEY}:{APP_SECRET}"
    auth_bytes = base64.b64encode(auth_string.encode()).decode()
    
    response = requests.post(
        "https://api.schwabapi.com/v1/oauth/token",
        headers={
            "Authorization": f"Basic {auth_bytes}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": CALLBACK_URL
        }
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def save_tokens(tokens):
    """Save tokens to file"""
    token_data = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "token_expiry": (datetime.now() + timedelta(seconds=tokens.get("expires_in", 1800))).isoformat()
    }
    
    with open(TOKEN_PATH, "w") as f:
        json.dump(token_data, f, indent=2)
    
    print(f"\n✅ Tokens saved to {TOKEN_PATH}")


def main():
    print("=" * 60)
    print("Schwab API Authentication Setup")
    print("=" * 60)
    
    if not APP_KEY or not APP_SECRET:
        print("\n❌ Error: Missing credentials!")
        print("\nSet these environment variables:")
        print("  export SCHWAB_APP_KEY='your-app-key'")
        print("  export SCHWAB_APP_SECRET='your-app-secret'")
        print("\nGet these from: https://developer.schwab.com/")
        return
    
    print(f"\nApp Key: {APP_KEY[:8]}...")
    print(f"Callback: {CALLBACK_URL}")
    
    # Generate auth URL
    auth_url = get_authorization_url()
    
    print("\n" + "=" * 60)
    print("STEP 1: Open this URL in your browser:")
    print("=" * 60)
    print(f"\n{auth_url}\n")
    
    # Try to open browser automatically
    try:
        webbrowser.open(auth_url)
        print("(Browser should open automatically)")
    except:
        print("(Please copy and paste the URL above into your browser)")
    
    print("\n" + "=" * 60)
    print("STEP 2: After logging in, you'll be redirected to a URL like:")
    print("  https://127.0.0.1:8182/?code=XXXX&session=YYYY")
    print("\nPaste the FULL redirect URL below:")
    print("=" * 60)
    
    redirect_url = input("\nPaste URL here: ").strip()
    
    # Extract authorization code
    try:
        parsed = urllib.parse.urlparse(redirect_url)
        params = urllib.parse.parse_qs(parsed.query)
        auth_code = params.get("code", [None])[0]
        
        if not auth_code:
            print("\n❌ Could not find authorization code in URL")
            return
        
        print(f"\n✓ Found authorization code: {auth_code[:20]}...")
        
    except Exception as e:
        print(f"\n❌ Error parsing URL: {e}")
        return
    
    # Exchange for tokens
    print("\nExchanging code for tokens...")
    tokens = exchange_code_for_tokens(auth_code)
    
    if tokens:
        save_tokens(tokens)
        print("\n" + "=" * 60)
        print("SUCCESS! You can now run the Options Screener with Schwab data.")
        print("=" * 60)
        print("\nMake sure to set these env vars when running the app:")
        print(f"  export SCHWAB_APP_KEY='{APP_KEY}'")
        print(f"  export SCHWAB_APP_SECRET='{APP_SECRET}'")
        print(f"  export SCHWAB_TOKEN_PATH='{TOKEN_PATH}'")
    else:
        print("\n❌ Failed to get tokens. Please try again.")


if __name__ == "__main__":
    main()