"""
AethelVision PyPI One-Click Publisher
=====================================
Uploads the verified aethelvision-1.0.0 distribution packages to PyPI.org.

Usage:
  python publish_to_pypi.py --token pypi-AgEIcHlwaS5vcmc...
  or
  python publish_to_pypi.py --testpypi --token pypi-AgEI...
  or simply:
  python publish_to_pypi.py (it will securely prompt for your PyPI token)
"""

import sys
import getpass
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Upload AethelVision to PyPI")
    parser.add_argument("--token", type=str, help="PyPI API Token (starts with pypi-)")
    parser.add_argument("--testpypi", action="store_true", help="Upload to TestPyPI instead of production PyPI")
    args = parser.parse_args()

    token = args.token
    if not token:
        print("="*65)
        print(" AethelVision PyPI Deployment Manager")
        print("="*65)
        target = "TestPyPI" if args.testpypi else "PyPI.org (Production)"
        print(f"Target: {target}")
        print("Please enter your PyPI API token (e.g. pypi-AgEI...):")
        token = getpass.getpass("Token: ").strip()

    if not token:
        print("[!] Error: No token provided. Aborting upload.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "twine", "upload",
        "--username", "__token__",
        "--password", token,
        "D:/MhyraVision/dist/aethelvision-1.0.0*"
    ]

    if args.testpypi:
        cmd.extend(["--repository", "testpypi"])

    print("\n[*] Uploading distribution packages to PyPI...")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n[SUCCESS] AethelVision v1.0.0 successfully published to PyPI!")
        if not args.testpypi:
            print("View live at: https://pypi.org/project/aethelvision/")
            print("Community install: pip install aethelvision")
    else:
        print(f"\n[!] Upload failed with return code {res.returncode}. Please verify your PyPI API token permissions.")

if __name__ == "__main__":
    main()
