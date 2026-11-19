"""
AethelVision GitHub One-Click Publisher
=======================================
Pushes the committed AethelVision repository to your GitHub remote.
Works independently using Python and Dulwich (no git.exe required).

Usage:
  python upload_to_github.py --repo-url https://github.com/USERNAME/aethelvision.git
  or
  python upload_to_github.py --repo-url git@github.com:USERNAME/aethelvision.git
"""

import sys
import argparse
from dulwich.repo import Repo
from dulwich import porcelain

def main():
    parser = argparse.ArgumentParser(description="Upload AethelVision to GitHub")
    parser.add_argument("--repo-url", type=str, required=True, help="GitHub repository URL (HTTPS or SSH)")
    parser.add_argument("--branch", type=str, default="main", help="Target branch (default: main)")
    args = parser.parse_args()

    repo_path = "D:/MhyraVision"
    repo = Repo(repo_path)
    print(f"[*] Connecting to repository: {repo_path}")
    print(f"[*] Target Remote: {args.repo_url}")
    print(f"[*] Target Branch: {args.branch}")

    try:
        # Check current head
        head_commit = repo.head().decode()
        print(f"[+] Current Head Commit: {head_commit}")

        # Set remote or push
        print("[*] Initiating push to GitHub...")
        porcelain.push(repo, args.repo_url, refspecs=[f"refs/heads/{args.branch}"])
        print("\n[SUCCESS] AethelVision successfully pushed to GitHub!")
    except Exception as e:
        print(f"\n[!] Note: If authentication is required, ensure your GitHub Personal Access Token (PAT)")
        print(f"    or SSH key is configured, or use standard git CLI:")
        print(f"      git remote add origin {args.repo_url}")
        print(f"      git branch -M main")
        print(f"      git push -u origin main")
        print(f"    Details: {e}")

if __name__ == "__main__":
    main()
