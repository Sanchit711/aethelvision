"""
Automated GitHub Device Authentication & Repository Publisher
============================================================
1. Launches GitHub Device Flow authorization.
2. Extracts one-time activation code and prompts user.
3. Awaits browser authorization.
4. Initializes remote, creates repo 'aethelvision', and pushes code automatically.
"""

import os
import sys
import time
import subprocess

GH_BIN = r"D:\tools\gh\bin\gh.exe"
GIT_BIN = r"D:\tools\git\cmd\git.exe"
REPO_DIR = r"D:\MhyraVision"

# Ensure git and gh are in environment PATH
git_dir = r"D:\tools\git\cmd"
gh_dir = r"D:\tools\gh\bin"
os.environ["PATH"] = git_dir + os.pathsep + gh_dir + os.pathsep + os.environ.get("PATH", "")

def run_cmd(cmd, cwd=REPO_DIR):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)

def main():
    print("="*65)
    print(" AethelVision Automated GitHub Device Setup & Publisher")
    print("="*65)
    sys.stdout.flush()

    # 1. Check if already authenticated
    res = run_cmd([GH_BIN, "auth", "status"])
    if res.returncode == 0:
        print("[+] Device is ALREADY authenticated with GitHub!")
    else:
        print("[*] Initiating GitHub Device Flow Authentication...")
        sys.stdout.flush()

        proc = subprocess.Popen(
            [GH_BIN, "auth", "login", "--web", "--git-protocol", "https", "--skip-ssh-key"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_DIR
        )

        code_found = False
        user_code = ""
        url = "https://github.com/login/device"

        # Read stderr for device code (gh outputs device code to stderr)
        start_t = time.time()
        while time.time() - start_t < 15:
            line = proc.stderr.readline()
            if line:
                line_str = line.strip()
                print(f"[GH] {line_str}")
                sys.stdout.flush()
                if "code (" in line_str:
                    try:
                        user_code = line_str.split("code (")[1].split(")")[0]
                        code_found = True
                    except:
                        pass
                elif "code:" in line_str.lower():
                    try:
                        user_code = line_str.split(":")[-1].strip()
                        code_found = True
                    except:
                        pass
                if "github.com/login/device" in line_str:
                    url = "https://github.com/login/device"
            if code_found:
                break
            time.sleep(0.3)

        if code_found and user_code:
            print("\n" + "#"*65)
            print(" ACTION REQUIRED: AUTHORIZE YOUR DEVICE")
            print(f" 1. Go to: {url}")
            print(f" 2. Enter One-Time Code: {user_code}")
            print("#"*65 + "\n")
            sys.stdout.flush()

            # Also save to a state file so it can be read from anywhere
            with open(os.path.join(REPO_DIR, "github_auth_code.txt"), "w") as f:
                f.write(f"URL: {url}\nCODE: {user_code}\n")

        print("[*] Waiting for browser approval...")
        sys.stdout.flush()

        retcode = proc.wait(timeout=300)
        if retcode != 0:
            print(f"[!] Authentication process exited with code {retcode}.")
            sys.exit(1)
        print("\n[SUCCESS] Device authenticated successfully!")
        sys.stdout.flush()

    # 2. Configure Git credential helper
    run_cmd([GH_BIN, "auth", "setup-git"])

    # 3. Get authenticated username
    user_res = run_cmd([GH_BIN, "api", "user", "-q", ".login"])
    username = user_res.stdout.strip() if user_res.returncode == 0 else "user"
    print(f"[+] Authenticated as GitHub user: @{username}")
    sys.stdout.flush()

    # 4. Check if repo already exists on GitHub
    repo_name = "aethelvision"
    check_repo = run_cmd([GH_BIN, "repo", "view", f"{username}/{repo_name}"])

    if check_repo.returncode == 0:
        print(f"[*] Repository {username}/{repo_name} already exists on GitHub. Pushing latest commits...")
    else:
        print(f"[*] Creating new public repository: {username}/{repo_name}...")
        create_res = run_cmd([
            GH_BIN, "repo", "create", repo_name,
            "--public",
            f"--description=AethelVision: Sovereign Clean-Room Edge AI Multi-Modal Computer Vision Platform (2D BB, OBB, Instance Segmentation, Pose Estimation)",
        ])
        print(create_res.stdout)
        if create_res.stderr:
            print(create_res.stderr)

    # 5. Configure remote and push branch main
    remote_check = run_cmd([GIT_BIN, "remote", "get-url", "origin"])
    expected_remote = f"https://github.com/{username}/{repo_name}.git"
    if remote_check.returncode != 0:
        run_cmd([GIT_BIN, "remote", "add", "origin", expected_remote])
    else:
        run_cmd([GIT_BIN, "remote", "set-url", "origin", expected_remote])

    print(f"[*] Pushing local branch 'main' to origin ({expected_remote})...")
    sys.stdout.flush()
    push_res = run_cmd([GIT_BIN, "push", "-u", "origin", "main"])
    print(push_res.stdout)
    if push_res.stderr:
        print(push_res.stderr)

    print(f"\n[DONE] AethelVision is officially live on GitHub at: https://github.com/{username}/{repo_name}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
