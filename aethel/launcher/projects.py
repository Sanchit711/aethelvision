"""AETHEL Enterprise AI Vision Platform — First-Class Project Manager.

Manages enterprise project lifecycles: Create, Open, Clone, Archive, Delete,
Recent Projects, Favorite Projects, and workspace directory structure.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProjectManager:
    """Enterprise Project Lifecycle Orchestrator."""

    SUBDIRS = ["datasets", "models", "experiments", "reports", "exports", "plugins", "configs", "logs"]

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or Path(".").resolve()
        self.projects_dir = self.root_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.active_project_path = self.projects_dir / "default_project"
        self.ensure_project_structure(self.active_project_path, name="AETHEL Core Vision")

    def ensure_project_structure(self, proj_dir: Path, name: str = "New Project") -> Dict[str, Any]:
        """Create standard enterprise project folder layout and manifest."""
        proj_dir.mkdir(parents=True, exist_ok=True)
        for sdir in self.SUBDIRS:
            (proj_dir / sdir).mkdir(parents=True, exist_ok=True)

        manifest_p = proj_dir / "project.json"
        if not manifest_p.exists():
            manifest = {
                "name": name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_opened": time.strftime("%Y-%m-%d %H:%M:%S"),
                "favorite": True,
                "version": "1.0.0",
                "framework": "AETHEL v1.0.0-ENT",
                "description": "Enterprise Computer Vision Intelligence Project",
            }
            with open(manifest_p, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            return manifest
        else:
            try:
                with open(manifest_p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"name": proj_dir.name}

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all available project workspaces."""
        projects = []
        # Check active project + projects subfolder
        dirs = [self.active_project_path] + list(self.projects_dir.iterdir())
        seen = set()

        for d in dirs:
            if d.is_dir() and str(d.resolve()) not in seen:
                seen.add(str(d.resolve()))
                mf_file = d / "project.json"
                name = d.name
                last_opened = "Unknown"
                fav = False
                if mf_file.exists():
                    try:
                        with open(mf_file, "r", encoding="utf-8") as f:
                            mdata = json.load(f)
                        name = mdata.get("name", d.name)
                        last_opened = mdata.get("last_opened", "Unknown")
                        fav = mdata.get("favorite", False)
                    except Exception:
                        pass
                
                # Count assets
                models_cnt = len(list((d / "models").glob("*.*"))) if (d / "models").exists() else 0
                datasets_cnt = len(list((d / "datasets").glob("*.*"))) if (d / "datasets").exists() else 0

                projects.append({
                    "name": name,
                    "folder": d.name,
                    "path": str(d),
                    "last_opened": last_opened,
                    "favorite": fav,
                    "models_count": models_cnt,
                    "datasets_count": datasets_cnt,
                    "status": "ACTIVE" if str(d.resolve()) == str(self.active_project_path.resolve()) else "OPEN",
                })
        return projects

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new project workspace."""
        safe_folder = "".join([c if c.isalnum() else "_" for c in name.lower()]).strip("_")
        proj_p = self.projects_dir / safe_folder
        proj_p.mkdir(parents=True, exist_ok=True)
        manifest = self.ensure_project_structure(proj_p, name=name)
        if description:
            manifest["description"] = description
            with open(proj_p / "project.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        self.active_project_path = proj_p
        return manifest

    def open_project(self, folder_name: str) -> Optional[Dict[str, Any]]:
        """Open an existing project and update last_opened timestamp."""
        proj_p = self.projects_dir / folder_name
        if not proj_p.exists():
            proj_p = self.root_dir / folder_name
        if proj_p.exists() and proj_p.is_dir():
            self.active_project_path = proj_p
            mf_file = proj_p / "project.json"
            manifest = self.ensure_project_structure(proj_p, name=proj_p.name)
            manifest["last_opened"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(mf_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            return manifest
        return None

    def clone_project(self, source_folder: str, new_name: str) -> Optional[Dict[str, Any]]:
        """Clone an existing project workspace into a new project."""
        src_p = self.projects_dir / source_folder
        if not src_p.exists():
            src_p = self.root_dir / source_folder
        if src_p.exists():
            safe_folder = "".join([c if c.isalnum() else "_" for c in new_name.lower()]).strip("_")
            dst_p = self.projects_dir / safe_folder
            shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
            manifest = self.ensure_project_structure(dst_p, name=new_name)
            manifest["cloned_from"] = source_folder
            manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(dst_p / "project.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            return manifest
        return None

    def archive_project(self, folder_name: str) -> bool:
        """Archive project workspace into a timestamped zip archive."""
        src_p = self.projects_dir / folder_name
        if src_p.exists():
            archive_dir = self.root_dir / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_name = f"{folder_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.make_archive(str(archive_dir / archive_name), "zip", src_p)
            return True
        return False


__all__ = ["ProjectManager"]
