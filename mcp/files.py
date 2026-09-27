"""Files MCP - File search, open, and management."""

from __future__ import annotations
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.files")


class FilesMCP(MCPBase):
    """File management MCP module."""

    name = "files"
    description = "File search, open, and management"
    commands = [
        "find file",
        "search file",
        "open file",
        "open folder",
        "show file",
        "recent files",
    ]

    def __init__(self):
        super().__init__()
        self._search_paths = [
            Path.home() / "Documents",
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path.home() / "Pictures",
            Path.home() / "Videos",
            Path.home() / "Music",
        ]
        self._recent_files: List[Dict] = []

    def initialize(self) -> bool:
        """Initialize files module."""
        self._initialized = True
        log.info("Files MCP initialized.")
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle file commands."""
        if command in ("find file", "search file"):
            if args:
                return self.find_file(args)
            return "What file are you looking for, sir?"

        if command in ("open file", "show file"):
            if args:
                return self.open_file(args)
            return "Which file should I open, sir?"

        if command in ("open folder",):
            if args:
                return self.open_folder(args)
            return "Which folder should I open, sir?"

        if command in ("recent files",):
            return self.list_recent_files()

        return None

    def find_file(self, query: str) -> str:
        """Search for files matching query."""
        query_lower = query.lower().strip('*')  # Remove wildcards for matching
        results = []
        
        for search_path in self._search_paths:
            if not search_path.exists():
                continue
            
            try:
                # Use glob if query has extension pattern
                if '.' in query_lower:
                    pattern = f"**/*{query_lower}"
                    for file_path in search_path.glob(pattern):
                        if file_path.is_file():
                            results.append({
                                "name": file_path.name,
                                "path": str(file_path),
                                "size": file_path.stat().st_size,
                            })
                            if len(results) >= 5:
                                break
                else:
                    # Search for files matching query
                    for file_path in search_path.rglob("*"):
                        if file_path.is_file() and query_lower in file_path.name.lower():
                            results.append({
                                "name": file_path.name,
                                "path": str(file_path),
                                "size": file_path.stat().st_size,
                            })
                            if len(results) >= 5:
                                break
            except (PermissionError, OSError):
                continue
        
        if not results:
            return f"No files found matching '{query}', sir."
        
        # Format results
        lines = []
        for i, f in enumerate(results[:3], 1):
            size = self._format_size(f["size"])
            lines.append(f"{i}. {f['name']} ({size})")
        
        # Store for potential opening
        self._search_results = results
        
        return f"Found {len(results)} files: " + "; ".join(lines)

    def open_file(self, query: str) -> str:
        """Open a file by name or index."""
        # Try to open by index from recent search
        if hasattr(self, '_search_results') and query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(self._search_results):
                file_path = self._search_results[idx]["path"]
                os.startfile(file_path)
                return f"Opening {self._search_results[idx]['name']}, sir."
        
        # Search for the file
        for search_path in self._search_paths:
            if not search_path.exists():
                continue
            
            try:
                for file_path in search_path.rglob("*"):
                    if file_path.is_file() and query.lower() in file_path.name.lower():
                        os.startfile(str(file_path))
                        return f"Opening {file_path.name}, sir."
            except (PermissionError, OSError):
                continue
        
        return f"I couldn't find a file matching '{query}', sir."

    def open_folder(self, folder_name: str) -> str:
        """Open a folder by name."""
        folder_name_lower = folder_name.lower()
        
        # Check common folders
        common_folders = {
            "documents": Path.home() / "Documents",
            "desktop": Path.home() / "Desktop",
            "downloads": Path.home() / "Downloads",
            "pictures": Path.home() / "Pictures",
            "videos": Path.home() / "Videos",
            "music": Path.home() / "Music",
            "this pc": Path("C:\\"),
            "my computer": Path("C:\\"),
        }
        
        if folder_name_lower in common_folders:
            folder = common_folders[folder_name_lower]
            if folder.exists():
                subprocess.Popen(["explorer", str(folder)])
                return f"Opening {folder_name} folder, sir."
        
        # Search for folder
        for search_path in self._search_paths:
            if not search_path.exists():
                continue
            
            try:
                for item in search_path.iterdir():
                    if item.is_dir() and folder_name_lower in item.name.lower():
                        subprocess.Popen(["explorer", str(item)])
                        return f"Opening {item.name} folder, sir."
            except (PermissionError, OSError):
                continue
        
        return f"I couldn't find a folder called '{folder_name}', sir."

    def list_recent_files(self) -> str:
        """List recently modified files."""
        recent_files = []
        
        for search_path in self._search_paths:
            if not search_path.exists():
                continue
            
            try:
                for file_path in search_path.iterdir():
                    if file_path.is_file():
                        mtime = file_path.stat().st_mtime
                        recent_files.append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "modified": mtime,
                        })
            except (PermissionError, OSError):
                continue
        
        # Sort by modification time
        recent_files.sort(key=lambda x: x["modified"], reverse=True)
        
        if not recent_files:
            return "No recent files found, sir."
        
        lines = []
        for f in recent_files[:5]:
            lines.append(f["name"])
        
        return "Recent files: " + ", ".join(lines)

    def _format_size(self, size_bytes: int) -> str:
        """Format file size to human readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "search_paths": len(self._search_paths),
        }
