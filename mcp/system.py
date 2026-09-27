"""System MCP - CPU, RAM, battery, disk, network monitoring."""

from __future__ import annotations
import logging
import subprocess
import time
from typing import Any, Dict, Optional

import psutil

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.system")


def _format_duration(seconds: int) -> str:
    """Format a second count as HH:MM:SS for the HUD."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class SystemMCP(MCPBase):
    """System monitoring MCP module."""

    name = "system"
    description = "System monitoring (CPU, RAM, battery, disk, network)"
    commands = [
        "system info",
        "system status",
        "computer status",
        "cpu",
        "ram",
        "memory",
        "battery",
        "disk",
        "storage",
        "network",
        "running apps",
        "what's running",
        "open applications",
    ]

    def __init__(self):
        super().__init__()
        self._last_cpu = 0.0
        self._last_net = {"bytes_sent": 0, "bytes_recv": 0}

    def initialize(self) -> bool:
        """Initialize system monitoring."""
        self._initialized = True
        log.info("System MCP initialized.")
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle system commands."""
        if command in ("system info", "system status", "computer status"):
            return self.get_system_info()

        if command in ("cpu",):
            return self._get_cpu_info()

        if command in ("ram", "memory"):
            return self._get_ram_info()

        if command == "battery":
            return self._get_battery_info()

        if command in ("disk", "storage"):
            return self._get_disk_info()

        if command == "network":
            return self._get_network_info()

        if command in ("running apps", "what's running", "open applications"):
            return self.get_running_apps()

        return None

    def get_system_info(self) -> str:
        """Get comprehensive system info."""
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\")

        parts = [
            f"CPU at {cpu:.1f} percent.",
            f"RAM: {ram.used / (1024**3):.1f} of {ram.total / (1024**3):.1f} GB used.",
            f"Disk: {disk.used / (1024**3):.0f} of {disk.total / (1024**3):.0f} GB used.",
        ]

        try:
            bat = psutil.sensors_battery()
            if bat:
                s = "charging" if bat.power_plugged else "on battery"
                parts.append(f"Battery at {bat.percent:.0f} percent, {s}.")
        except Exception:
            pass

        return " ".join(parts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics for UI."""
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\")
        net = psutil.net_io_counters()

        metrics = {
            "cpu": round(cpu, 1),
            "ram_used": round(ram.used / (1024**3), 1),
            "ram_total": round(ram.total / (1024**3), 1),
            "ram_pct": round(ram.percent, 1),
            "disk_pct": round(disk.percent, 1),
            "disk_used": round(disk.used / (1024**3), 0),
            "disk_total": round(disk.total / (1024**3), 0),
            "net_sent": net.bytes_sent,
            "net_recv": net.bytes_recv,
        }

        try:
            bat = psutil.sensors_battery()
            if bat:
                metrics["battery"] = round(bat.percent, 0)
                metrics["charging"] = bat.power_plugged
        except Exception:
            pass

        return metrics

    def get_ui_metrics(self) -> Dict[str, Any]:
        """Get metrics in the exact shape the HUD expects."""
        metrics = self.get_metrics()
        uptime_s = int(time.time() - psutil.boot_time())
        return {
            "type": "metrics",
            "cpu_pct": metrics.get("cpu", 0.0),
            "ram_pct": metrics.get("ram_pct", 0.0),
            "ram_used": metrics.get("ram_used", 0.0),
            "ram_total": metrics.get("ram_total", 0.0),
            "disk_pct": metrics.get("disk_pct", 0.0),
            "disk_used": metrics.get("disk_used", 0),
            "disk_total": metrics.get("disk_total", 0),
            "battery": metrics.get("battery"),
            "charging": metrics.get("charging"),
            "net_sent": metrics.get("net_sent", 0),
            "net_recv": metrics.get("net_recv", 0),
            "uptime": _format_duration(uptime_s),
            "process_count": len(psutil.pids()),
        }

    def _get_cpu_info(self) -> str:
        """Get CPU information."""
        cpu = psutil.cpu_percent(interval=0.5)
        freq = psutil.cpu_freq()
        cores = psutil.cpu_count()
        parts = [f"CPU at {cpu:.1f} percent.", f"{cores} cores available."]
        if freq:
            parts.append(f"Running at {freq.current:.0f} MHz.")
        return " ".join(parts)

    def _get_ram_info(self) -> str:
        """Get RAM information."""
        ram = psutil.virtual_memory()
        return f"RAM: {ram.used / (1024**3):.1f} of {ram.total / (1024**3):.1f} GB used ({ram.percent:.1f}%)."

    def _get_battery_info(self) -> str:
        """Get battery information."""
        try:
            bat = psutil.sensors_battery()
            if bat:
                s = "charging" if bat.power_plugged else "on battery"
                return f"Battery at {bat.percent:.0f} percent, {s}."
            return "No battery detected, sir."
        except Exception:
            return "Unable to read battery status, sir."

    def _get_disk_info(self) -> str:
        """Get disk information."""
        disk = psutil.disk_usage("C:\\")
        return f"Disk: {disk.used / (1024**3):.0f} of {disk.total / (1024**3):.0f} GB used ({disk.percent:.1f}%)."

    def _get_network_info(self) -> str:
        """Get network information."""
        net = psutil.net_io_counters()
        return f"Network: Sent {net.bytes_sent / (1024**2):.1f} MB, Received {net.bytes_recv / (1024**2):.1f} MB."

    def get_running_apps(self) -> str:
        """Get list of running applications."""
        apps = set()
        for proc in psutil.process_iter(["name"]):
            try:
                name = proc.info["name"]
                if name and not name.startswith("System") and name not in (
                    "svchost.exe", "csrss.exe", "lsass.exe", "services.exe",
                    "wininit.exe", "smss.exe", "dwm.exe", "sihost.exe",
                ):
                    apps.add(name.replace(".exe", ""))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not apps:
            return "I couldn't enumerate running applications."
        return f"Running applications: {', '.join(sorted(apps)[:15])}, sir."

    def execute_command(self, cmd: str) -> str:
        """Execute a system command."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=15
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                output = result.stderr.strip() or "Command failed."
            if len(output) > 500:
                output = output[:500] + "..."
            return output or "Command executed successfully."
        except subprocess.TimeoutExpired:
            return "Command timed out."
        except Exception as e:
            return f"Error: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "cpu": psutil.cpu_percent(interval=0.1),
            "ram_pct": psutil.virtual_memory().percent,
        }
