"""Weather MCP - Weather information and forecasts."""

from __future__ import annotations
import logging
import os
from typing import Any, Dict, Optional

import requests

from .base import MCPBase

log = logging.getLogger("jarvis.mcp.weather")


class WeatherMCP(MCPBase):
    """Weather information MCP module."""

    name = "weather"
    description = "Weather information and forecasts"
    commands = [
        "weather",
        "check weather",
        "temperature",
        "forecast",
        "today's weather",
        "tomorrow weather",
    ]

    def __init__(self):
        super().__init__()
        self._default_city = os.getenv("DEFAULT_CITY", "London")

    def initialize(self) -> bool:
        """Initialize weather module."""
        self._initialized = True
        log.info("Weather MCP initialized.")
        return True

    def handle(self, command: str, args: str) -> Optional[str]:
        """Handle weather commands."""
        if command in ("weather", "check weather", "temperature", "today's weather"):
            return self.get_weather(args)
        
        if command in ("forecast", "tomorrow weather"):
            return self.get_forecast(args)
        
        return None

    def get_weather(self, city: str = "") -> str:
        """Get current weather for a city."""
        city = city.strip() or self._default_city
        
        try:
            # Quick attempt with short timeout
            r = requests.get(
                f"https://wttr.in/{city}?format=%C+%t",
                headers={"User-Agent": "curl/7.81"},
                timeout=4,
            )
            if r.status_code == 200:
                raw = r.text.strip().encode('ascii', 'ignore').decode('ascii').strip()
                if raw:
                    return f"Weather in {city}: {raw}"
            
            # Fallback: return cached or generic response
            return "Weather service is slow right now, sir. Try again in a moment."
        except requests.exceptions.Timeout:
            return "Weather service timed out, sir. Your network may be slow."
        except Exception as e:
            log.error("Weather error: %s", e)
            return "Unable to fetch weather data, sir."

    def _format_weather(self, city: str, raw: str) -> str:
        """Format weather data nicely."""
        # Raw format: "Condition Temp Humidity Wind"
        parts = raw.split()
        if len(parts) >= 4:
            condition = parts[0]
            temp = parts[1]
            humidity = parts[2]
            wind = parts[3]
            return f"Weather in {city}: {condition}, {temp}, humidity {humidity}, wind {wind}"
        return f"Weather in {city}: {raw}"

    def get_forecast(self, city: str = "", days: int = 3) -> str:
        """Get weather forecast."""
        city = city.strip() or self._default_city
        
        try:
            r = requests.get(
                f"https://wttr.in/{city}?format=j1",
                headers={"User-Agent": "curl/7.81"},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                current = data.get("current_condition", [{}])[0]
                forecasts = data.get("weather", [])
                
                # Current weather
                current_desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
                current_temp = current.get("temp_C", "--")
                
                lines = [f"Current: {current_desc}, {current_temp} C"]
                
                # Forecast
                for i, day in enumerate(forecasts[:days]):
                    date = day.get("date", "")
                    max_temp = day.get("maxtempC", "--")
                    min_temp = day.get("mintempC", "--")
                    desc = day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "Unknown")
                    lines.append(f"{date}: {desc}, {min_temp} to {max_temp} C")
                
                return " | ".join(lines)
            
            return f"Unable to fetch forecast for {city}, sir."
        except Exception as e:
            log.error("Forecast error: %s", e)
            return "Unable to fetch forecast data, sir."

    def get_weather_dict(self, city: str = "") -> Dict[str, Any]:
        """Get weather as dictionary."""
        city = city.strip() or self._default_city
        try:
            r = requests.get(
                f"https://wttr.in/{city}?format=j1",
                headers={"User-Agent": "curl/7.81"},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                current = data.get("current_condition", [{}])[0]
                return {
                    "temp_c": current.get("temp_C", "--"),
                    "temp_f": current.get("temp_F", "--"),
                    "desc": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
                    "humidity": current.get("humidity", "--"),
                    "wind_kmph": current.get("windspeedKmph", "--"),
                    "city": city,
                }
        except Exception:
            pass
        return {"temp_c": "--", "desc": "Unknown", "city": city}

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "default_city": self._default_city,
        }
