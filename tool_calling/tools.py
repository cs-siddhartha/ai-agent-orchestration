import httpx

from tool_calling.schemas import (
    CurrentWeather,
    LocationCoordinates,
    WeatherResult,
    WeatherToolArguments,
)

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10.0


def resolve_location(location: str) -> LocationCoordinates:
    """Converts a human-readable location into coordinates for the weather request."""

    response = httpx.get(
        GEOCODING_API_URL,
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"Could not find a location matching {location!r}.")

    match = results[0]
    return LocationCoordinates(
        name=match["name"],
        country=match.get("country"),
        latitude=match["latitude"],
        longitude=match["longitude"],
    )


def get_current_weather(arguments: WeatherToolArguments) -> WeatherResult:
    """Fetches and validates current weather for model-provided tool arguments."""

    location = resolve_location(arguments.location)
    response = httpx.get(
        WEATHER_API_URL,
        params={
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": (
                "temperature_2m,apparent_temperature,"
                "weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    current = payload["current"]
    units = payload["current_units"]

    return WeatherResult(
        location=location,
        current=CurrentWeather(
            time=current["time"],
            temperature=current["temperature_2m"],
            temperature_unit=units["temperature_2m"],
            apparent_temperature=current["apparent_temperature"],
            wind_speed=current["wind_speed_10m"],
            wind_speed_unit=units["wind_speed_10m"],
            weather_code=current["weather_code"],
        ),
    )
