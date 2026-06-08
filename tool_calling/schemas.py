from typing import Any, Literal

from pydantic import BaseModel, Field


class WeatherToolArguments(BaseModel):
    """Defines and validates the location input the model sends to the weather tool."""

    location: str = Field(
        min_length=2,
        description="City, state, country, or postal code to get current weather for.",
    )


class FunctionSchema(BaseModel):
    """Represents the function metadata required by the tool-calling API."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolSchema(BaseModel):
    """Builds a validated function-tool definition before it is sent to the model."""

    type: Literal["function"] = "function"
    function: FunctionSchema


class LocationCoordinates(BaseModel):
    """Stores the resolved location and coordinates needed by the weather API."""

    name: str
    country: str | None = None
    latitude: float
    longitude: float


class CurrentWeather(BaseModel):
    """Validates the current conditions returned by the weather API."""

    time: str
    temperature: float
    temperature_unit: str
    apparent_temperature: float
    wind_speed: float
    wind_speed_unit: str
    weather_code: int


class WeatherResult(BaseModel):
    """Provides a stable, serializable result for returning weather data to the model."""

    location: LocationCoordinates
    current: CurrentWeather


WEATHER_TOOL_SCHEMA = ToolSchema(
    function=FunctionSchema(
        name="get_current_weather",
        description="Get the current weather for a location.",
        parameters=WeatherToolArguments.model_json_schema(),
    )
).model_dump()
