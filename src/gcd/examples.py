"""Pydantic models used by the tests and the benchmark (each paired with task prompts)."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str
    age: int
    email: str
    is_student: bool


class Product(BaseModel):
    id: int
    name: str
    price: float
    tags: list[str] = Field(max_length=4)
    in_stock: bool


class Condition(str, Enum):
    sunny = "sunny"
    cloudy = "cloudy"
    rain = "rain"
    snow = "snow"


class Weather(BaseModel):
    city: str
    temperature_c: float
    condition: Condition
    humidity_pct: int


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class Customer(BaseModel):
    name: str
    address: Address


class LineItem(BaseModel):
    sku: str
    quantity: int


class Order(BaseModel):
    order_id: str
    customer: Customer
    items: list[LineItem] = Field(min_length=1)
    express: bool


class Event(BaseModel):
    title: str
    date: str
    attendees: list[str]
    location: Optional[str] = None


class Ingredient(BaseModel):
    name: str
    amount: float
    unit: Literal["g", "ml", "piece", "tbsp"]


class Recipe(BaseModel):
    name: str
    servings: int
    ingredients: list[Ingredient]
    vegetarian: bool


TASKS: dict[type[BaseModel], list[str]] = {
    Person: ["a fictional software engineer from Madrid", "a retired teacher who loves chess", "a university student in Tokyo"],
    Product: ["a wireless mechanical keyboard", "an organic green tea pack", "a 4K action camera"],
    Weather: ["the current weather in Oslo in winter", "a hot summer afternoon in Seville", "a rainy day in London"],
    Order: ["an online order of two books shipped to Berlin", "a hardware store order with three items", "a single express order of headphones"],
    Event: ["a team offsite meeting next month", "a birthday party with four friends", "an online webinar about databases"],
    Recipe: ["a simple tomato pasta", "pancakes for breakfast", "a vegan lentil soup"],
}
