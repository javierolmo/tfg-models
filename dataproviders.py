"""Backwards-compatibility wrapper for data providers."""

from data.providers import AzureDataProvider, DataProvider, LocalDataProvider

__all__ = ["DataProvider", "AzureDataProvider", "LocalDataProvider"]
