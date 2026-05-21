"""
GSMArena scraping sub-package.

Provides async scraping of device specifications from GSMArena.com.

Usage
-----
from app.gsmarena.scraper import GSMArenaScraper

async with GSMArenaScraper() as scraper:
    spec = await scraper.lookup(brand="Samsung", model="Galaxy S24")
"""

from app.gsmarena.scraper import GSMArenaScraper  # noqa: F401

__all__ = ["GSMArenaScraper"]
