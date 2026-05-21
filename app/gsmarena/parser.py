"""
GSMArena HTML spec-page parser.

Parses the raw HTML of a GSMArena device spec page and returns a flat dict
of {section -> {label -> value}} that the field mapper can consume.

Also parses search-result pages to extract (device_name, url) pairs.
"""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup, Tag

from app.logger import get_logger

logger = get_logger(__name__)

# GSMArena base URL
GSMARENA_BASE = "https://www.gsmarena.com"


# ---------------------------------------------------------------------------
# Search result parsing
# ---------------------------------------------------------------------------


def parse_search_results(html: str) -> list[tuple[str, str]]:
    """
    Parse a GSMArena search results page.

    Parameters
    ----------
    html:
        Raw HTML from the search endpoint.

    Returns
    -------
    list[tuple[str, str]]
        List of (device_name, relative_url) pairs, in result order.
        Returns empty list if no results found.
    """
    soup = BeautifulSoup(html, "lxml")
    results: list[tuple[str, str]] = []

    # GSMArena search results are in <div class="makers"> > <ul> > <li> > <a>
    makers_div = soup.find("div", class_="makers")
    if not makers_div:
        # Fallback: look for any result links matching the device URL pattern
        for a in soup.find_all("a", href=re.compile(r"^[a-z0-9_]+-\d+\.php$")):
            name = a.get_text(strip=True)
            href = a.get("href", "")
            if name and href:
                results.append((name, href))
        return results

    for li in makers_div.find_all("li"):
        a_tag = li.find("a")
        if not a_tag:
            continue
        href = a_tag.get("href", "")
        if not href or not re.match(r"^[a-z0-9_]+-\d+\.php$", href):
            continue

        # Device name is in <strong> > <span> elements or directly in <strong>
        strong = a_tag.find("strong")
        if strong:
            spans = strong.find_all("span")
            if spans:
                name = " ".join(s.get_text(strip=True) for s in spans if s.get_text(strip=True))
            else:
                name = strong.get_text(strip=True)
        else:
            name = a_tag.get_text(strip=True)

        if name and href:
            results.append((name, href))

    logger.debug("Search page parsed: %d results", len(results))
    return results


# ---------------------------------------------------------------------------
# Spec page parsing
# ---------------------------------------------------------------------------


def parse_spec_page(html: str) -> dict[str, dict[str, str]]:
    """
    Parse a GSMArena device spec page into a nested dict.

    Parameters
    ----------
    html:
        Raw HTML of a device spec page.

    Returns
    -------
    dict[str, dict[str, str]]
        {section_name -> {label -> value}}

        Example::

            {
              "Display": {
                "Size": "6.7 inches, 107.4 cm2 (~88.3% screen-to-body ratio)",
                "Resolution": "1080 x 2400 pixels, 20:9 ratio (~393 ppi density)",
                "Refresh Rate": "120Hz",
              },
              "Platform": {
                "Chipset": "Qualcomm SM8550-AB Snapdragon 8 Gen 2 (4 nm)",
                "CPU": "Octa-core (1x3.36 GHz Cortex-X3 ...",
                "GPU": "Adreno 740",
              },
              ...
            }
    """
    soup = BeautifulSoup(html, "lxml")
    specs: dict[str, dict[str, str]] = {}

    # Also capture device title (useful for logging / cross-checks)
    title_tag = soup.find("h1", class_="specs-phone-name-title")
    if not title_tag:
        title_tag = soup.find("h1")
    if title_tag:
        specs["_meta"] = {"device_title": title_tag.get_text(strip=True)}

    # The spec tables are inside <div id="specs-list">
    specs_list = soup.find("div", id="specs-list")
    if not specs_list:
        # Some pages render without wrapper — fall back to all tables
        specs_list = soup

    current_section: str = ""

    for table in specs_list.find_all("table"):
        for row in table.find_all("tr"):
            # Section header: <th rowspan="N">Section name</th>
            # IMPORTANT: the <th> and the first spec cells share the same <tr>,
            # so we must NOT skip this row — process both the header AND the cells.
            th = row.find("th")
            if th:
                current_section = th.get_text(strip=True)
                if current_section not in specs:
                    specs[current_section] = {}
                # Fall through to process ttl/nfo cells in this same row

            # Spec cells: <td class="ttl">Label</td> <td class="nfo">Value</td>
            ttl_td = row.find("td", class_="ttl")
            nfo_td = row.find("td", class_="nfo")
            if not ttl_td or not nfo_td:
                continue

            label = ttl_td.get_text(strip=True)
            value = _clean_cell_text(nfo_td)

            if value and current_section:
                # Use label as key; for empty labels (continuation rows) use a
                # unique placeholder so data isn't silently dropped
                key = label if label else f"_cont_{len(specs[current_section])}"
                specs[current_section][key] = value

    if not any(k for k in specs if not k.startswith("_")):
        logger.warning("Spec page parsed but found 0 sections — possibly blocked/CAPTCHA")

    logger.debug(
        "Spec page parsed: %d sections, %d total fields",
        len(specs),
        sum(len(v) for v in specs.values()),
    )
    return specs


def _clean_cell_text(td: Tag) -> str:
    """
    Extract clean text from a spec value cell.

    Strips footnote superscripts, normalizes whitespace, and removes
    stray non-breaking spaces / special Unicode.
    """
    # Remove <sup> tags (footnote references)
    for sup in td.find_all("sup"):
        sup.decompose()

    text = td.get_text(separator=" ", strip=True)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Replace non-breaking space with regular space
    text = text.replace("\u00a0", " ").replace("\xa0", " ")
    return text


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def extract_device_name_from_spec_page(html: str) -> Optional[str]:
    """Extract just the device name from the spec page title."""
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1", class_="specs-phone-name-title")
    if not h1:
        h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else None
