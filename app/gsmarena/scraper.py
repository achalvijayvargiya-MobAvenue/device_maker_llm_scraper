"""
Async GSMArena scraper using the public device index + direct page access.

Strategy (no search endpoint needed — avoids Cloudflare Turnstile):
  1. On first use, fetch the quicksearch index (~420 KB, one HTTP request).
     This JSON contains all ~5,700 indexed GSMArena devices with device IDs.
  2. Build a SQLite-backed local index: brand → [devices].
  3. For each lookup, fuzzy-match brand+model against the local index to find
     the device ID and construct the direct page URL.
  4. Scrape the device spec page directly (no CAPTCHA on individual pages).

Fallback: if the constructed URL returns 404 or the spec page has no sections,
try alternate URL slug constructions before giving up.

Rate limiting: RATE_LIMIT_DELAY seconds between HTTP requests (on spec page
fetches; the index fetch counts as one request).
Caching: scraped spec pages stored in SQLite so restarts cost nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

import httpx
from rapidfuzz import fuzz, process

from app.gsmarena.field_mapper import map_gsmarena_specs
from app.gsmarena.parser import GSMARENA_BASE, parse_spec_page
from app.logger import get_logger
from app.models import DeviceInput, DeviceSpec, RawDeviceSpec
from app.normalizer import normalize

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATE_LIMIT_DELAY: float = 4.0         # seconds between spec page fetches (respectful crawling)
REQUEST_TIMEOUT: float = 30.0
MAX_RETRIES: int = 4
FUZZY_THRESHOLD: int = 70             # minimum RapidFuzz token_set_ratio to accept
INDEX_URL = "https://www.gsmarena.com/quicksearch-8054.jpg?sSearch=x"

# 429 backoff sequence (seconds): waits get progressively longer
_BACKOFF_429 = [30.0, 60.0, 120.0, 180.0]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.gsmarena.com/",
}

_DEFAULT_CACHE_DB = (
    Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "gsmarena.db"
)


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """
    Convert a brand or model name to a GSMArena URL slug.

    - Lowercase
    - Remove accents
    - Replace non-alphanumeric (except digits) with underscore
    - Collapse consecutive underscores
    - Strip leading/trailing underscores
    """
    text = str(text).strip().lower()
    # Normalize unicode to ASCII (remove accents)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Replace special chars with underscore
    text = re.sub(r"[^a-z0-9]", "_", text)
    # Collapse runs of underscores
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _build_device_url(
    brand_name: str, model_name: str, device_id: int
) -> str:
    """Construct a GSMArena device page URL from brand, model, and device ID."""
    brand_slug = _slugify(brand_name)
    model_slug = _slugify(model_name)
    return f"{GSMARENA_BASE}/{brand_slug}_{model_slug}-{device_id}.php"


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS device_index (
            device_id   INTEGER PRIMARY KEY,
            brand_id    INTEGER,
            brand_name  TEXT,
            model_name  TEXT,
            model_variant TEXT,
            search_key  TEXT    -- lowercased "brand model" for fast lookup
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spec_cache (
            cache_key  TEXT PRIMARY KEY,
            url        TEXT,
            specs_json TEXT,
            scraped_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_miss (
            cache_key   TEXT PRIMARY KEY,
            query       TEXT,
            recorded_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    return conn


def _index_loaded(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT value FROM index_meta WHERE key = 'loaded'"
    ).fetchone()
    return row is not None and row[0] == "1"


def _store_index(
    conn: sqlite3.Connection,
    brands: dict[str, str],
    devices: list[list],
) -> None:
    """Populate the device_index table from raw quicksearch data."""
    rows = []
    for entry in devices:
        if len(entry) < 3:
            continue
        brand_id = int(entry[0])
        device_id = int(entry[1])
        model_name = str(entry[2]).strip()
        model_variant = str(entry[3]).strip() if len(entry) > 3 else ""
        brand_name = brands.get(str(brand_id), "")
        if not model_name or not brand_name:
            continue
        search_key = f"{brand_name} {model_name}".lower()
        rows.append(
            (device_id, brand_id, brand_name, model_name, model_variant, search_key)
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO device_index
        (device_id, brand_id, brand_name, model_name, model_variant, search_key)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.execute("INSERT OR REPLACE INTO index_meta VALUES ('loaded', '1')")
    conn.commit()
    logger.info("GSMArena device index stored: %d entries", len(rows))


def _query_index(
    conn: sqlite3.Connection, brand_name: str
) -> list[dict]:
    """Return all device entries for a given brand name (case-insensitive)."""
    rows = conn.execute(
        "SELECT device_id, brand_name, model_name, model_variant FROM device_index "
        "WHERE LOWER(brand_name) = LOWER(?)",
        (brand_name,),
    ).fetchall()
    return [
        {
            "device_id": r[0],
            "brand_name": r[1],
            "model_name": r[2],
            "model_variant": r[3],
        }
        for r in rows
    ]


def _query_index_all(conn: sqlite3.Connection) -> list[dict]:
    """Return all device entries (for cross-brand fuzzy search)."""
    rows = conn.execute(
        "SELECT device_id, brand_name, model_name, model_variant FROM device_index"
    ).fetchall()
    return [
        {
            "device_id": r[0],
            "brand_name": r[1],
            "model_name": r[2],
            "model_variant": r[3],
        }
        for r in rows
    ]


def _cache_key(brand: str, model: str) -> str:
    raw = f"{brand.strip().lower()}|{model.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_get(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT specs_json FROM spec_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    return json.loads(row[0]) if row else None


def _cache_put(conn: sqlite3.Connection, key: str, url: str, specs: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO spec_cache (cache_key, url, specs_json) VALUES (?, ?, ?)",
        (key, url, json.dumps(specs, ensure_ascii=False)),
    )
    conn.commit()


def _miss_known(conn: sqlite3.Connection, key: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM search_miss WHERE cache_key = ?", (key,)
    ).fetchone() is not None


def _miss_put(conn: sqlite3.Connection, key: str, query: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO search_miss (cache_key, query) VALUES (?, ?)",
        (key, query),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------


class GSMArenaScraper:
    """
    Async GSMArena scraper using the local device index + direct page access.

    Usage
    -----
    async with GSMArenaScraper() as scraper:
        spec = await scraper.lookup(brand="Samsung", model="Galaxy S24")
    """

    def __init__(
        self,
        cache_db: Optional[Path] = None,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
    ) -> None:
        self._cache_db = cache_db or _DEFAULT_CACHE_DB
        self._rate_delay = rate_limit_delay
        self._conn: Optional[sqlite3.Connection] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_sem = asyncio.Semaphore(1)
        self._last_request_at: float = 0.0
        self._consecutive_429s: int = 0   # circuit-breaker counter

    async def __aenter__(self) -> "GSMArenaScraper":
        self._conn = _init_db(self._cache_db)
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        # Load/refresh device index on first use
        if not _index_loaded(self._conn):
            await self._fetch_and_store_index()
        else:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM device_index"
            ).fetchone()[0]
            logger.info("GSMArena device index loaded from cache: %d entries", count)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
        if self._conn:
            self._conn.close()

    # ------------------------------------------------------------------
    # Public lookup
    # ------------------------------------------------------------------

    async def lookup(self, brand: str, model: str) -> Optional[DeviceSpec]:
        """
        Look up a device by brand+model and return a populated DeviceSpec.

        Returns None if no matching device is found or the spec page cannot
        be parsed.
        """
        key = _cache_key(brand, model)

        # Spec cache hit
        if self._conn:
            cached = _cache_get(self._conn, key)
            if cached is not None:
                logger.debug("Spec cache hit for %s %s", brand, model)
                return _dict_to_spec(cached, brand, model)

            if _miss_known(self._conn, key):
                logger.debug("Known miss for %s %s — skipping", brand, model)
                return None

        # Find device in local index
        match = self._find_in_index(brand, model)
        if match is None:
            logger.info("Index miss — %s %s", brand, model)
            _miss_put(self._conn, key, f"{brand} {model}")
            return None

        # Construct URL and scrape
        url = _build_device_url(match["brand_name"], match["model_name"], match["device_id"])
        logger.info(
            "Index match: '%s %s' -> '%s %s' (id=%d) | %s",
            brand, model,
            match["brand_name"], match["model_name"],
            match["device_id"],
            url,
        )

        specs = await self._scrape_spec_page(url)

        if specs is None:
            # Try alt URL: image-filename-style slug
            alt_url = self._alt_url(match, url)
            if alt_url and alt_url != url:
                logger.debug("Trying alt URL: %s", alt_url)
                specs = await self._scrape_spec_page(alt_url)
                if specs:
                    url = alt_url

        if specs is None:
            _miss_put(self._conn, key, f"{brand} {model}")
            return None

        _cache_put(self._conn, key, url, specs)
        return _dict_to_spec(specs, brand, model)

    # ------------------------------------------------------------------
    # Index lookup
    # ------------------------------------------------------------------

    def _find_in_index(self, brand: str, model: str) -> Optional[dict]:
        """
        Find the best-matching device in the local index.

        Strategy:
        1. Filter by brand name (exact match, case-insensitive).
        2. Fuzzy match model name using RapidFuzz token_set_ratio.
        3. If no brand match, try cross-brand fuzzy search.
        """
        brand_entries = _query_index(self._conn, brand)

        if brand_entries:
            return self._fuzzy_match_model(model, brand_entries)

        # Brand not found → try with known brand name variants
        alt_brand = _BRAND_ALIASES.get(brand.lower())
        if alt_brand:
            brand_entries = _query_index(self._conn, alt_brand)
            if brand_entries:
                return self._fuzzy_match_model(model, brand_entries)

        # Cross-brand search with full brand+model query
        full_query = f"{brand} {model}"
        all_entries = _query_index_all(self._conn)
        if not all_entries:
            return None

        # Use token_set_ratio against "brand_name model_name"
        choices = {
            f"{e['brand_name']} {e['model_name']}": e for e in all_entries
        }
        result = process.extractOne(
            full_query,
            choices.keys(),
            scorer=fuzz.token_set_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result:
            matched_key, score, _ = result
            logger.debug(
                "Cross-brand fuzzy: '%s' -> '%s' (score=%d)", full_query, matched_key, score
            )
            return choices[matched_key]

        return None

    def _fuzzy_match_model(
        self, model: str, entries: list[dict]
    ) -> Optional[dict]:
        """
        Fuzzy match a model name against a list of same-brand entries.

        Priority order:
        1. Exact match (case-insensitive).
        2. Character-level ratio (≥75) — penalizes extra tokens such as "Ultra".
        3. token_set_ratio (≥FUZZY_THRESHOLD) with length tie-breaker.

        A variant-suffix guard prevents "Reno11 A" from matching "Reno11 F":
        if the query ends with a distinctive single-letter or short suffix, the
        matched candidate must share that suffix.
        """
        query = model.strip().lower()
        choices = {e["model_name"]: e for e in entries}

        # 1. Exact match
        for name, entry in choices.items():
            if name.strip().lower() == query:
                logger.debug("Model exact match: '%s'", name)
                return entry

        def _suffix_ok(q: str, m: str) -> bool:
            """
            Return False if query ends with a distinctive suffix that differs from match,
            OR if query contains specific numbers absent from match (prevents Reno11 A
            from matching Reno A, etc.).
            """
            # Number token guard: query numbers must be a subset of match numbers
            q_nums = set(re.findall(r"\d+", q))
            m_nums = set(re.findall(r"\d+", m))
            if q_nums and not q_nums.issubset(m_nums):
                return False

            q_words = re.findall(r"[a-z]+", q.lower())
            m_words = re.findall(r"[a-z]+", m.lower())
            if not q_words or not m_words:
                return True
            q_last = q_words[-1]
            m_last = m_words[-1]
            variant_suffixes = {
                "pro", "plus", "ultra", "max", "fe", "lite", "mini",
                "edge", "gt", "neo", "go", "se",
            }
            # Single-letter suffix or short variant word → must match exactly
            if (len(q_last) == 1 and q_last.isalpha()) or q_last in variant_suffixes:
                return q_last == m_last
            return True

        # 2. Character-level ratio
        result = process.extractOne(
            query,
            [n.lower() for n in choices.keys()],
            scorer=fuzz.ratio,
            score_cutoff=75,
        )
        if result:
            matched_lower, score, _ = result
            if _suffix_ok(query, matched_lower):
                for original_name in choices:
                    if original_name.lower() == matched_lower:
                        logger.debug(
                            "Model ratio match: '%s' -> '%s' (score=%d)",
                            model, original_name, score,
                        )
                        return choices[original_name]

        # 3. token_set_ratio with length tie-breaker
        candidates = process.extract(
            query,
            [n.lower() for n in choices.keys()],
            scorer=fuzz.token_set_ratio,
            limit=5,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if candidates:
            query_len = len(query)
            # Filter by suffix compatibility
            compatible = [
                c for c in candidates if _suffix_ok(query, c[0])
            ]
            if compatible:
                best = min(compatible, key=lambda r: (abs(len(r[0]) - query_len), -r[1]))
                matched_lower = best[0]
                best_score = best[1]
                for original_name in choices:
                    if original_name.lower() == matched_lower:
                        logger.debug(
                            "Model token_set match: '%s' -> '%s' (score=%d)",
                            model, original_name, best_score,
                        )
                        return choices[original_name]

        return None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _fetch_and_store_index(self) -> None:
        """Fetch the quicksearch device index and store it in SQLite."""
        logger.info("Fetching GSMArena device index from %s", INDEX_URL)
        html = await self._get_raw(INDEX_URL)
        if not html:
            logger.error("Failed to fetch device index")
            return
        try:
            data = json.loads(html)
            brands: dict[str, str] = data[0]    # {str_id: brand_name}
            devices: list[list] = data[1]        # [[brand_id, device_id, model, ...], ...]
            _store_index(self._conn, brands, devices)
        except (json.JSONDecodeError, IndexError, KeyError) as exc:
            logger.error("Failed to parse device index: %s", exc)

    async def _scrape_spec_page(self, url: str) -> Optional[dict]:
        """Fetch a spec page and return the parsed spec dict."""
        html = await self._get(url)
        if not html:
            return None
        specs = parse_spec_page(html)
        if not any(k for k in specs if not k.startswith("_")):
            logger.debug("Spec page has no parseable sections (CAPTCHA/error?): %s", url)
            return None
        return specs

    async def _get(self, url: str) -> Optional[str]:
        """Rate-limited GET with retry logic."""
        async with self._rate_sem:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._rate_delay:
                await asyncio.sleep(self._rate_delay - elapsed)
            result = await self._get_raw(url)
            self._last_request_at = time.monotonic()
            return result

    async def _get_raw(self, url: str) -> Optional[str]:
        """HTTP GET without rate limiting (for index fetch)."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.get(url)
                if resp.status_code == 200:
                    self._consecutive_429s = 0
                    return resp.text
                if resp.status_code == 404:
                    logger.debug("404: %s", url)
                    return None
                if resp.status_code in (429, 503):
                    self._consecutive_429s += 1
                    wait = _BACKOFF_429[min(attempt - 1, len(_BACKOFF_429) - 1)]
                    # If we keep hitting 429, add extra cooldown on top
                    if self._consecutive_429s > 2:
                        wait = min(wait + 60.0 * (self._consecutive_429s - 2), 300.0)
                    logger.warning(
                        "Rate limited (%d) — cooling down %.0fs (consecutive=%d)",
                        resp.status_code, wait, self._consecutive_429s,
                    )
                    # Also bump the global rate delay temporarily
                    self._rate_delay = min(
                        RATE_LIMIT_DELAY + 2.0 * self._consecutive_429s, 10.0
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.debug("HTTP %d: %s", resp.status_code, url)
                return None
            except (httpx.ConnectTimeout, httpx.ReadTimeout):
                logger.warning("Timeout attempt %d/%d: %s", attempt, MAX_RETRIES, url)
                await asyncio.sleep(3.0 * attempt)
            except httpx.RequestError as exc:
                logger.warning("Request error: %s — %s", exc, url)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(3.0 * attempt)
                else:
                    return None
        return None

    @staticmethod
    def _alt_url(match: dict, failed_url: str) -> Optional[str]:
        """
        Build an alternate URL slug when the primary URL returned no specs.

        Uses the model_variant field from the index as additional slug text.
        """
        # If model has a variant (e.g. "5G"), include it
        variant = match.get("model_variant", "").strip()
        if not variant:
            return None
        # Strip keyword tags like "5G Notch PHC" — only take first word
        first_variant = variant.split()[0] if variant else ""
        if first_variant.upper() in ("5G", "NOTCH", "PHC", "UDC"):
            return None
        combined_model = f"{match['model_name']} {first_variant}"
        return _build_device_url(match["brand_name"], combined_model, match["device_id"])


# ---------------------------------------------------------------------------
# Known brand aliases (for brands not in GSMArena's list exactly)
# ---------------------------------------------------------------------------

_BRAND_ALIASES: dict[str, str] = {
    "lg": "LG",
    "oneplus": "OnePlus",
    "poco": "POCO",
    "zte": "ZTE",
    "htc": "HTC",
    "lge": "LG",
    "tcl": "TCL",
    "nothing": "Nothing",
    "google": "Google",
    "nokia": "Nokia",
    "motorola": "Motorola",
    "samsung": "Samsung",
    "apple": "Apple",
    "xiaomi": "Xiaomi",
    "huawei": "Huawei",
    "oppo": "Oppo",
    "vivo": "vivo",
    "realme": "Realme",
    "infinix": "Infinix",
    "tecno": "Tecno",
    "honor": "Honor",
    "asus": "Asus",
    "sony": "Sony",
    "lenovo": "Lenovo",
}


# ---------------------------------------------------------------------------
# Spec dict → DeviceSpec conversion
# ---------------------------------------------------------------------------


def _dict_to_spec(
    raw_specs: dict, original_brand: str, original_model: str
) -> Optional[DeviceSpec]:
    """Convert a cached/freshly-scraped spec dict into a DeviceSpec."""
    try:
        mapped = map_gsmarena_specs(raw_specs, original_brand, original_model)
        raw = RawDeviceSpec(
            device_manufacturer=mapped.get("device_manufacturer"),
            device_model=mapped.get("device_model"),
            display_size_inch=mapped.get("display_size_inch"),
            screen_resolution=mapped.get("screen_resolution"),
            display_refresh_hz=mapped.get("display_refresh_hz"),
            price_inr=mapped.get("price_inr"),
            back_camera_mp_total=mapped.get("back_camera_mp_total"),
            front_camera_mp=mapped.get("front_camera_mp"),
            cpu_gpu=mapped.get("cpu_gpu"),
            chipset=mapped.get("chipset"),
            chipset_tier=mapped.get("chipset_tier"),
            cpu_cores=mapped.get("cpu_cores"),
            gpu_class=mapped.get("gpu_class"),
            antutu_score=mapped.get("antutu_score"),
            ram_gb=mapped.get("ram_gb"),
            storage_gb=mapped.get("storage_gb"),
            battery_mah=mapped.get("battery_mah"),
            cooling_system=mapped.get("cooling_system"),
            wifi=mapped.get("wifi"),
            nfc=mapped.get("nfc"),
            five_g_supported=mapped.get("five_g_supported"),
            launch_date=mapped.get("launch_date"),
            months_since_launch=mapped.get("months_since_launch"),
        )
        return normalize(raw)
    except Exception as exc:
        logger.warning(
            "Failed to convert spec for %s %s: %s", original_brand, original_model, exc
        )
        return None
