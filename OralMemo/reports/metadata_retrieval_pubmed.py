import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
_THREAD_LOCAL = threading.local()
TIMEOUT = 30
DEFAULT_CONCURRENCY = 5
DEFAULT_API_RPS_WITH_KEY = 8.0
DEFAULT_API_RPS_WITHOUT_KEY = 2.5
DEFAULT_BATCH_EFETCH_THRESHOLD = 20
DEFAULT_MIN_TITLE_SIMILARITY = 0.75
DEFAULT_PUBMED_PROXY = "http://147.8.145.20:8282"
DEFAULT_SRMA_BUCKET_SIZE = 5
DEFAULT_CACHE_SAVE_INTERVAL_SECONDS = 300.0
NCBI_FETCH_MAX_RETRIES = 5
NCBI_FETCH_RETRY_BASE_DELAY = 1.0
PUBMED_SEARCH_MAX_ATTEMPTS = 3
PUBMED_SEARCH_RETRY_BASE_DELAY = 2.0

PUBMED_TITLE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for",
    "from", "had", "has", "have", "in", "into", "is", "it", "its", "of", "on", "onto",
    "or", "per", "than", "that", "the", "their", "there", "these", "this", "those", "to",
    "vs", "versus", "via", "was", "were", "with", "without",
}

INCLUDED_STUDIES_DIR = Path(__file__).resolve().parent
PAPER_INFO_DIR = Path(__file__).resolve().parent.parent / "srma" / "paper_info_all"
METADATA_BASE_DIR = INCLUDED_STUDIES_DIR / "metadata"
METADATA_SR_ONLY_DIR = METADATA_BASE_DIR / "sr_only"
METADATA_SRMA_DIR = METADATA_BASE_DIR / "srma"
DEFAULT_CACHE_PATH = METADATA_BASE_DIR / "pubmed_cache.json"

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from pubmed_api import efetch_full_articles, esummary, _sleep_for_rate  # noqa: E402


class NCBIRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / max(0.1, float(requests_per_second))
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
                now = time.monotonic()
            self.next_at = now + self.interval


RATE_LIMITER: Optional[NCBIRateLimiter] = None


def configure_rate_limiter(requests_per_second: Optional[float]) -> None:
    global RATE_LIMITER
    RATE_LIMITER = NCBIRateLimiter(requests_per_second) if requests_per_second else None


def configure_pubmed_proxy(proxy_url: Optional[str]) -> None:
    proxy = (proxy_url or "").strip()
    if not proxy:
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = proxy
    os.environ["NCBI_USE_SYSTEM_PROXY"] = "1"


def _wait_for_ncbi_request() -> None:
    if RATE_LIMITER is not None:
        RATE_LIMITER.wait()


def _sleep_between_ncbi_calls(api_key: Optional[str]) -> None:
    if RATE_LIMITER is None:
        _sleep_for_rate(api_key)


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        raw = (os.getenv("NCBI_USE_SYSTEM_PROXY") or "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            session.trust_env = False
        _THREAD_LOCAL.session = session
    return session


def atomic_write_json(path: Path, payload: Any, *, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False)
            f.flush()
        tmp_path.replace(path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


class PubMedMetadataCache:
    def __init__(self, path: Optional[Path], *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled and path is not None
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {"search": {}, "efetch": {}}
        self.dirty = False
        if self.enabled and self.path and self.path.is_file():
            try:
                with open(self.path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data["search"] = loaded.get("search") if isinstance(loaded.get("search"), dict) else {}
                    self.data["efetch"] = loaded.get("efetch") if isinstance(loaded.get("efetch"), dict) else {}
            except Exception as e:
                log_runtime_error(f"load PubMed cache path={str(self.path)!r}", e)

    def get_search(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self.lock:
            item = self.data.get("search", {}).get(key)
            return dict(item) if isinstance(item, dict) else None

    def set_search_if_stable(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled or value.get("found") != "yes" or value.get("error"):
            return
        with self.lock:
            new_value = dict(value)
            search_cache = self.data.setdefault("search", {})
            if search_cache.get(key) != new_value:
                search_cache[key] = new_value
                self.dirty = True

    def get_efetch(self, pmid: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self.lock:
            item = self.data.get("efetch", {}).get(str(pmid))
            return dict(item) if isinstance(item, dict) else None

    def set_efetch(self, pmid: str, value: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(value, dict):
            return
        with self.lock:
            key = str(pmid)
            new_value = dict(value)
            efetch_cache = self.data.setdefault("efetch", {})
            if efetch_cache.get(key) != new_value:
                efetch_cache[key] = new_value
                self.dirty = True

    def save(self, *, force: bool = False) -> None:
        if not self.enabled or self.path is None:
            return
        if not force and not self.dirty:
            return
        try:
            atomic_write_json(self.path, self.data)
            self.dirty = False
        except Exception as e:
            log_runtime_error(f"save PubMed cache path={str(self.path)!r}", e)


def _pubmed_search_cache_key(
    title: str,
    authors: Optional[List[str]],
    query_year: Optional[int],
    min_title_similarity: float,
) -> str:
    payload = {
        "title": _normalize_pubmed_literature_query_text(title or ""),
        "authors": [str(a).strip() for a in (authors or []) if str(a).strip()],
        "query_year": query_year,
        "min_title_similarity": float(min_title_similarity),
        "version": 1,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def log_runtime_error(context: str, err: Exception) -> None:
    print(f"[ERROR] {context}: {err}", file=sys.stderr, flush=True)


def _progress(iterable: Iterable[Any], desc: str, *, enabled: bool = True, unit: str = "row") -> Iterable[Any]:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit)


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def similarity(a: Optional[str], b: Optional[str]) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def parse_year(value: Optional[str]) -> Optional[int]:
    if value is None or not str(value).strip():
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def extract_included_studies_entries(data: Dict[str, Any]) -> Dict[str, Any]:
    raw_details = data.get("included_studies_details")
    if isinstance(raw_details, dict):
        return {str(k): v for k, v in raw_details.items()}
    if isinstance(raw_details, list):
        return {f"study_{i + 1}": v for i, v in enumerate(raw_details)}

    legacy = data.get("included_studies") or {}
    if isinstance(legacy, dict):
        return {str(k): v for k, v in legacy.items()}
    return {}


def authors_string_to_list(s: Optional[str]) -> List[str]:
    if not s or not str(s).strip():
        return []
    parts = re.split(r"\s*,\s*", str(s))
    return [p.strip() for p in parts if p.strip()]


def _normalize_pubmed_literature_query_text(s: str) -> str:
    if not s:
        return ""
    t = str(s)
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"[\r\n]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_pubmed_title_only_term(title: str) -> Optional[str]:
    t = _normalize_pubmed_literature_query_text(title or "")
    if not t:
        return None
    raw_tokens = re.findall(r"[A-Za-z0-9]+", t.lower())
    tokens = [tok for tok in raw_tokens if tok and tok not in PUBMED_TITLE_STOPWORDS]
    if not tokens:
        return None
    return " ".join(f"{tok}[Title]" for tok in tokens)


def build_pubmed_keyword_author_year_term(
    title: str,
    authors: Optional[List[str]] = None,
    query_year: Optional[int] = None,
    *,
    max_keywords: int = 10,
) -> Optional[str]:
    t = _normalize_pubmed_literature_query_text(title or "")
    if not t:
        return None
    raw_tokens = re.findall(r"[A-Za-z0-9]+", t.lower())
    tokens = [tok for tok in raw_tokens if tok and tok not in PUBMED_TITLE_STOPWORDS]
    if max_keywords > 0:
        tokens = tokens[:max_keywords]

    parts: List[str] = []
    if tokens:
        parts.append(" ".join(tokens))

    first_author = ""
    if authors:
        first_author = str(authors[0] or "").strip()
    if first_author:
        parts.append(first_author)

    if query_year is not None:
        parts.append(str(query_year))

    term = " ".join(p for p in parts if p).strip()
    return term or None


def ncbi_fetch(method: str, url: str, form: Dict[str, Any]) -> requests.Response:
    req_kw: Dict[str, Any] = {"timeout": TIMEOUT}
    method_upper = method.upper()
    last_err: Optional[Exception] = None
    for attempt in range(1, NCBI_FETCH_MAX_RETRIES + 1):
        try:
            _wait_for_ncbi_request()
            session = _get_session()
            if method_upper == "POST":
                resp = session.post(url, data=form, **req_kw)
            else:
                resp = session.get(url, params=form, **req_kw)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            if attempt >= NCBI_FETCH_MAX_RETRIES:
                break
            # DNS/temporary connectivity hiccups are common; use exponential backoff.
            sleep_s = NCBI_FETCH_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(sleep_s)
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("ncbi_fetch failed without a captured exception")


def esearch_pubmed(
    term: str,
    *,
    retmax: int = 25,
    retstart: int = 0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{NCBI_EUTILS_BASE}/esearch.fcgi"
    form: Dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(retmax),
        "retstart": str(retstart),
        "sort": "relevance",
    }
    if api_key:
        form["api_key"] = api_key
    resp = ncbi_fetch("POST", url, form)
    return resp.json().get("esearchresult", {})


def esummary_pubmed(pmids: List[str], *, api_key: Optional[str] = None) -> Dict[str, Any]:
    _wait_for_ncbi_request()
    return esummary(pmids, api_key=api_key)


def efetch_full_articles_pubmed(
    pmids: List[str],
    *,
    api_key: Optional[str] = None,
    batch_size: int = 10,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(pmids), batch_size):
        _wait_for_ncbi_request()
        chunk = pmids[i : i + batch_size]
        out.update(efetch_full_articles(chunk, api_key=api_key, batch_size=batch_size))
    return out


def normalize_pmcid_value(pmcid: Optional[str]) -> Optional[str]:
    if pmcid is None:
        return None
    s = str(pmcid).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    matches = list(re.finditer(r"(?i)PMC\s*(\d+)", s))
    if matches:
        return str(int(matches[-1].group(1)))
    return None


def extract_doi_pmcid_from_esummary_article(article: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    for item in article.get("articleids", []) or []:
        it = str(item.get("idtype") or "").lower()
        val = str(item.get("value") or "").strip()
        if not val:
            continue
        if it == "doi":
            doi = val.replace("https://doi.org/", "").strip()
        elif it in ("pmc", "pmcid"):
            pmcid = normalize_pmcid_value(val)
    return doi, pmcid


def pubmed_search_by_title_only(
    title: str,
    *,
    authors: Optional[List[str]] = None,
    query_year: Optional[int] = None,
    retmax: int = 25,
    min_title_similarity: float = DEFAULT_MIN_TITLE_SIMILARITY,
    include_full_article: bool = True,
    cache: Optional[PubMedMetadataCache] = None,
) -> Dict[str, Any]:
    empty: Dict[str, Any] = {
        "found": "no",
        "pmid": None,
        "pmcid": None,
        "pubmed_url": None,
        "doi": None,
        "matched_title": None,
        "title_similarity": None,
        "abstract": None,
        "mesh_terms": [],
        "error": None,
    }
    term_primary = build_pubmed_title_only_term(title)
    term_fallback = _normalize_pubmed_literature_query_text(title or "")
    term_fallback_kw_author_year = build_pubmed_keyword_author_year_term(
        title,
        authors=authors,
        query_year=query_year,
    )
    terms_to_try: List[str] = []
    if term_primary:
        terms_to_try.append(term_primary)
    if term_fallback:
        terms_to_try.append(term_fallback)
    if term_fallback_kw_author_year:
        terms_to_try.append(term_fallback_kw_author_year)
    if not terms_to_try:
        return dict(empty)

    cache_key = _pubmed_search_cache_key(title, authors, query_year, min_title_similarity)
    cached = cache.get_search(cache_key) if cache is not None else None
    if cached is not None:
        if include_full_article and cached.get("found") == "yes" and cached.get("pmid"):
            cached = fill_pubmed_block_from_efetch(cached, cache=cache)
        return cached

    api_key = (NCBI_API_KEY or "").strip() or None
    last_err: Optional[Exception] = None
    for attempt in range(1, PUBMED_SEARCH_MAX_ATTEMPTS + 1):
        try:
            pmids: List[str] = []
            for i, term in enumerate(terms_to_try):
                es = esearch_pubmed(term, retmax=retmax, api_key=api_key)
                pmids = [str(x) for x in (es.get("idlist") or []) if x]
                if pmids:
                    break
                if i < len(terms_to_try) - 1:
                    _sleep_between_ncbi_calls(api_key)
            if not pmids:
                return dict(empty)

            _sleep_between_ncbi_calls(api_key)
            summary = esummary_pubmed(pmids, api_key=api_key)
            result = summary.get("result") or {}
            uids = [str(u) for u in (result.get("uids") or [])]
            if not uids:
                return dict(empty)

            best_uid: Optional[str] = None
            best_title = ""
            best_title_similarity = -1.0
            best_rank_score = -1.0
            for uid in uids:
                article = result.get(uid) or {}
                hit_title = article.get("title") or ""
                ts = similarity(title, hit_title)
                ys = 0.0
                if query_year is not None:
                    pd = str(article.get("pubdate") or "")
                    if str(query_year) in pd:
                        ys = 0.12
                combined = ts + ys
                if combined > best_rank_score:
                    best_rank_score = combined
                    best_title_similarity = ts
                    best_title = hit_title
                    best_uid = uid

            if best_uid is None or best_title_similarity < min_title_similarity:
                return dict(empty)

            article = result.get(best_uid) or {}
            doi, pmcid = extract_doi_pmcid_from_esummary_article(article)
            out = {
                "found": "yes",
                "pmid": best_uid,
                "pmcid": pmcid,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{best_uid}/",
                "doi": doi,
                "matched_title": best_title,
                "title_similarity": round(best_title_similarity, 4),
                "abstract": None,
                "mesh_terms": [],
                "error": None,
            }
            if cache is not None:
                cache.set_search_if_stable(cache_key, out)
            if include_full_article:
                _sleep_between_ncbi_calls(api_key)
                out = fill_pubmed_block_from_efetch(out, cache=cache)
            return out
        except Exception as e:
            last_err = e
            if attempt < PUBMED_SEARCH_MAX_ATTEMPTS:
                time.sleep(PUBMED_SEARCH_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                continue

    if last_err is not None:
        log_runtime_error(f"pubmed_search_by_title_only title={title!r}", last_err)
    out = dict(empty)
    out["error"] = str(last_err) if last_err is not None else "unknown error"
    return out


def fill_pubmed_block_from_efetch(
    pubmed_block: Dict[str, Any],
    *,
    cache: Optional[PubMedMetadataCache] = None,
) -> Dict[str, Any]:
    out = dict(pubmed_block)
    pmid = str(out.get("pmid") or "").strip()
    if out.get("found") != "yes" or not pmid:
        return out

    extra = cache.get_efetch(pmid) if cache is not None else None
    if extra is None:
        api_key = (NCBI_API_KEY or "").strip() or None
        try:
            efetch_data = efetch_full_articles_pubmed([pmid], api_key=api_key, batch_size=10)
            extra = efetch_data.get(pmid) or {}
            if cache is not None and extra:
                cache.set_efetch(pmid, extra)
        except Exception as e:
            log_runtime_error(f"efetch pmid={pmid}", e)
            out["error"] = f"efetch failed: {e}"
            return out

    out["abstract"] = (extra.get("abstract") or "").strip() or None
    out["mesh_terms"] = list(extra.get("mesh_terms") or [])
    out["error"] = None
    return out


def _item_has_retryable_api_error(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    pubmed = item.get("pubmed") if isinstance(item.get("pubmed"), dict) else {}
    return bool(pubmed.get("error"))


def enrich_included_study_from_pubmed(
    study: Dict[str, Any],
    *,
    min_title_similarity: float = DEFAULT_MIN_TITLE_SIMILARITY,
    include_full_article: bool = True,
    cache: Optional[PubMedMetadataCache] = None,
) -> Dict[str, Any]:
    title = (study.get("title") or "").strip()
    authors_raw = (study.get("authors") or "").strip() or None
    year_int = parse_year(study.get("year"))
    authors_list = authors_string_to_list(authors_raw)
    if not authors_list and authors_raw:
        authors_list = [authors_raw.strip()]

    pubmed_block = pubmed_search_by_title_only(
        title,
        authors=authors_list,
        query_year=year_int,
        min_title_similarity=min_title_similarity,
        include_full_article=include_full_article,
        cache=cache,
    )
    out: Dict[str, Any] = dict(study)
    out["title"] = title or out.get("title")
    out["authors"] = authors_list
    out["year"] = year_int if year_int is not None else study.get("year")
    out["pubmed"] = pubmed_block
    if pubmed_block.get("error"):
        out["api_errors"] = {"pubmed": str(pubmed_block.get("error"))}
    return out


def _batch_fill_efetch_for_items(
    items: Dict[str, Dict[str, Any]],
    *,
    cache: Optional[PubMedMetadataCache] = None,
    batch_size: int = 100,
) -> None:
    pmids: List[str] = []
    seen: set[str] = set()
    for item in items.values():
        pubmed = item.get("pubmed") if isinstance(item.get("pubmed"), dict) else {}
        pmid = str(pubmed.get("pmid") or "").strip()
        if pubmed.get("found") == "yes" and pmid and pmid not in seen:
            seen.add(pmid)
            if cache is None or cache.get_efetch(pmid) is None:
                pmids.append(pmid)

    if pmids:
        api_key = (NCBI_API_KEY or "").strip() or None
        for i in range(0, len(pmids), batch_size):
            chunk = pmids[i : i + batch_size]
            try:
                fetched = efetch_full_articles_pubmed(chunk, api_key=api_key, batch_size=len(chunk))
            except Exception as e:
                log_runtime_error(f"batch efetch pmids={','.join(chunk)}", e)
                for item in items.values():
                    pubmed = item.get("pubmed") if isinstance(item.get("pubmed"), dict) else {}
                    pmid = str(pubmed.get("pmid") or "").strip()
                    if pmid in chunk:
                        pubmed["error"] = f"efetch failed: {e}"
                        item["pubmed"] = pubmed
                        item["api_errors"] = {"pubmed": str(pubmed["error"])}
                continue
            if cache is not None:
                for pmid, extra in fetched.items():
                    cache.set_efetch(pmid, extra)

    for item in items.values():
        pubmed = item.get("pubmed") if isinstance(item.get("pubmed"), dict) else {}
        updated_pubmed = fill_pubmed_block_from_efetch(pubmed, cache=cache)
        item["pubmed"] = updated_pubmed
        if updated_pubmed.get("error"):
            item["api_errors"] = {"pubmed": str(updated_pubmed["error"])}
        elif isinstance(item.get("api_errors"), dict):
            item["api_errors"].pop("pubmed", None)
            if not item["api_errors"]:
                item.pop("api_errors", None)


def process_paper_info_file(
    json_path: Path,
    *,
    delay_seconds: float = 0.25,
    concurrency: int = DEFAULT_CONCURRENCY,
    existing_included: Optional[Dict[str, Any]] = None,
    retry_only_errored_items: bool = False,
    min_title_similarity: float = DEFAULT_MIN_TITLE_SIMILARITY,
    cache: Optional[PubMedMetadataCache] = None,
    batch_efetch_threshold: int = DEFAULT_BATCH_EFETCH_THRESHOLD,
    batch_efetch_size: int = 100,
) -> Dict[str, Any]:
    srma_pmid = json_path.stem
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    raw_included = extract_included_studies_entries(data)

    def _included_study_sort_key(k: Any) -> Tuple[int, int, str]:
        s = str(k)
        m = re.match(r"^study_(\d+)$", s, flags=re.IGNORECASE)
        if m:
            return (0, int(m.group(1)), s)
        return (1, 0, s)

    existing_included = existing_included or {}
    out_included: Dict[str, Any] = {}
    keys_sorted = sorted(raw_included.keys(), key=_included_study_sort_key)
    keys_to_process = keys_sorted
    if retry_only_errored_items:
        keys_to_process = []
        for key in keys_sorted:
            existing_item = existing_included.get(key)
            if _item_has_retryable_api_error(existing_item):
                keys_to_process.append(key)
            elif isinstance(existing_item, dict):
                out_included[key] = existing_item

    worker_count = max(1, int(concurrency))
    if worker_count == 1:
        for i, key in enumerate(keys_to_process):
            entry = raw_included[key]
            if not isinstance(entry, dict):
                continue
            out_included[key] = enrich_included_study_from_pubmed(
                entry,
                min_title_similarity=min_title_similarity,
                include_full_article=len(keys_to_process) < batch_efetch_threshold,
                cache=cache,
            )
            if delay_seconds > 0 and i < len(keys_to_process) - 1:
                time.sleep(delay_seconds)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_key = {}
            for key in keys_to_process:
                entry = raw_included[key]
                if not isinstance(entry, dict):
                    continue
                fut = executor.submit(
                    enrich_included_study_from_pubmed,
                    entry,
                    min_title_similarity=min_title_similarity,
                    include_full_article=len(keys_to_process) < batch_efetch_threshold,
                    cache=cache,
                )
                future_to_key[fut] = key
            for fut in as_completed(future_to_key):
                key = future_to_key[fut]
                try:
                    out_included[key] = fut.result()
                except Exception as e:
                    log_runtime_error(f"process_paper_info_file pmid={srma_pmid} key={key}", e)
                    out_included[key] = dict(raw_included.get(key) or {})
                    out_included[key]["error"] = str(e)

    if len(keys_to_process) >= batch_efetch_threshold:
        processed_items = {
            key: out_included[key]
            for key in keys_to_process
            if isinstance(out_included.get(key), dict)
        }
        _batch_fill_efetch_for_items(
            processed_items,
            cache=cache,
            batch_size=max(1, int(batch_efetch_size)),
        )
        out_included.update(processed_items)

    return {
        "srma_pmid": srma_pmid,
        "source_paper_info": str(json_path.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_of_included_studies": len(out_included),
        "included_studies": out_included,
    }


def discover_paper_info_json_paths() -> List[Path]:
    if not PAPER_INFO_DIR.is_dir():
        return []
    return sorted(p for p in PAPER_INFO_DIR.glob("*.json") if p.is_file())


def included_studies_count(json_path: Path) -> int:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log_runtime_error(f"load paper_info for included-studies count json_path={str(json_path)!r}", e)
        return 0
    if not isinstance(data, dict):
        return 0
    return sum(1 for item in extract_included_studies_entries(data).values() if isinstance(item, dict))


def srma_bucket_upper_bound(count: int, *, bucket_size: int = DEFAULT_SRMA_BUCKET_SIZE) -> int:
    size = max(1, int(bucket_size))
    if count <= 0:
        return size
    return ((count - 1) // size + 1) * size


def order_paper_info_paths(
    paths: List[Path],
    *,
    srma_bucket_size: int = DEFAULT_SRMA_BUCKET_SIZE,
) -> Tuple[List[Path], Dict[Path, int], Dict[int, int]]:
    srma_items: List[Tuple[int, int, str, Path]] = []
    other_paths: List[Path] = []
    srma_bucket_by_path: Dict[Path, int] = {}
    srma_bucket_counts: Dict[int, int] = {}

    for path in paths:
        out_dir = classify_output_dir_for_paper_info(path)
        if out_dir == METADATA_SRMA_DIR:
            study_count = included_studies_count(path)
            bucket = srma_bucket_upper_bound(study_count, bucket_size=srma_bucket_size)
            srma_items.append((bucket, study_count, path.stem, path))
            srma_bucket_by_path[path] = bucket
            srma_bucket_counts[bucket] = srma_bucket_counts.get(bucket, 0) + 1
        else:
            other_paths.append(path)

    srma_paths_sorted = [
        path for _, _, _, path in sorted(srma_items, key=lambda item: (item[0], item[1], item[2]))
    ]
    return srma_paths_sorted + other_paths, srma_bucket_by_path, srma_bucket_counts


def _normalize_yes_no(value: Any) -> str:
    t = str(value).strip().lower()
    if t == "yes":
        return "yes"
    if t == "no":
        return "no"
    return ""


def classify_output_dir_for_paper_info(json_path: Path) -> Optional[Path]:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    systematic_review = _normalize_yes_no(data.get("systematic_review"))
    meta_analysis = _normalize_yes_no(data.get("meta-analysis"))

    if systematic_review == "yes" and meta_analysis == "no":
        return METADATA_SR_ONLY_DIR
    if systematic_review == "yes" and meta_analysis == "yes":
        return METADATA_SRMA_DIR
    # Includes the explicit ("No","No") skip case and all non-target combinations.
    return None


def included_studies_details_null_ratio(
    json_path: Path,
) -> Tuple[float, int, int]:
    """
    Return (null_ratio, null_count, total_count) for included_studies_details entries
    where either title or authors is null.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log_runtime_error(f"load paper_info for null-ratio check json_path={str(json_path)!r}", e)
        return (0.0, 0, 0)

    details = data.get("included_studies_details")
    if isinstance(details, str) and details.strip().lower() == "inconsistent":
        return (0.0, 0, -1)
    if isinstance(details, dict):
        studies_iter = [v for v in details.values() if isinstance(v, dict)]
    elif isinstance(details, list):
        studies_iter = [v for v in details if isinstance(v, dict)]
    else:
        return (0.0, 0, 0)

    total_count = len(studies_iter)
    if total_count == 0:
        return (0.0, 0, 0)

    null_count = 0
    for study in studies_iter:
        if study.get("title") is None or study.get("authors") is None:
            null_count += 1
    null_ratio = null_count / total_count
    return (null_ratio, null_count, total_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read srma/paper_info/{pmid}.json and run PubMed-only retrieval for included studies. "
            "Write outputs to included_studies/metadata/sr_only or included_studies/metadata/srma. "
            "Skips existing files unless they contain PubMed API errors."
        )
    )
    parser.add_argument("--srma-pmid", type=str, default=None, metavar="PMID")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--api-rps",
        type=float,
        default=None,
        help=(
            "Global PubMed request rate limit. Defaults to "
            f"{DEFAULT_API_RPS_WITH_KEY} rps with NCBI_API_KEY, "
            f"{DEFAULT_API_RPS_WITHOUT_KEY} rps without it."
        ),
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Persistent cache for stable PubMed matches and PMID efetch payloads.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable PubMed metadata cache.")
    parser.add_argument(
        "--cache-save-interval",
        type=float,
        default=DEFAULT_CACHE_SAVE_INTERVAL_SECONDS,
        help=(
            "Minimum seconds between PubMed cache writes during the run. "
            "Set to 0 to save after every processed output file."
        ),
    )
    parser.add_argument(
        "--batch-efetch-threshold",
        type=int,
        default=DEFAULT_BATCH_EFETCH_THRESHOLD,
        help="Use one batched efetch per review when the number of processed studies reaches this threshold.",
    )
    parser.add_argument("--batch-efetch-size", type=int, default=100)
    parser.add_argument(
        "--min-title-similarity",
        type=float,
        default=DEFAULT_MIN_TITLE_SIMILARITY,
        help="Minimum normalized title similarity required to accept a PubMed match.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on an unexpected per-file error instead of logging it and continuing.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=os.getenv("PUBMED_PROXY", DEFAULT_PUBMED_PROXY),
        help=(
            "Proxy used by this PubMed retrieval process only. "
            "Set to an empty string to keep the existing environment proxy."
        ),
    )
    args = parser.parse_args()

    configure_pubmed_proxy(args.proxy)
    delay = max(0.0, args.delay)
    show_progress = not args.no_progress
    concurrency = max(1, int(args.concurrency))
    api_key = (NCBI_API_KEY or "").strip() or None
    api_rps = float(args.api_rps) if args.api_rps is not None else (
        DEFAULT_API_RPS_WITH_KEY if api_key else DEFAULT_API_RPS_WITHOUT_KEY
    )
    configure_rate_limiter(api_rps)
    cache = PubMedMetadataCache(args.cache_path, enabled=not args.no_cache)
    cache_save_interval = max(0.0, float(args.cache_save_interval))
    last_cache_save_at = time.monotonic()

    paths = discover_paper_info_json_paths()
    if args.srma_pmid is not None:
        stem = str(args.srma_pmid).strip()
        paths = [p for p in paths if p.stem == stem]
        if not paths:
            raise SystemExit(f"No paper_info JSON for PMID {stem!r} under {PAPER_INFO_DIR}")
    elif not paths:
        raise SystemExit(f"No *.json under {PAPER_INFO_DIR}")

    paths, srma_bucket_by_path, srma_bucket_counts = order_paper_info_paths(paths)

    METADATA_SR_ONLY_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_SRMA_DIR.mkdir(parents=True, exist_ok=True)
    file_iter = _progress(paths, desc="paper_info_pubmed", enabled=show_progress, unit="file")
    current_srma_bucket: Optional[int] = None
    for i, jp in enumerate(file_iter):
        out_dir = classify_output_dir_for_paper_info(jp)
        if out_dir is None:
            continue
        bucket = srma_bucket_by_path.get(jp)
        bucket_prefix = f"[srma <= {bucket}] " if bucket is not None else ""
        if bucket is not None and bucket != current_srma_bucket:
            bucket_file_count = srma_bucket_counts.get(bucket, 0)
            print(
                f"=== Processing SRMA bucket <= {bucket} included studies "
                f"({bucket_file_count} file(s)) ===",
                flush=True,
            )
            current_srma_bucket = bucket
        null_ratio, _, total_count = included_studies_details_null_ratio(jp)
        if total_count == -1:
            continue
        if total_count > 0 and null_ratio > 0.10:
            continue
        out_path = out_dir / f"{jp.stem}.json"
        existing_included: Dict[str, Any] = {}
        retry_mode = False
        regenerate_existing = False
        if out_path.is_file():
            try:
                with open(out_path, encoding="utf-8") as f:
                    existing_payload = json.load(f)
                if not isinstance(existing_payload, dict):
                    raise ValueError("existing metadata root is not a JSON object")
                existing_included_raw = existing_payload.get("included_studies")
                if not isinstance(existing_included_raw, dict):
                    raise ValueError("existing metadata included_studies is not a JSON object")
                existing_included = existing_included_raw
            except Exception as e:
                log_runtime_error(f"load existing metadata out_path={str(out_path)!r}", e)
                existing_included = {}
                regenerate_existing = True
            failed_keys = [
                k for k, v in existing_included.items()
                if isinstance(v, dict) and _item_has_retryable_api_error(v)
            ]
            if not failed_keys and not regenerate_existing:
                continue
            if failed_keys:
                retry_mode = True
                print(
                    f"{bucket_prefix}Resume {jp.stem}: re-run {len(failed_keys)} errored item(s) "
                    f"in {out_path.resolve()}",
                    flush=True,
                )
            elif regenerate_existing:
                print(
                    f"{bucket_prefix}Regenerate {jp.stem}: existing metadata is unreadable "
                    f"at {out_path.resolve()}",
                    flush=True,
                )

        try:
            payload = process_paper_info_file(
                jp,
                delay_seconds=delay,
                concurrency=concurrency,
                existing_included=existing_included,
                retry_only_errored_items=retry_mode,
                min_title_similarity=float(args.min_title_similarity),
                cache=cache,
                batch_efetch_threshold=max(0, int(args.batch_efetch_threshold)),
                batch_efetch_size=max(1, int(args.batch_efetch_size)),
            )
            now = time.monotonic()
            if cache_save_interval <= 0 or now - last_cache_save_at >= cache_save_interval:
                cache.save()
                last_cache_save_at = now
            atomic_write_json(out_path, payload, indent=2)
        except Exception as e:
            log_runtime_error(f"process paper_info file={str(jp)!r}", e)
            cache.save(force=True)
            if args.stop_on_error:
                raise
            continue
        n = int(payload.get("num_of_included_studies") or len(payload.get("included_studies") or {}))
        if n == 0:
            continue
        if delay > 0 and i < len(paths) - 1:
            time.sleep(delay)
    cache.save(force=True)
