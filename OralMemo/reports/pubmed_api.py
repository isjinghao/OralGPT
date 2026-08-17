#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PubMed search demo using NCBI E-utilities.

Usage:
    python pubmed_search.py "cancer immunotherapy"

Optional:
    export NCBI_API_KEY="your_api_key"
"""

import os
import sys
import time
import re
from datetime import date, timedelta
import argparse
import json
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DB = "pubmed"
NCBI_FETCH_MAX_RETRIES = 8
NCBI_FETCH_RETRY_BASE_DELAY = 1.5
NCBI_DEFAULT_TOOL = "OralMemo"
NCBI_RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)
NCBI_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
    requests.exceptions.SSLError,
)


def _ncbi_identity_params() -> Dict[str, str]:
    params: Dict[str, str] = {"tool": (os.getenv("NCBI_TOOL") or NCBI_DEFAULT_TOOL).strip() or NCBI_DEFAULT_TOOL}
    email = (os.getenv("NCBI_EMAIL") or "").strip()
    if email:
        params["email"] = email
    api_key = (os.getenv("NCBI_API_KEY") or "").strip()
    if api_key:
        params["api_key"] = api_key
    return params


def _with_ncbi_identity(params: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(params)
    for key, value in _ncbi_identity_params().items():
        merged.setdefault(key, value)
    return merged


def _ncbi_user_agent() -> str:
    tool = _ncbi_identity_params().get("tool", NCBI_DEFAULT_TOOL)
    email = _ncbi_identity_params().get("email")
    return f"{tool}/1.0" + (f" ({email})" if email else "")


def _ncbi_requests_extra(*, timeout: float | int) -> Dict[str, Any]:
    """
    Match ``metadata_retrieval.ncbi_fetch`` / ``NCBI_USE_SYSTEM_PROXY``:
    when set to 0/false/no/off, bypass HTTP(S)_PROXY for E-utilities (broken corporate proxies are common).
    """
    extra: Dict[str, Any] = {
        "timeout": timeout,
        "headers": {"User-Agent": _ncbi_user_agent()},
    }
    raw = (os.getenv("NCBI_USE_SYSTEM_PROXY") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        extra["proxies"] = {"http": None, "https": None}
    return extra


_LAST_NCBI_CALL_AT = 0.0


def _throttle_ncbi() -> None:
    """
    Global pacing for every E-utilities call to avoid NCBI's
    "WWW Error Blocked Diagnostic" block page. Paging / date-split loops issue
    many requests, so throttling must live at the request layer, not only in main().
    ~3 rps without an API key, ~9 rps with NCBI_API_KEY.
    """
    global _LAST_NCBI_CALL_AT
    min_interval = 0.11 if (os.getenv("NCBI_API_KEY") or "").strip() else 0.34
    now = time.monotonic()
    wait = _LAST_NCBI_CALL_AT + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _LAST_NCBI_CALL_AT = time.monotonic()


def _ncbi_get_with_retry(url: str, *, params: Dict[str, Any], timeout: float | int) -> requests.Response:
    last_err: Exception | None = None
    request_params = _with_ncbi_identity(params)
    for attempt in range(1, NCBI_FETCH_MAX_RETRIES + 1):
        try:
            _throttle_ncbi()
            resp = requests.get(url, params=request_params, **_ncbi_requests_extra(timeout=timeout))
            if resp.status_code in NCBI_RETRY_STATUS_CODES:
                if attempt >= NCBI_FETCH_MAX_RETRIES:
                    resp.raise_for_status()
                sleep_s = NCBI_FETCH_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp
        except NCBI_TRANSIENT_EXCEPTIONS as e:
            last_err = e
            if attempt >= NCBI_FETCH_MAX_RETRIES:
                break
            sleep_s = NCBI_FETCH_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"NCBI GET retry {attempt}/{NCBI_FETCH_MAX_RETRIES} after transient error: {e}", file=sys.stderr)
            time.sleep(sleep_s)
    if last_err is not None:
        raise last_err
    raise RuntimeError("NCBI GET failed without a captured exception")


def _ncbi_post_with_retry(url: str, *, data: Dict[str, Any], timeout: float | int) -> requests.Response:
    last_err: Exception | None = None
    request_data = _with_ncbi_identity(data)
    for attempt in range(1, NCBI_FETCH_MAX_RETRIES + 1):
        try:
            _throttle_ncbi()
            resp = requests.post(url, data=request_data, **_ncbi_requests_extra(timeout=timeout))
            if resp.status_code in NCBI_RETRY_STATUS_CODES:
                if attempt >= NCBI_FETCH_MAX_RETRIES:
                    resp.raise_for_status()
                sleep_s = NCBI_FETCH_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp
        except NCBI_TRANSIENT_EXCEPTIONS as e:
            last_err = e
            if attempt >= NCBI_FETCH_MAX_RETRIES:
                break
            sleep_s = NCBI_FETCH_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"NCBI POST retry {attempt}/{NCBI_FETCH_MAX_RETRIES} after transient error: {e}", file=sys.stderr)
            time.sleep(sleep_s)
    if last_err is not None:
        raise last_err
    raise RuntimeError("NCBI POST failed without a captured exception")


def _safe_json_from_response(resp: requests.Response) -> Dict[str, Any]:
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("NCBI response JSON is not an object.")
    except ValueError:
        text = resp.text or ""
        # Some upstream responses contain raw control chars; strip them and retry.
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            preview = cleaned[:300].replace("\n", "\\n")
            raise ValueError(
                f"Failed to parse NCBI JSON response after cleaning control chars: {e}. "
                f"Response preview: {preview}"
            ) from e
        if not isinstance(parsed, dict):
            raise ValueError("NCBI response JSON is not an object after cleaning.")
        return parsed


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def esearch(
    term: str,
    retstart: int = 0,
    retmax: int = 10,
    api_key: str | None = None,
    usehistory: bool = False,
) -> Dict[str, Any]:
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": DB,
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "retstart": retstart,
        "sort": "relevance",
    }
    if usehistory:
        params["usehistory"] = "y"
    if api_key:
        params["api_key"] = api_key

    # Use POST to avoid 414 URI Too Long for large boolean queries.
    resp = _ncbi_post_with_retry(url, data=params, timeout=30)
    return _safe_json_from_response(resp).get("esearchresult", {})


def esearch_all_pmids(
    term: str,
    *,
    page_size: int = 10,
    max_records: int | None = None,
    api_key: str | None = None,
) -> List[str]:
    """
    Fetch as many PMIDs as possible for a term using server-side history and paging.
    """
    first = esearch(term=term, retstart=0, retmax=min(page_size, 9999), api_key=api_key, usehistory=True)
    count = int(first.get("count", "0") or 0)
    if count == 0:
        return []

    # PubMed ESearch can only page first 9,999 records for one query.
    # For broad queries, split by publication-date windows and merge.
    if count > 9999:
        pmids = _esearch_all_pmids_split_by_pdat(
            base_term=term,
            page_size=page_size,
            max_records=max_records,
            api_key=api_key,
        )
        return _dedup_preserve_order(pmids)

    webenv = first.get("webenv")
    query_key = first.get("querykey")
    if not webenv or not query_key:
        # Fallback: use the first page's idlist only
        return list(first.get("idlist", []))

    want = count if max_records is None else min(count, max_records)
    pmids: List[str] = []

    retstart = 0
    while retstart < want:
        batch_retmax = min(page_size, want - retstart)
        url = f"{EUTILS_BASE}/esearch.fcgi"
        params = {
            "db": DB,
            "term": term,
            "retmode": "json",
            "retstart": retstart,
            "retmax": batch_retmax,
            "sort": "relevance",
            "query_key": query_key,
            "WebEnv": webenv,
        }
        if api_key:
            params["api_key"] = api_key
        # Use POST to avoid 414 URI Too Long for large boolean queries.
        resp = _ncbi_post_with_retry(url, data=params, timeout=30)
        data = _safe_json_from_response(resp).get("esearchresult", {})
        pmids.extend(data.get("idlist", []))
        retstart += batch_retmax

    # Keep order but de-dup just in case
    return _dedup_preserve_order(pmids)


def _dedup_preserve_order(pmids: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for pid in pmids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _pdat_clause(start: date, end: date) -> str:
    return (
        f'("{start.strftime("%Y/%m/%d")}"[Date - Publication] : '
        f'"{end.strftime("%Y/%m/%d")}"[Date - Publication])'
    )


def _date_midpoint(start: date, end: date) -> date:
    return start + timedelta(days=(end - start).days // 2)


def _esearch_count(term: str, api_key: str | None) -> int:
    data = esearch(term=term, retstart=0, retmax=0, api_key=api_key, usehistory=False)
    return int(data.get("count", "0") or 0)


def _esearch_ids_for_term(
    term: str,
    *,
    page_size: int,
    api_key: str | None,
    max_records: int | None,
) -> List[str]:
    first = esearch(
        term=term,
        retstart=0,
        retmax=min(page_size, 9999),
        api_key=api_key,
        usehistory=True,
    )
    count = int(first.get("count", "0") or 0)
    if count == 0:
        return []

    webenv = first.get("webenv")
    query_key = first.get("querykey")
    if not webenv or not query_key:
        return list(first.get("idlist", []))

    want = count if max_records is None else min(count, max_records)
    pmids: List[str] = []
    retstart = 0
    step = min(page_size, 9999)
    while retstart < want:
        batch_retmax = min(step, want - retstart)
        url = f"{EUTILS_BASE}/esearch.fcgi"
        params = {
            "db": DB,
            "term": term,
            "retmode": "json",
            "retstart": retstart,
            "retmax": batch_retmax,
            "sort": "relevance",
            "query_key": query_key,
            "WebEnv": webenv,
        }
        if api_key:
            params["api_key"] = api_key
        resp = _ncbi_post_with_retry(url, data=params, timeout=30)
        data = _safe_json_from_response(resp).get("esearchresult", {})
        pmids.extend(data.get("idlist", []))
        retstart += batch_retmax
    return pmids


def _esearch_all_pmids_split_by_pdat(
    *,
    base_term: str,
    page_size: int,
    max_records: int | None,
    api_key: str | None,
) -> List[str]:
    start = date(1900, 1, 1)
    end = date.today()

    def collect_range(range_start: date, range_end: date, remaining: int | None) -> List[str]:
        if remaining is not None and remaining <= 0:
            return []

        term = f"({base_term}) AND {_pdat_clause(range_start, range_end)}"
        count = _esearch_count(term, api_key=api_key)
        if count == 0:
            return []

        if count <= 9999:
            local_max = None if remaining is None else remaining
            return _esearch_ids_for_term(
                term,
                page_size=page_size,
                api_key=api_key,
                max_records=local_max,
            )

        if range_start >= range_end:
            # Cannot split further by date. Return first retrievable chunk.
            local_max = None if remaining is None else min(remaining, 9999)
            return _esearch_ids_for_term(
                term,
                page_size=min(page_size, 9999),
                api_key=api_key,
                max_records=local_max,
            )

        mid = _date_midpoint(range_start, range_end)
        left = collect_range(range_start, mid, remaining)
        remaining_after_left = None if remaining is None else max(0, remaining - len(left))
        right_start = mid + timedelta(days=1)
        right = collect_range(right_start, range_end, remaining_after_left)
        return left + right

    return collect_range(start, end, max_records)


def esummary(
    pmids: List[str],
    api_key: str | None = None,
    batch_size: int = 100,
    show_progress: bool = False,
) -> Dict[str, Any]:
    if not pmids:
        return {}

    url = f"{EUTILS_BASE}/esummary.fcgi"
    merged_result: Dict[str, Any] = {"uids": []}
    header: Dict[str, Any] = {}
    batch_size = max(1, batch_size)
    total_batches = (len(pmids) + batch_size - 1) // batch_size

    for i in range(0, len(pmids), batch_size):
        chunk = pmids[i : i + batch_size]
        if show_progress:
            batch_no = i // batch_size + 1
            print(f"Fetching esummary batch {batch_no}/{total_batches} ({len(chunk)} PMIDs)", file=sys.stderr, flush=True)

        params = {
            "db": DB,
            "id": ",".join(chunk),
            "retmode": "json",
        }
        if api_key:
            params["api_key"] = api_key

        resp = _ncbi_post_with_retry(url, data=params, timeout=60)
        data = _safe_json_from_response(resp)
        header = data.get("header", header)
        result = data.get("result", {})
        uids = result.get("uids", [])
        merged_result["uids"].extend(uids)
        for uid in uids:
            if uid in result:
                merged_result[uid] = result[uid]

    return {"header": header, "result": merged_result}


def _first_sentence(text: str) -> str:
    s = " ".join((text or "").split())
    if not s:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)
    return parts[0].strip()


def efetch_abstract_first_sentences(
    pmids: List[str],
    *,
    api_key: str | None = None,
    batch_size: int = 10,
) -> Dict[str, str]:
    """
    Return PMID -> first sentence of abstract (or empty string if none).
    Uses efetch retmode=xml and parses AbstractText.
    """
    out: Dict[str, str] = {}
    if not pmids:
        return out

    url = f"{EUTILS_BASE}/efetch.fcgi"
    for i in range(0, len(pmids), batch_size):
        chunk = pmids[i : i + batch_size]
        params = {"db": DB, "id": ",".join(chunk), "retmode": "xml"}
        if api_key:
            params["api_key"] = api_key
        resp = _ncbi_post_with_retry(url, data=params, timeout=60)

        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//MedlineCitation/PMID")
            pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
            if not pmid:
                continue

            abs_texts: List[str] = []
            for abs_el in article.findall(".//Article/Abstract/AbstractText"):
                # AbstractText may contain nested tags; itertext() collects all
                abs_texts.append("".join(abs_el.itertext()).strip())
            abstract = " ".join([t for t in abs_texts if t])
            out[pmid] = _first_sentence(abstract)

    return out


def efetch_full_articles(
    pmids: List[str],
    *,
    api_key: str | None = None,
    batch_size: int = 10,
    show_progress: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Return full per-article data parsed from efetch XML:
    PMID -> {"abstract", "abstract_first_sentence", "mesh_terms", "keywords"}.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not pmids:
        return out

    url = f"{EUTILS_BASE}/efetch.fcgi"
    total_batches = (len(pmids) + batch_size - 1) // batch_size
    for i in range(0, len(pmids), batch_size):
        chunk = pmids[i : i + batch_size]
        if show_progress:
            batch_no = i // batch_size + 1
            print(f"Fetching efetch batch {batch_no}/{total_batches} ({len(chunk)} PMIDs)", file=sys.stderr, flush=True)
        params = {"db": DB, "id": ",".join(chunk), "retmode": "xml"}
        if api_key:
            params["api_key"] = api_key
        resp = _ncbi_post_with_retry(url, data=params, timeout=60)

        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//MedlineCitation/PMID")
            pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
            if not pmid:
                continue

            abs_texts: List[str] = []
            for abs_el in article.findall(".//Article/Abstract/AbstractText"):
                abs_texts.append("".join(abs_el.itertext()).strip())
            abstract = " ".join([t for t in abs_texts if t]).strip()

            mesh_terms: List[str] = []
            for mesh in article.findall(".//MeshHeadingList/MeshHeading"):
                desc = mesh.find("DescriptorName")
                if desc is not None and desc.text:
                    mesh_terms.append(desc.text.strip())

            keywords: List[str] = []
            for kw in article.findall(".//KeywordList/Keyword"):
                text = "".join(kw.itertext()).strip()
                if text:
                    keywords.append(text)

            out[pmid] = {
                "abstract": abstract,
                "abstract_first_sentence": _first_sentence(abstract),
                "mesh_terms": mesh_terms,
                "keywords": keywords,
            }
    return out


def _sleep_for_rate(api_key: str | None) -> None:
    # Conservative E-utilities pacing: ~3 rps without key, ~10 rps with key.
    time.sleep(0.34 if not api_key else 0.11)


def probe_max_rate_no_key(
    *,
    seconds_per_step: float = 4.0,
    start_rps: float = 1.0,
    max_rps: float = 6.0,
    step_rps: float = 0.5,
    term: str = "cancer",
) -> float:
    """
    Best-effort rate probe without API key.
    Sends lightweight esearch requests and increases RPS until errors appear.
    Returns the highest RPS that completed a full step without HTTP errors.
    """
    url = f"{EUTILS_BASE}/esearch.fcgi"

    best = 0.0
    rps = start_rps
    while rps <= max_rps + 1e-9:
        interval = 1.0 / rps
        deadline = time.monotonic() + seconds_per_step
        ok = 0
        try:
            while time.monotonic() < deadline:
                t0 = time.monotonic()
                params = {"db": DB, "term": term, "retmode": "json", "retmax": 1, "retstart": 0, "sort": "relevance"}
                _ncbi_get_with_retry(url, params=params, timeout=30)
                ok += 1
                dt = time.monotonic() - t0
                sleep_left = interval - dt
                if sleep_left > 0:
                    time.sleep(sleep_left)
        except requests.HTTPError:
            break
        except requests.RequestException:
            break

        if ok > 0:
            best = rps
        rps += step_rps

    return best


def format_authors(article: Dict[str, Any], max_authors: int = 5) -> str:
    authors = article.get("authors", [])
    names = [a.get("name", "") for a in authors if a.get("name")]
    if not names:
        return "N/A"
    if len(names) <= max_authors:
        return ", ".join(names)
    return ", ".join(names[:max_authors]) + " et al."


def main() -> None:
    parser = argparse.ArgumentParser(description="PubMed search demo using NCBI E-utilities.")
    parser.add_argument("term", nargs="?", help='Search term, e.g. "cancer immunotherapy"')
    parser.add_argument("--all", action="store_true", help="Fetch as many PMIDs as possible (paged via History).")
    parser.add_argument("--max-records", type=int, default=None, help="Cap total records when using --all.")
    parser.add_argument("--page-size", type=int, default=100, help="PMID page size for esearch paging.")
    parser.add_argument("--abstract-batch-size", type=int, default=100, help="Batch size for efetch abstract retrieval.")
    parser.add_argument("--probe-rate-no-key", action="store_true", help="Probe best-effort max RPS without API key.")
    parser.add_argument("--api-key", default=None, help="NCBI API key. Prefer NCBI_API_KEY env var for repeated use.")
    parser.add_argument("--email", default=None, help="Email passed to NCBI E-utilities. Or set NCBI_EMAIL.")
    parser.add_argument("--count-only", action="store_true", help="Only print the PubMed hit count for the built-in query.")
    parser.add_argument("--quiet", action="store_true", help="Write JSON without printing every article to stdout.")
    parser.add_argument("--output", default="pubmed_api.json", help="Output JSON file path.")
    args = parser.parse_args()

    load_dotenv(".env")
    if args.api_key:
        os.environ["NCBI_API_KEY"] = args.api_key.strip()
    if args.email:
        os.environ["NCBI_EMAIL"] = args.email.strip()
    api_key = (os.getenv("NCBI_API_KEY") or "").strip() or None
    print(f"NCBI API key: {'enabled' if api_key else 'not set'}", file=sys.stderr)

    if args.probe_rate_no_key:
        best = probe_max_rate_no_key()
        print(f"Best-effort max RPS without api_key (probe): {best:.1f}")
        return

    # Expanded dental / maxillofacial disease case-report search.
    # Scope: oral/dental/maxillofacial case reports, English, humans, with abstracts,
    # since 2010. The journal whitelist intentionally mixes high-impact dental journals,
    # specialty case-report-friendly journals, and open-access/PMC-friendly journals to
    # increase downloadable full texts while keeping the topic clinically relevant.
    term = '''
        (
    "Stomatognathic Diseases"[Mesh]
    OR "Tooth Diseases"[Mesh]
    OR "Mouth Diseases"[Mesh]
    OR "Jaw Diseases"[Mesh]
    OR "Maxillofacial Abnormalities"[Mesh]
    OR dental[tiab]
    OR dentistry[tiab]
    OR odontogenic[tiab]
    OR "oral cavity"[tiab]
    OR maxillofacial[tiab]
    OR oromaxillofacial[tiab]
    OR stomatognathic[tiab]
    OR periodontal[tiab]
    OR endodontic[tiab]
    OR prosthodontic[tiab]
    OR orthodontic[tiab]
    OR peri-implant[tiab]
    OR periimplant[tiab]
    OR mandibular[tiab]
    OR maxillary[tiab]
    OR oral[tiab]
    )
    AND "Case Reports"[Publication Type]
    AND English[Language]
    AND Humans[Mesh]
    AND hasabstract
    AND ("2010/01/01"[Date - Publication] : "3000"[Date - Publication])
    AND (
    "Periodontology 2000"[Journal]
    OR "Journal of Dental Research"[Journal]
    OR "Journal of Clinical Periodontology"[Journal]
    OR "Clinical Oral Implants Research"[Journal]
    OR "Dental Materials"[Journal]
    OR "Journal of Endodontics"[Journal]
    OR "International Endodontic Journal"[Journal]
    OR "Oral Oncology"[Journal]
    OR "International Journal of Oral Science"[Journal]
    OR "Journal of Dentistry"[Journal]
    OR "Clinical Implant Dentistry and Related Research"[Journal]
    OR "Journal of Prosthodontic Research"[Journal]
    OR "Clinical Oral Investigations"[Journal]
    OR "BMC Oral Health"[Journal]
    OR "Head & Face Medicine"[Journal]
    OR "Journal of Oral and Maxillofacial Surgery"[Journal]
    OR "International Journal of Oral and Maxillofacial Surgery"[Journal]
    OR "British Journal of Oral and Maxillofacial Surgery"[Journal]
    OR "Journal of Cranio-Maxillo-Facial Surgery"[Journal]
    OR "Maxillofacial Plastic and Reconstructive Surgery"[Journal]
    OR "Journal of the Korean Association of Oral and Maxillofacial Surgeons"[Journal]
    OR "Oral Diseases"[Journal]
    OR "Oral Surgery, Oral Medicine, Oral Pathology and Oral Radiology"[Journal]
    OR "Journal of Oral Pathology & Medicine"[Journal]
    OR "Journal of Oral and Maxillofacial Pathology"[Journal]
    OR "Head and Neck Pathology"[Journal]
    OR "Medicina Oral, Patologia Oral y Cirugia Bucal"[Journal]
    OR "European Journal of Dentistry"[Journal]
    OR "Journal of Applied Oral Science"[Journal]
    OR "Australian Dental Journal"[Journal]
    OR "Special Care in Dentistry"[Journal]
    OR "Dental Traumatology"[Journal]
    OR "International Journal of Paediatric Dentistry"[Journal]
    OR "European Archives of Paediatric Dentistry"[Journal]
    OR "Pediatric Dentistry"[Journal]
    OR "The Angle Orthodontist"[Journal]
    OR "American Journal of Orthodontics and Dentofacial Orthopedics"[Journal]
    OR "European Journal of Orthodontics"[Journal]
    OR "Orthodontics & Craniofacial Research"[Journal]
    OR "Journal of Periodontology"[Journal]
    OR "Clinical Advances in Periodontics"[Journal]
    OR "The International Journal of Periodontics & Restorative Dentistry"[Journal]
    OR "The International Journal of Oral & Maxillofacial Implants"[Journal]
    OR "Journal of Prosthetic Dentistry"[Journal]
    OR "The International Journal of Prosthodontics"[Journal]
    OR "Journal of Esthetic and Restorative Dentistry"[Journal]
    OR "Operative Dentistry"[Journal]
    OR "Quintessence International"[Journal]
    OR "Gerodontology"[Journal]
    OR "Odontology"[Journal]
    OR "Case Reports in Dentistry"[Journal]
    OR "International Journal of Dentistry"[Journal]
    OR "Journal of Dental Sciences"[Journal]
    OR "BDJ Open"[Journal]
    OR "Frontiers in Oral Health"[Journal]
    OR "Dentistry Journal"[Journal]
    OR "Clinical, Cosmetic and Investigational Dentistry"[Journal]
    OR "Journal of Clinical and Experimental Dentistry"[Journal]
    OR "Contemporary Clinical Dentistry"[Journal]
    OR "Journal of Conservative Dentistry"[Journal]
    OR "Restorative Dentistry & Endodontics"[Journal]
    OR "European Endodontic Journal"[Journal]
    OR "Saudi Dental Journal"[Journal]
    OR "Dental Research Journal"[Journal]
    OR "Imaging Science in Dentistry"[Journal]
    OR "Journal of Oral Biology and Craniofacial Research"[Journal]
    OR "Clinical Case Reports"[Journal]
    OR "BMJ Case Reports"[Journal]
    OR "Medicine"[Journal]
    OR "Cureus"[Journal]
    OR "Heliyon"[Journal]
    OR "Case Reports in Otolaryngology"[Journal]
    )
    '''
    try:
        if args.count_only:
            count = _esearch_count(term, api_key=api_key)
            print(f"PubMed count: {count}")
            return

        if args.all:
            pmids = esearch_all_pmids(
                term=term,
                page_size=max(1, args.page_size),
                max_records=args.max_records,
                api_key=api_key,
            )
        else:
            pmids = list(esearch(term=term, retstart=0, retmax=min(args.page_size, 10), api_key=api_key).get("idlist", []))

        print(f"Fetched PMID count: {len(pmids)}", file=sys.stderr, flush=True)

        # avoid requesting too fast (and keep a gap between different endpoints)
        _sleep_for_rate(api_key)

        full_articles = efetch_full_articles(
            pmids,
            api_key=api_key,
            batch_size=max(1, args.abstract_batch_size),
            show_progress=not args.quiet,
        )

        _sleep_for_rate(api_key)
        print("Fetching PubMed summaries...", file=sys.stderr, flush=True)

        summary = esummary(
            pmids,
            api_key=api_key,
            batch_size=max(1, args.abstract_batch_size),
            show_progress=not args.quiet,
        )
        result = summary.get("result", {})
        uids = result.get("uids", [])

        if not uids:
            print("No results found.")
            return

        output_articles: List[Dict[str, Any]] = []

        for i, uid in enumerate(uids, start=1):
            article = result.get(uid, {})
            title = article.get("title", "N/A")
            pubdate = article.get("pubdate", "N/A")
            source = article.get("source", "N/A")
            authors = format_authors(article)
            doi = "N/A"
            efetch_data = full_articles.get(uid, {})
            abstract_full = efetch_data.get("abstract", "")
            abs_first = efetch_data.get("abstract_first_sentence", "")

            article_ids = article.get("articleids", [])
            for item in article_ids:
                if item.get("idtype") == "doi":
                    doi = item.get("value", "N/A")
                    break

            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"

            output_articles.append(
                {
                    "index": i,
                    "pmid": uid,
                    "title": title,
                    "authors": [a.get("name", "") for a in article.get("authors", []) if a.get("name")],
                    "authors_display": authors,
                    "journal": source,
                    "pubdate": pubdate,
                    "doi": doi,
                    "pubmed_url": pubmed_url,
                    "abstract": abstract_full,
                    "abstract_first_sentence": abs_first,
                    "mesh_terms": efetch_data.get("mesh_terms", []),
                    "keywords": efetch_data.get("keywords", []),
                    "esummary_raw": article,
                }
            )

            if not args.quiet:
                print(f"[{i}] PMID: {uid}")
                print(f"Title   : {title}")
                print(f"Authors : {authors}")
                print(f"Journal : {source}")
                print(f"PubDate : {pubdate}")
                print(f"DOI     : {doi}")
                print(f"Abstract: {abs_first if abs_first else 'N/A'}")
                print(f"URL     : {pubmed_url}")
                print("-" * 80)

        payload = {
            "query": {
                "term": term,
                "all": args.all,
                "max_records": args.max_records,
                "page_size": args.page_size,
                "abstract_batch_size": args.abstract_batch_size,
            },
            "api_key_used": bool(api_key),
            "record_count": len(output_articles),
            "articles": output_articles,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved full records JSON to: {args.output}")

    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        sys.exit(2)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(3)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()
