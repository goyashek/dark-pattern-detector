"""
scrape_real.py — Collect REAL Indian-context dark-pattern UI copy with Playwright.

WHY THIS EXISTS
---------------
The training corpus is ~68% template-generated (src/collect_data.py); the rare CCPA
classes are 90-100% synthetic. There is no public labelled corpus for India's 13 CCPA
categories, so to honestly test generalisation we collect REAL UI strings from live
Indian e-commerce / travel / food / fintech sites and use them as a held-out
out-of-distribution (OOD) test set.

WHAT IT DOES
------------
For each curated URL:
  1. Navigate with a real Chrome profile, let the page settle, capture any popup
     (popups are themselves dark patterns — e.g. countdown timers), then screenshot.
  2. Extract short UI text *blocks* using element-level innerText (NOT raw text-node
     walking — timers like "11h 36m 9s" and "₹3,698 + ₹185 taxes & fees" are split
     across child nodes and only reassemble at the element level).
  3. Auto-SUGGEST a CCPA label per block using high-precision regex lexicons.
  4. Drop benign false-positives, de-duplicate by a number-normalised key (so 40
     near-identical "+ ₹N taxes & fees" strings collapse to one representative).
  5. Append rows to data/raw/scraped_candidates.csv:  text, url, site, auto_label,
     all_hits, screenshot.

This auto-label is a SUGGESTION ONLY. A human confirm/correct pass (the next step in
the pipeline) is mandatory before any row is promoted to the OOD test set.

Covers the "easy" classes reachable without login (False Urgency, Drip Pricing,
Disguised Advertisement, Interface Interference, Nagging, Forced Action, plus
scarcity which maps to False Urgency). Login-gated classes (Subscription Trap, SaaS
Billing, Basket Sneaking) are collected separately by hand-driving the MCP browser.

Run:  python -m src.scrape_real            # default URL list, headed
      python -m src.scrape_real --headless
      python -m src.scrape_real --urls my_urls.txt
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(HERE, "data", "raw", "scraped_candidates.csv")
SHOTS_DIR = os.path.join(HERE, "data", "raw", "screenshots")

# --------------------------------------------------------------------------- #
# Curated seed URLs. Listing / detail / category pages where dark patterns
# render WITHOUT login. Edit freely; --urls can override entirely.
# --------------------------------------------------------------------------- #
SEED_URLS = [
    # --- Travel: urgency timers, scarcity (rooms left), drip pricing, sponsored ---
    ("makemytrip", "https://www.makemytrip.com/hotels/hotel-listing/?checkin=date_today&city=CTGOI&checkout=date_tomorrow&roomStayQualifier=2e0e&locusId=CTGOI&country=IN&locusType=city&searchText=Goa"),
    ("makemytrip", "https://www.makemytrip.com/hotels/hotel-listing/?city=CTXBLR&country=IN&checkin=date_today&checkout=date_tomorrow&roomStayQualifier=2e0e&searchText=Bengaluru"),
    ("goibibo",    "https://www.goibibo.com/hotels/hotels-in-goa-ct/"),
    # --- E-commerce: scarcity, urgency, sponsored, discount framing ---
    ("flipkart",   "https://www.flipkart.com/search?q=wireless+earbuds"),
    ("flipkart",   "https://www.flipkart.com/search?q=running+shoes"),
    ("myntra",     "https://www.myntra.com/sportswear"),
    ("amazon_in",  "https://www.amazon.in/s?k=power+bank"),
    ("amazon_in",  "https://www.amazon.in/s?k=bluetooth+speaker"),
    # --- Food / quick-commerce: urgency, 'X people ordering', delivery fees, nag popups ---
    ("swiggy",     "https://www.swiggy.com/restaurants"),
    ("zomato",     "https://www.zomato.com/bangalore/delivery"),
    ("blinkit",    "https://blinkit.com/"),
    # NOTE: Meesho + EaseMyTrip dropped — JS/bot-walled, returned ~0 usable blocks.
    # Login/checkout-gated classes (Subscription Trap, SaaS Billing, Basket Sneaking)
    # are hand-collected via the MCP browser, not this passive script.
]

# --------------------------------------------------------------------------- #
# High-precision signal lexicons -> provisional CCPA category.
# Tightened vs. src/features.py after the trial showed bare "% off" / "Per Night"
# over-firing. Each block can match several; we keep all hits and pick a primary.
# --------------------------------------------------------------------------- #
SIGNALS = {
    "False Urgency": [
        r"\bends? in\b", r"\d+\s*h\s*:?\s*\d+\s*m", r"\d+\s*hrs?\b.*\bleft\b",
        r"hurry", r"last chance", r"act now", r"expires?\b", r"today only",
        r"limited period", r"deal ends", r"sale ends", r"offer ends",
        r"book now,? before", r"deal of the day", r"flash sale", r"only today",
        # scarcity (CCPA folds scarcity into False Urgency)
        r"only \d+ (left|rooms?|seats?|items?)", r"\d+ (rooms?|seats?|items?) left",
        r"last \d+ (rooms?|left)", r"few (rooms?|left|seats?)", r"sold out soon",
        r"selling fast", r"filling fast", r"almost (gone|sold|full)", r"in high demand",
    ],
    "Drip Pricing": [
        r"taxes? (&|and) fees", r"\+\s*₹\s*\d[\d,]*\s*tax", r"\bconvenience fee\b",
        r"\bservice (charge|fee)\b", r"\bhandling (charge|fee)\b", r"\bbooking fee\b",
        r"\bplatform fee\b", r"excl.*tax", r"\+ ₹\d", r"extra charges? at",
        r"delivery (fee|charge) of", r"gst extra", r"fees? applicable",
    ],
    "Disguised Advertisement": [
        r"\bsponsored\b", r"\bpromoted\b", r"\badvertisement\b", r"\bad\b\s*$",
        r"featured listing", r"paid partnership",
    ],
    "Nagging": [
        r"allow notifications", r"enable notifications", r"turn on notifications",
        r"download (the )?app", r"get the app", r"open in app", r"subscribe to",
        r"sign ?up for( our)? newsletter", r"join .* rewards", r"add to home screen",
        r"remind me later", r"maybe later",
    ],
    "Forced Action": [
        r"login to (book|continue|view|see|proceed|unlock|check)",
        r"sign ?in to (continue|view|see|proceed|book)",
        r"create (an )?account to", r"register to (continue|view|proceed)",
        r"share to (unlock|continue|download)", r"refer .* to (unlock|get)",
        r"verify .* to (continue|proceed)",
    ],
    "Interface Interference": [
        r"no thanks,? i", r"skip (this|for now)", r"\bno,? continue without\b",
        r"pre-?selected", r"pre-?checked", r"by default", r"recommended for you\b",
    ],
}
COMPILED = {cat: [re.compile(p, re.I) for p in pats] for cat, pats in SIGNALS.items()}

# Phrases that LOOK like a signal but are benign chrome / filter labels.
# Killed the trial's "Price Per Night" -> drip and "Popularity" -> social-proof noise.
BENIGN_DENY = [
    re.compile(p, re.I) for p in [
        r"^price per night$", r"^per night$", r"^popularity$", r"^price$",
        r"^sort by", r"^filter", r"^price \(low to high\)", r"^price \(high to low\)",
        r"^free cancellation$", r"^breakfast included$", r"^view details$",
        r"^\d+ properties", r"^showing \d+", r"^home$", r"^hotels$",
    ]
]


def normalise_key(text):
    """Number/currency-insensitive key so template siblings collapse in dedup."""
    t = text.lower()
    t = re.sub(r"(₹|\$|rs\.?)\s*[\d,]+(\.\d+)?", "#money", t)  # ₹, $, and "Rs." prices
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^a-z#& ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def signal_key(text, hits):
    """Dedup key built from ONLY the matched signal spans, so product-name siblings
    ('Nike ... Only Few Left!' vs 'Puma ... Only Few Left!') collapse to one row."""
    spans = []
    for cat in hits:
        for rx in COMPILED.get(cat, []):
            m = rx.search(text)
            if m:
                spans.append(m.group(0).lower())
    spans = sorted(set(spans))
    key = " ".join(spans) if spans else text.lower()
    return re.sub(r"\d+", "#", key)


def is_benign(text):
    return any(rx.search(text) for rx in BENIGN_DENY)


def label_block(text):
    """Return (primary_label, [all_hit_labels]) or (None, []) if no signal."""
    hits = [cat for cat, rxs in COMPILED.items() if any(rx.search(text) for rx in rxs)]
    if not hits:
        return None, []
    # primary = the most "specific" hit; prefer non-discount signals in priority order
    priority = ["Forced Action", "Drip Pricing", "Disguised Advertisement",
                "False Urgency", "Nagging", "Interface Interference"]
    primary = next((c for c in priority if c in hits), hits[0])
    return primary, hits


# JS injected into each page. Element-level innerText of small, leaf-ish, visible
# blocks — the method that worked in the trial where text-node walking failed.
EXTRACT_JS = r"""
() => {
  const blocks = [];
  const seen = new Set();
  document.querySelectorAll('div,span,p,li,button,a,section,label,h1,h2,h3,h4').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || +st.opacity === 0) return;
    const t = (el.innerText || '').trim().replace(/\s+/g, ' ');
    if (t.length < 6 || t.length > 180) return;
    if (el.querySelectorAll('div,span,p,li').length > 4) return;  // leaf-ish only
    const key = t.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    blocks.push(t);
  });
  return blocks;
}
"""


def scrape(urls, headless, profile_dir, settle):
    from playwright.sync_api import sync_playwright

    os.makedirs(SHOTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    seen_keys = set()

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            channel="chrome",
            headless=headless,
            viewport={"width": 1280, "height": 900},
            locale="en-IN",
        )
        page = ctx.new_page()

        for site, url in urls:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            print(f"\n[{site}] {url}")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                print(f"  !! navigation failed: {e}")
                continue
            time.sleep(settle)  # let popups render
            # scroll in steps to trigger lazy-loaded cards (Amazon/Meesho ship an
            # almost-empty initial DOM otherwise), then return to top for the shot.
            try:
                for frac in (0.25, 0.5, 0.75, 1.0):
                    page.evaluate("f => window.scrollTo(0, document.body.scrollHeight * f)", frac)
                    time.sleep(1.2)
                page.evaluate("() => window.scrollTo(0, 0)")
                time.sleep(0.5)
            except Exception:
                pass

            shot = os.path.join(SHOTS_DIR, f"{site}-{stamp}.jpeg")
            try:
                page.screenshot(path=shot, type="jpeg", quality=80)
            except Exception:
                shot = ""

            try:
                blocks = page.evaluate(EXTRACT_JS)
            except Exception as e:
                print(f"  !! extract failed: {e}")
                continue

            # sort longest-first so a fuller block ("₹3,698 + ₹185 taxes & fees")
            # is kept before its fragment ("+ ₹185 taxes & fees"); the fragment is
            # then dropped as a substring of an already-kept block from this page.
            page_kept_texts = []
            kept = 0
            for t in sorted(blocks, key=len, reverse=True):
                if is_benign(t):
                    continue
                primary, hits = label_block(t)
                if primary is None:
                    continue
                # dedup on the matched signal spans, not the whole string, so
                # 'Nike ... Only Few Left!' and 'Puma ... Only Few Left!' collapse.
                key = (site, signal_key(t, hits))
                if key in seen_keys:
                    continue
                tl = t.lower()
                if any(tl in kt or kt in tl for kt in page_kept_texts):
                    continue  # nested fragment of another kept block on this page
                seen_keys.add(key)
                page_kept_texts.append(tl)
                rows.append({
                    "text": t,
                    "url": url,
                    "site": site,
                    "auto_label": primary,
                    "all_hits": "|".join(hits),
                    "screenshot": os.path.relpath(shot, HERE) if shot else "",
                    "verified_label": "",   # human fills this in during review
                })
                kept += 1
            print(f"  blocks={len(blocks)}  kept(after dedup+filter)={kept}")

        ctx.close()

    # append-or-create so repeated runs accumulate
    file_exists = os.path.exists(OUT_CSV)
    fieldnames = ["text", "url", "site", "auto_label", "all_hits",
                  "screenshot", "verified_label"]
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} candidate rows -> {OUT_CSV}")
    print("Auto-label distribution:")
    from collections import Counter
    for lab, n in Counter(r["auto_label"] for r in rows).most_common():
        print(f"  {lab:26} {n}")
    print(f"\nScreenshots in {SHOTS_DIR}")
    print("NEXT: human review pass — fill the `verified_label` column "
          "(blank/`drop` to discard a row).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="run without a visible window")
    ap.add_argument("--urls", help="text file of `site,url` lines to scrape instead of the seed list")
    ap.add_argument("--profile", default=os.path.expanduser("~/.dark-pattern-scrape-profile"),
                    help="Chrome user-data-dir (kept out of the repo)")
    ap.add_argument("--settle", type=float, default=4.0, help="seconds to wait after load")
    args = ap.parse_args()

    if args.urls:
        urls = []
        with open(args.urls) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                site, _, u = line.partition(",")
                urls.append((site.strip(), u.strip()))
    else:
        urls = SEED_URLS

    print(f"Scraping {len(urls)} URLs (headless={args.headless}, profile={args.profile})")
    scrape(urls, args.headless, args.profile, args.settle)


if __name__ == "__main__":
    main()
