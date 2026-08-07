#!/usr/bin/env python3
"""Wake / keep a Streamlit Community Cloud app actually running.

Plain HTTP (curl, UptimeRobot) only hits the static SPA shell and gets HTTP 200
while the Python process stays asleep. A real browser is required so JS runs and
the WebSocket (/_stcore/stream) starts — that is what wakes the app.
"""

from __future__ import annotations

import argparse
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

WAKE_BUTTON = "Yes, get this app back up!"
SLEEP_MARKERS = ("Zzzz", "gone to sleep due to inactivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        required=True,
        help="Streamlit Cloud app URL, e.g. https://myapp.streamlit.app/",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=60,
        help="Keep the browser session open after wake/confirm (default: 60).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=120_000,
        help="Navigation / wake timeout in milliseconds (default: 120000).",
    )
    return parser.parse_args()


def is_sleeping(page) -> bool:
    text = page.locator("body").inner_text(timeout=5_000)
    return any(marker in text for marker in SLEEP_MARKERS)


def app_looks_ready(page) -> bool:
    """Streamlit mounts into #root; sleep page has the wake CTA instead."""
    if page.get_by_role("button", name=WAKE_BUTTON).count() > 0:
        return False
    if is_sleeping(page):
        return False
    # Prefer a positive Streamlit signal when available.
    for selector in (
        "[data-testid='stApp']",
        "[data-testid='stAppViewContainer']",
        "section.main",
        "#root [class*='stApp']",
    ):
        if page.locator(selector).count() > 0:
            return True
    # Fallback: page loaded and sleep UI is gone.
    return page.locator("#root").count() > 0 and not is_sleeping(page)


def wake_or_confirm(page, url: str, timeout_ms: int, hold_seconds: int) -> None:
    print(f"Opening {url}")
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_timeout(5_000)

    wake_btn = page.get_by_role("button", name=WAKE_BUTTON)
    if wake_btn.count() > 0:
        print("App is sleeping — clicking wake button")
        wake_btn.click()
        # Cold start can take a while on Community Cloud.
        deadline = time.monotonic() + (timeout_ms / 1000)
        while time.monotonic() < deadline:
            page.wait_for_timeout(5_000)
            if app_looks_ready(page):
                break
        else:
            raise RuntimeError("Timed out waiting for app to wake after button click")
    else:
        print("Wake button not present — checking that the app is actually running")

    if not app_looks_ready(page):
        raise RuntimeError(
            "App still looks asleep (sleep page / wake button still present). "
            "HTTP monitors would report UP incorrectly here."
        )

    print(f"App is awake — holding session for {hold_seconds}s")
    page.wait_for_timeout(hold_seconds * 1000)
    print("Done")


def main() -> int:
    args = parse_args()
    url = args.url if args.url.endswith("/") else args.url + "/"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()
        try:
            wake_or_confirm(page, url, args.timeout_ms, args.hold_seconds)
        except PlaywrightTimeout as exc:
            print(f"ERROR: Playwright timeout: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 — surface clearly in Actions logs
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        finally:
            context.close()
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
