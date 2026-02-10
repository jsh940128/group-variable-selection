"""
Scrape Michelin Key hotels (name + URL) from the Michelin Guide article page.

Key improvements vs XPath div[37]:
- Targets the FULL content container(s): div.detail-page__content
- Scrolls to bottom to trigger lazy rendering (if any)
- Extracts ALL hotel links (/h/<id>) within those containers
- Saves output to ./data/ with an absolute path printed
"""

import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


URL = "https://guide.michelin.com/my/en/article/travel/all-the-key-hotels-france-michelin-guide"
HOTEL_URL_RE = re.compile(r"^https://guide\.michelin\.com/h/\d+$")


def scroll_to_bottom(driver, pause=0.8, max_steps=25):
    """
    Some sites render content lazily. This forces the browser to scroll down until
    the page height stops changing (or until max_steps is reached).
    """
    last_height = driver.execute_script("return document.body.scrollHeight")

    for _ in range(max_steps):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        last_height = new_height


def extract_hotels_from_html(html: str) -> pd.DataFrame:
    """
    Parse HTML and extract all hotel name + url pairs from anchors
    linking to https://guide.michelin.com/h/<id>.
    """
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    seen = set()

    # Extract every matching hotel link
    for a in soup.select("a[href]"):
        name = a.get_text(strip=True)
        href = (a.get("href") or "").strip()

        # Normalize relative to absolute (just in case)
        if href.startswith("/h/"):
            href = "https://guide.michelin.com" + href

        if not HOTEL_URL_RE.match(href):
            continue

        key = (name, href)
        if key in seen:
            continue
        seen.add(key)

        rows.append({"hotel_name": name, "hotel_url": href})

    return pd.DataFrame(rows, columns=["hotel_name", "hotel_url"])


def main():
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Uncomment if you want headless mode:
    # chrome_options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(URL)

        # Wait for at least one detail-page__content to appear
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.detail-page__content"))
        )

        # Force lazy-loaded content to render
        scroll_to_bottom(driver)

        # Grab ALL detail-page__content blocks (sometimes there can be more than one)
        content_divs = driver.find_elements(By.CSS_SELECTOR, "div.detail-page__content")

        if not content_divs:
            raise RuntimeError("No div.detail-page__content found. Page structure may have changed.")

        # Combine HTML from all content blocks (safer than assuming one node)
        combined_html = "\n".join(d.get_attribute("outerHTML") for d in content_divs)

        df = extract_hotels_from_html(combined_html)

        # ---- save to explicit location ----
        out_dir = Path.cwd() / "data"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "michelin_key_hotels_france.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print("=" * 70)
        print("Extraction complete")
        print(f"detail-page__content blocks: {len(content_divs)}")
        print(f"Hotels extracted: {len(df)}")
        print(f"Saved to: {out_path.resolve()}")
        print("=" * 70)
        print(df.head(15).to_string(index=False))

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
