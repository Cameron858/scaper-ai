"""
Web scraper for fishkeeping caresheets.

Fetches caresheet data from fishkeeping.co.uk and saves to local files.
"""

import hashlib
import logging
import random
import time
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from pyprojroot import here

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

HEADERS = {
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
}
TIMEOUT = 30


def fetch(client: httpx.Client, url: str, max_retries=3) -> None | httpx.Response:
    """Fetch a URL with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                raise


def main():
    base_url = "https://www.fishkeeping.co.uk"

    logger.info("Fetching caresheets index page...")
    r = httpx.get(
        "https://www.fishkeeping.co.uk/modules/caresheets/",
        headers=HEADERS,
        timeout=TIMEOUT,
        follow_redirects=True,
    )
    logger.info(f"Index page status: {r.status_code}")

    logger.info("Parsing links from index page...")
    soup = BeautifulSoup(r.text, "html.parser")

    a_tags = soup.find_all("a")
    # str() just to avoid linting issues with BeautifulSoup's Tag objects
    links = [str(a["href"]) for a in a_tags if "caresheetID" in a["href"]]
    links = sorted(links, key=lambda x: int(x.split("=")[-1]))

    logger.info(f"Found {len(links)} caresheet links")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = here() / "data" / "rip" / f"{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {output_path}")

    processed_count = 0
    failed_count = 0

    with httpx.Client(
        headers=HEADERS, timeout=TIMEOUT, follow_redirects=True
    ) as client:
        for i, link in enumerate(links, 1):
            try:
                logger.debug(f"Processing link {i}/{len(links)}: {link}")

                page_link = base_url + link
                r = fetch(client, page_link)

                soup = BeautifulSoup(r.text, "html.parser")

                data_card = soup.select_one(
                    "#caresheets > div.container.maincontainer > div > div.col-sm-9.col-md-9 > fieldset:nth-child(1) > table"
                )
                short_card, long_card = data_card.find_all("table")

                details = short_card.find("p").text.split("\n")
                details = [d.strip().replace("\n", "") for d in details if d.strip()]

                rows = long_card.find_all("tr")
                description = [row.text.strip().replace("\n", "") for row in rows]

                file_name = f"{hashlib.md5(page_link.encode()).hexdigest()}.txt"
                output_file = output_path / file_name

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(details) + "\n" + "\n".join(description))

                processed_count += 1
                logger.info(f"Saved caresheet {i}/{len(links)} to {file_name}")

                # Random jitter to avoid hitting the server too hard
                time.sleep(0.5 + random.random() * 0.7)

            except Exception as e:
                failed_count += 1
                logger.error(
                    f"Failed to process link {i}/{len(links)}: {e}", exc_info=True
                )
                continue

    logger.info(
        f"Scraping complete. Processed: {processed_count}, Failed: {failed_count}"
    )


if __name__ == "__main__":
    main()
