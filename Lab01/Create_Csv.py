import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://www.goodreads.com/list/show/541.Best_Twists"

# Goodreads currently has around 88 pages for this list.
# We use 100 pages as a safety limit.
MAX_PAGES = 100

OUTPUT_FILE = "goodreads_best_twists.csv"


# ============================================================
# HTTP HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# FUNCTION: CLEAN TEXT
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# FUNCTION: EXTRACT NUMBER
# ============================================================

def extract_number(text):

    if not text:
        return None

    text = text.replace(",", "")

    numbers = re.findall(r"\d+", text)

    if numbers:
        return int(numbers[0])

    return None


# ============================================================
# FUNCTION: SCRAPE ONE PAGE
# ============================================================

def scrape_page(page_number):

    url = f"{BASE_URL}?page={page_number}"

    print("\n------------------------------------------")
    print(f"Scraping page {page_number}")
    print(url)
    print("------------------------------------------")

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        print("Status code:", response.status_code)

        if response.status_code != 200:
            print("Could not access page.")
            return []

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        # Goodreads list items
        books = soup.select("tr")

        page_data = []

        for row in books:

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            title_tag = row.select_one(
                "a.bookTitle"
            )

            if not title_tag:
                continue

            title = clean_text(
                title_tag.get_text()
            )

            book_url = title_tag.get("href")

            if book_url:

                if book_url.startswith("/"):
                    book_url = (
                        "https://www.goodreads.com"
                        + book_url
                    )

            # ------------------------------------------------
            # AUTHOR
            # ------------------------------------------------

            author_tag = row.select_one(
                "a.authorName"
            )

            if author_tag:

                author = clean_text(
                    author_tag.get_text()
                )

            else:

                author = ""

            # ------------------------------------------------
            # RATING
            # ------------------------------------------------

            rating_tag = row.select_one(
                "span.minirating"
            )

            rating = None
            ratings_count = None

            if rating_tag:

                rating_text = clean_text(
                    rating_tag.get_text()
                )

                # Example:
                # 4.27 avg rating — 1,706,216 ratings

                rating_match = re.search(
                    r"(\d+\.\d+)\s+avg rating",
                    rating_text
                )

                if rating_match:

                    rating = float(
                        rating_match.group(1)
                    )

                ratings_match = re.search(
                    r"([\d,]+)\s+ratings",
                    rating_text
                )

                if ratings_match:

                    ratings_count = int(
                        ratings_match.group(1)
                        .replace(",", "")
                    )

            # ------------------------------------------------
            # SCORE
            # ------------------------------------------------

            row_text = clean_text(
                row.get_text(" ", strip=True)
            )

            score = None

            score_match = re.search(
                r"score:\s*([\d,]+)",
                row_text,
                re.IGNORECASE
            )

            if score_match:

                score = int(
                    score_match.group(1)
                    .replace(",", "")
                )

            # ------------------------------------------------
            # VOTES
            # ------------------------------------------------

            votes = None

            votes_match = re.search(
                r"([\d,]+)\s+people voted",
                row_text,
                re.IGNORECASE
            )

            if votes_match:

                votes = int(
                    votes_match.group(1)
                    .replace(",", "")
                )

            # ------------------------------------------------
            # RANK
            # ------------------------------------------------

            rank = None

            rank_tag = row.select_one(
                ".number"
            )

            if rank_tag:

                rank_text = clean_text(
                    rank_tag.get_text()
                )

                rank_match = re.search(
                    r"\d+",
                    rank_text
                )

                if rank_match:

                    rank = int(
                        rank_match.group()
                    )

            # ------------------------------------------------
            # SAVE BOOK
            # ------------------------------------------------

            book_data = {

                "rank": rank,

                "title": title,

                "author": author,

                "rating": rating,

                "ratings_count": ratings_count,

                "score": score,

                "votes": votes,

                "book_url": book_url,

                "page": page_number
            }

            page_data.append(book_data)

        print(
            f"Books found on page {page_number}: "
            f"{len(page_data)}"
        )

        return page_data

    except Exception as e:

        print(
            f"Error while scraping page "
            f"{page_number}: {e}"
        )

        return []


# ============================================================
# MAIN SCRAPER
# ============================================================

def main():

    all_books = []

    print("\n==========================================")
    print("GOODREADS BEST TWISTS SCRAPER")
    print("==========================================")

    for page in range(1, MAX_PAGES + 1):

        books = scrape_page(page)

        # ----------------------------------------------------
        # STOP IF PAGE HAS NO BOOKS
        # ----------------------------------------------------

        if not books:

            print(
                f"\nNo books found on page {page}."
            )

            print(
                "Assuming scraping is finished."
            )

            break

        all_books.extend(books)

        print(
            f"Total books collected: "
            f"{len(all_books)}"
        )

        # ----------------------------------------------------
        # WAIT BETWEEN REQUESTS
        # ----------------------------------------------------

        wait_time = random.uniform(
            2,
            5
        )

        print(
            f"Waiting {wait_time:.2f} seconds..."
        )

        time.sleep(wait_time)

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not all_books:

        print(
            "\nNo data was collected."
        )

        return

    df = pd.DataFrame(
        all_books
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    df.drop_duplicates(
        subset=[
            "title",
            "author"
        ],
        inplace=True
    )

    # --------------------------------------------------------
    # SORT DATA
    # --------------------------------------------------------

    if "rank" in df.columns:

        df = df.sort_values(
            by="rank",
            na_position="last"
        )

    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n==========================================")
    print("SCRAPING COMPLETED")
    print("==========================================")

    print(
        f"Total unique books: {len(df)}"
    )

    print(
        f"CSV file created: {OUTPUT_FILE}"
    )

    print("\nFirst 10 records:")

    print(
        df.head(10).to_string()
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()