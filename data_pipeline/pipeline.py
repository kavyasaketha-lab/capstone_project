import sqlite3
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# Configuration the variables
# ============================================================

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

GBP_TO_INR = 105.50

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "zepto_books.db"

MAX_PAGES = 5

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
    "six":6
}

# ============================================================
# SCRAPING DATA OF BOOKS USING URL
# ============================================================

def scrape_books(num_pages=5):
    """
    Scrape books from the first `num_pages` pages.

    Returns:
        list[dict]: Raw scraped book records.
    """

    books = []

    for page_number in range(1, num_pages + 1):

        url = BASE_URL.format(page_number)

        print(f"Scraping page {page_number}: {url}")

        try:
            response = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )
            # Explicitly check HTTP status.
            response.raise_for_status()

        except requests.RequestException as error:
            print(f"Failed to download page {page_number}: {error}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Each product is represented by an article.
        product_articles = soup.select("article.product_pod")
        for product in product_articles:

            # ------------------------------------------------
            # Title
            # ------------------------------------------------

            title_element = product.select_one("h3 a")

            title = (
                title_element.get("title", "").strip()
                if title_element
                else None
            )

            # ------------------------------------------------
            # Price
            # ------------------------------------------------

            price_element = product.select_one(".price_color")

            price_text = (
                price_element.get_text(strip=True)
                if price_element
                else None
            )

            # ------------------------------------------------
            # Rating
            # ------------------------------------------------

            rating_element = product.select_one("p.star-rating")

            star_rating = None

            if rating_element:
                classes = rating_element.get("class", [])

                # Example:
                # ["star-rating", "Three"]

                if len(classes) >= 2:
                    star_rating = classes[1]

            # ------------------------------------------------
            # Availability
            # ------------------------------------------------

            availability_element = product.select_one(
                ".availability"
            )

            availability = (
                availability_element.get_text(
                    " ",
                    strip=True
                )
                if availability_element
                else None
            )

            # ------------------------------------------------
            # Category
            # ------------------------------------------------

            # The listing page itself does not expose the
            # category directly.
            #
            # Therefore we follow the product detail page
            # and obtain the category from the breadcrumb.

            category = None

            if title_element and title_element.get("href"):

                detail_url = requests.compat.urljoin(
                    url,
                    title_element["href"]
                )

                try:
                    detail_response = requests.get(
                        detail_url,
                        timeout=15,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )

                    detail_response.raise_for_status()

                    detail_soup = BeautifulSoup(
                        detail_response.text,
                        "html.parser"
                    )

                    breadcrumb = detail_soup.select(
                        "ul.breadcrumb li a"
                    )

                    # Breadcrumb:
                    # Home > Books > Category > Book
                    #
                    # Category is normally index 2.
                    if len(breadcrumb) >= 3:
                        category = breadcrumb[2].get_text(
                            strip=True
                        )

                except requests.RequestException as error:
                    print(
                        f"Could not retrieve category "
                        f"for '{title}': {error}"
                    )

            books.append({
                "title": title,
                "price_raw": price_text,
                "star_rating_raw": star_rating,
                "availability_raw": availability,
                "category": category
            })

    return books



# ============================================================
# CONVERTING PRICE FROM GBP TO PLAIN FLOAT NUM
# ============================================================

def parse_price(value):
    """
    Convert a price such as '£51.77' to 51.77.
    """

    if value is None:
        return None

    try:
        cleaned = (
            str(value)
            .replace("Â£", "")
            .replace(",", "")
            .strip()
        )

        return float(cleaned)

    except (ValueError, TypeError):
        return None

# ============================================================
# PARSING RATE FROM STRING TO INT
# ============================================================
def parse_rating(value):
    """
    Convert textual rating to integer.

    Example:
        Three -> 3
    """

    if value is None:
        return None

    return RATING_MAP.get(str(value).strip())

# ============================================================
# CONVERTING STOCK AVAILABILITY TO BOOLEAN DATATYPE FROM TEXT
# ============================================================
def parse_stock(value):
    """
    Convert availability text to boolean.

    'In stock' -> True
    Anything else containing stock information -> False
    """

    if value is None:
        return None

    text = str(value).strip().lower()

    if "in stock" in text:
        return True

    if "out of stock" in text:
        return False

    return None


# ============================================================
# CLEANING DATA FRAME
# ============================================================

def clean_dataframe(df):
    """
    Clean and transform the raw DataFrame.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    df["price_gbp"] = df["price_raw"].apply(parse_price)

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    df["rating"] = df["star_rating_raw"].apply(parse_rating)

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    df["in_stock"] = df["availability_raw"].apply(parse_stock)

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    df["category"] = df["category"].fillna("Unknown")

    # --------------------------------------------------------
    # Numeric missing values
    #
    # Requirement:
    # numeric parsing failures should use median
    # imputation rather than crashing.
    # --------------------------------------------------------

    for column in ["price_gbp", "rating"]:

        median_value = df[column].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Unable to calculate median for {column}"
            )

        df[column] = df[column].fillna(median_value)

    # Rating must remain an integer.
    df["rating"] = df["rating"].round().astype(int)

    # --------------------------------------------------------
    # Boolean missing values
    #
    # For an unknown availability value, we drop the row.
    # It would be unsafe to assume that an unknown value
    # means either in-stock or out-of-stock.
    # --------------------------------------------------------

    df = df.dropna(subset=["in_stock"])

    df["in_stock"] = df["in_stock"].astype(bool)

    # --------------------------------------------------------
    # INR conversion
    #
    # Fixed project-defined rate:
    #
    # 1 GBP = 105.50 INR
    # --------------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    return df.reset_index(drop=True)



# ============================================================
# DATABASE CREATION
# ============================================================

def create_database(connection):
    """
    Create normalized categories and books tables.
    """

    cursor = connection.cursor()

    # Enable foreign key enforcement.
    cursor.execute("PRAGMA foreign_keys = ON")

    # Remove old tables so that the pipeline is reproducible.
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # --------------------------------------------------------
    # Books
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    connection.commit()


# ============================================================
# LOAD DATA
# ============================================================

def load_data(df, connection):
    """
    Insert cleaned DataFrame into normalized SQLite tables.
    """

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Insert unique categories
    # --------------------------------------------------------

    categories = sorted(
        df["category"].dropna().unique()
    )

    cursor.executemany(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        [(category,) for category in categories]
    )

    # --------------------------------------------------------
    # Build category lookup
    # --------------------------------------------------------

    cursor.execute("""
        SELECT category_id, category_name
        FROM categories
    """)

    category_lookup = {
        row[1]: row[0]
        for row in cursor.fetchall()
    }

    # --------------------------------------------------------
    # Insert books
    # --------------------------------------------------------

    book_records = []

    for _, row in df.iterrows():

        category_id = category_lookup[row["category"]]

        book_records.append(
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_id
            )
        )

    cursor.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        book_records
    )

    connection.commit()


# ============================================================
# SQL QUERIES
# ============================================================

QUERIES = {

    "select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp > 40
        LIMIT 10;
    """,

    "order_by": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "distinct": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,

    "between": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,

    "join": """
        SELECT
            b.title,
            c.category_name,
            b.rating,
            b.price_gbp,
            b.price_inr,
            b.in_stock
        FROM books AS b
        INNER JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name,
            b.rating DESC,
            b.title;
    """
}


def execute_queries(connection):
    """
    Execute required SQL queries and print results.
    """

    results = {}

    for query_name, query in QUERIES.items():

        print("\n" + "=" * 70)
        print(f"QUERY: {query_name}")
        print("=" * 70)

        print(query.strip())

        result = pd.read_sql_query(
            query,
            connection
        )

        print("\nOUTPUT:")
        print(result.to_string(index=False))

        results[query_name] = result

    return results


# ============================================================
# 6. VALIDATION
# ============================================================

def validate_pipeline(df, connection):
    """
    Validate acceptance criteria.
    """

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # DataFrame checks

    print(f"Scraped/cleaned rows: {len(df)}")
    print(
        f"Unique categories: "
        f"{df['category'].nunique()}"
    )

    assert len(df) >= 60, (
        "Dataset contains fewer than 60 books."
    )

    assert df["category"].nunique() >= 3, (
        "Fewer than 3 categories found."
    )

    assert df["price_gbp"].notna().all()
    assert df["rating"].between(1, 5).all()
    assert df["in_stock"].notna().all()
    assert df["price_inr"].notna().all()

    # Check conversion.

    expected_inr = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    assert (
        df["price_inr"] == expected_inr
    ).all()

    # Database checks.

    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM books")

    book_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM categories"
    )

    category_count = cursor.fetchone()[0]

    print(f"Books in database: {book_count}")
    print(
        f"Categories in database: {category_count}"
    )

    assert book_count >= 60
    assert category_count >= 3

    print("\nAll validation checks passed.")


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("ZEPTO DATA PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Extract DATA FROM URL
    # --------------------------------------------------------

    raw_books = scrape_books(MAX_PAGES)

    print(
        f"\nRaw records scraped: {len(raw_books)}"
    )

    raw_df = pd.DataFrame(raw_books)

    print("\nRaw data sample:")
    print(raw_df.head())

    # --------------------------------------------------------
    # CLEANING DATA EXTRACTED FROM THE URL
    # --------------------------------------------------------

    cleaned_datadf = clean_dataframe(raw_df)
    print(cleaned_datadf["title"])
    print(
        f"\nCleaned records: {len(cleaned_datadf)}"
    )

    print("\nCleaned data sample:")
    print(cleaned_datadf.head())

    print("\nData types:")
    print(cleaned_datadf.dtypes)

    # --------------------------------------------------------
    # Loading sql data
    # --------------------------------------------------------

    connection = sqlite3.connect(DB_PATH)

    try:

        create_database(connection)

        load_data(
            cleaned_datadf,
            connection
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validate_pipeline(
            cleaned_datadf,
            connection
        )

        # ----------------------------------------------------
        # Queries
        # ----------------------------------------------------

        results = execute_queries(
            connection
        )

        # Save query outputs as CSV files.
        for query_name, result in results.items():

            output_path = (
                BASE_DIR /
                f"{query_name}_output.csv"
            )

            result.to_csv(
                output_path,
                index=False
            )

            print(
                f"Saved: {output_path}"
            )

    finally:

        connection.close()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()