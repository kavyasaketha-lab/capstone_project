------------------------------------------------------------------------------------------------------------

## Objectives

The pipeline performs the following tasks:

1. Scrape book information using `requests` and `BeautifulSoup`.
2. Collect at least **60 books** across at least **3 categories**.
3. Clean and convert the scraped fields into appropriate data types.
4. Convert GBP prices to INR using the required fixed project rate.
5. Store the cleaned data in a normalized SQLite database.
6. Create a primary-key/foreign-key relationship between categories and books.
7. Execute SQL queries demonstrating the required SQL clauses.
8. Read SQL query results into pandas DataFrames.
9. Reproduce the JOIN result using `pandas.merge()`.
10. Compare the SQL JOIN and pandas JOIN results.

------------------------------------------------------------------------------------------------------------


****************************************************************
1. Scrape book information using `requests` and `BeautifulSoup`.
****************************************************************
The project uses the public scraping-practice website:
`books.toscrape.com`
The website does not requires any login, API key, or paid subscription.

***************Data Source***************
The data is collected from:Books to Scrape
Website:https://books.toscrape.com/
This is a website specifically created for practicing web scraping.

The scraper uses the `requests` library to retrieve HTML pages and `BeautifulSoup` to parse the HTML.

Example approach:

python
import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

****************************************************************
2. Collect at least **60 books** across at least **3 categories**.
****************************************************************
The scraper navigates through the required catalogue/category pages and extracts book information.

The final dataset contains:

* At least 60 books
* At least 3 categories


The scraper collects the following fields:

| Field          | Description                                                |
| -------------- | ---------------------------------------------------------- |
| `title`        | Book title                                                 |
| `price`        | Original price displayed on the website                    |
| `star_rating`  | Rating as text, e.g. `One`, `Two`, `Three`, `Four`, `Five` |
| `availability` | Availability text displayed on the website                 |
| `category`     | Book category                                              |

-
****************************************************************
3. Clean and convert the scraped fields into appropriate data types.
****************************************************************
## Price Cleaning
****************************************************************
The website provides prices in GBP, for example:


Â£51.77


The currency symbol is removed and the value is converted to a floating-point number:


Â£51.77 → 51.77


****************************************************************
## Star Rating Cleaning
****************************************************************
The website represents ratings as text:


One
Two
Three
Four
Five


These values are converted into integers:

| Original | Cleaned |
| -------- | ------: |
| One      |       1 |
| Two      |       2 |
| Three    |       3 |
| Four     |       4 |
| Five     |       5 |

****************************************************************
## Availability Cleaning
****************************************************************
The availability text is converted into a Boolean value.

For example:


In stock → True
Out of stock → False


****************************************************************

## Handling Invalid or Unexpected Values
****************************************************************
The scraper includes defensive parsing so that an unexpected value does not cause the entire pipeline to fail.

For numeric fields such as `price_gbp` and `rating`:

1. Attempt to parse the value.
2. If parsing fails, represent the value as missing.
3. Apply median imputation for the affected numeric field.

This follows the assignment requirement that numeric parsing failures should use median imputation rather than leaving the pipeline to crash.

For non-numeric fields where the value cannot be reliably interpreted, the affected row is dropped if the missing value prevents the row from being meaningfully stored.

All cleaning decisions are applied before loading the data into SQLite.

****************************************************************
4. Convert GBP prices to INR using the required fixed project rate.
****************************************************************
The project uses the required fixed conversion rate:

> **1 GBP = 105.50 INR**

This is a **project-defined fixed baseline rate**.

It is not a live exchange rate and no currency API is required.

The INR price is calculated as:


price_inr = price_gbp × 105.50


For example:


£10.00 × 105.50 = ₹1,055.00


The cleaned dataset contains:

| Cleaned Field | Data Type | Description                            |
| ------------- | --------- | -------------------------------------- |
| `title`       | TEXT      | Book title                             |
| `price_gbp`   | FLOAT     | Book price in GBP                      |
| `price_inr`   | FLOAT     | Converted price in INR                 |
| `rating`      | INTEGER   | Rating from 1 to 5                     |
| `in_stock`    | BOOLEAN   | Whether the book is currently in stock |
| `category`    | TEXT      | Book category                          |



****************************************************************
5. Store the cleaned data in a normalized SQLite database.
****************************************************************
SQLite is used as the relational database.

The database contains two normalized tables:


categories: 1 to many books


## Categories Table

sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);


Each category is stored only once.

---

## Books Table

sql
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
);


The `category_id` column connects each book to the corresponding category.

---

## Relationship

The database uses a one-to-many relationship:


categories
-----------
category_id (PK)
category_name
     
books
-----------
book_id (PK)
title
price_gbp
price_inr
rating
in_stock
category_id (FK)


This avoids storing the category name repeatedly for every book and provides a normalized relational structure.

****************************************************************
6. Create a primary-key/foreign-key relationship between categories and books.
****************************************************************
The cleaned data is inserted into SQLite using Python's `sqlite3` library.

Foreign-key support is enabled to enforce the relationship:

python
connection.execute("PRAGMA foreign_keys = ON")


Categories are inserted into the `categories` table first.

Each book is then inserted into the `books` table using the corresponding `category_id`.

The database can therefore be recreated from scratch by running the pipeline.

****************************************************************
7. Execute SQL queries demonstrating the required SQL clauses.
****************************************************************

At least five SQL queries are executed against the SQLite database.

The queries collectively demonstrate:

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `IN` or `BETWEEN`
* `JOIN`

The SQL queries are stored in:


queries.sql


Their executed results are saved in:


output/query_outputs.txt


---

## Query 1: SELECT and WHERE

Example:

sql
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4;


This retrieves books with a rating of 4 or higher.

---

## Query 2: ORDER BY

Example:

sql
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC;


This sorts books from the highest price to the lowest price.

---

## Query 3: LIMIT

Example:

sql
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;


This returns the 10 most expensive books.

---

## Query 4: DISTINCT

Example:

sql
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;


This returns the unique book categories.

---

## Query 5: IN / BETWEEN

Example using `BETWEEN`:

sql
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40;


This returns books whose GBP price falls between £20 and £40.

---

# 7. JOIN Query

A JOIN is used to combine information from the `books` and `categories` tables.

Example:

sql
SELECT
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.title
LIMIT 10;


This demonstrates the primary-key/foreign-key relationship between the two tables.

The JOIN result is saved as part of the query output.

---
****************************************************************
8. Read SQL query results into pandas DataFrames.
****************************************************************

SQL query results are loaded into pandas DataFrames using:

python
pd.read_sql()


Example:

python
df = pd.read_sql(
    """
    SELECT
        b.title,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock,
        c.category_name
    FROM books b
    JOIN categories c
        ON b.category_id = c.category_id
    ORDER BY b.rating DESC, b.title
    LIMIT 10;
    """,
    connection
)


At least two SQL query results are read into pandas DataFrames.

****************************************************************
9. Reproduce the JOIN result using `pandas.merge()`.
****************************************************************
The JOIN result is independently reproduced using `pandas.merge()`.

The two source tables are first loaded into DataFrames:

python
books_df = pd.read_sql(
    "SELECT * FROM books",
    connection
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    connection
)


The JOIN is then performed without SQL:

python
merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)


The required columns and ordering are then applied to produce the equivalent result.

This demonstrates that the relational JOIN can be reproduced directly using pandas.

---
****************************************************************
10. Compare the SQL JOIN and pandas JOIN results.
****************************************************************
The SQL JOIN result and pandas `merge()` result are compared to ensure that both approaches produce equivalent data.

Example comparison:

python
sql_result = pd.read_sql(sql_join_query, connection)

pandas_result = (
    pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )
    [["title", "price_gbp", "price_inr",
      "rating", "in_stock", "category_name"]]
    .sort_values(["rating", "title"], ascending=[False, True])
    .head(10)
    .reset_index(drop=True)
)

sql_result = sql_result.reset_index(drop=True)

print(sql_result.equals(pandas_result))


The expected output is:


True


This confirms that the SQL JOIN and pandas `merge()` produce equivalent results.


****************************************************************
# 11. Expected Output
****************************************************************
After successfully running the pipeline, the following should be available:

### Scraped dataset

At least:


60+ book records
3+ categories


### Cleaned fields

title
price_gbp
price_inr
rating
in_stock
category


### SQLite database

Zepto_books.db


containing tables:
categories
books


### SQL output

The repository contains the executed outputs for all required SQL queries.

### Pandas output

The repository contains:

* At least two `pd.read_sql()` results
* SQL JOIN result
* pandas `merge()` result
* Comparison confirming equivalent results


****************************************************************
# 12. Error Handling
****************************************************************
The scraper handles common issues such as:

* HTTP request failures
* Missing HTML elements
* Unexpected price formats
* Unexpected rating values
* Unexpected availability text
* Missing category information
* Invalid numeric values

HTTP errors are handled using:

"python response.raise_for_status()"
Parsing operations are protected so that a single malformed record does not cause the complete pipeline to crash.

****************************************************************
# 15. Git Workflow
****************************************************************

The overall repository should demonstrate the required Git workflow.
create a directory where you want the project files 
then change the path using 


A feature branch should be created from `main`:

```bash
cd **path_of_the_direcctory**
git clone https://github.com/kavyasaketha-lab/capstone_project

git pull
git checkout -b feature/data_pipeline
```

Make the first commit:

```bash
git add .
git commit -m "Add book scraping pipeline"
```

Make further changes and commit again:

```bash
git add .
git commit -m "Add SQLite database and SQL analysis"
```

Push the feature branch:

```bash
git push -u origin feature/data_pipeline
```

The feature branch should then be merged back into `main`.

Example:

```bash
git checkout main
git merge feature/data_pipeline
git push origin main
```

The repository history should therefore show:

```text
main
  |
  └── feature/data-pipeline
          |
          ├── commit 1
          └── commit 2
                    |
                    └── merged into main
```

This satisfies the repository-level requirement for:

* Feature branch creation
* At least two commits
* Merge back into `main`
