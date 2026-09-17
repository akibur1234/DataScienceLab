import pandas as pd
import numpy as np
import os
import re


# ============================================================
# 1. FILE SETTINGS
# ============================================================

INPUT_FILE = "goodreads_best_twists.csv"
OUTPUT_FILE = "goodreads_cleaned.csv"


# ============================================================
# 2. CHECK FILE
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("ERROR: CSV file was not found!")
    print()
    print("Make sure your CSV file is inside the same folder")
    print("as this Python file.")
    print()
    print("Expected file:")
    print(INPUT_FILE)

    exit()


# ============================================================
# 3. READ CSV FILE
# ============================================================

print("=" * 60)
print("GOODREADS DATA CLEANING")
print("=" * 60)

print("\nReading CSV file...")

try:

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

except UnicodeDecodeError:

    df = pd.read_csv(
        INPUT_FILE,
        encoding="latin1"
    )

except Exception as e:

    print("Error reading CSV file:")
    print(e)
    exit()


print("\nCSV file loaded successfully!")


# ============================================================
# 4. BASIC INFORMATION BEFORE CLEANING
# ============================================================

print("\n" + "=" * 60)
print("BEFORE CLEANING")
print("=" * 60)

print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")

for column in df.columns:
    print("-", column)


# ============================================================
# 5. SHOW FIRST 5 ROWS
# ============================================================

print("\nFirst 5 rows:")

print(
    df.head().to_string()
)


# ============================================================
# 6. REMOVE EXTRA SPACES FROM COLUMN NAMES
# ============================================================

print("\nCleaning column names...")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


print("New column names:")

for column in df.columns:
    print("-", column)


# ============================================================
# 7. REMOVE COMPLETELY EMPTY ROWS
# ============================================================

before_empty_rows = len(df)

df.dropna(
    how="all",
    inplace=True
)

after_empty_rows = len(df)

print(
    "\nCompletely empty rows removed:",
    before_empty_rows - after_empty_rows
)


# ============================================================
# 8. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

before_columns = len(df.columns)

df.dropna(
    axis=1,
    how="all",
    inplace=True
)

after_columns = len(df.columns)

print(
    "Completely empty columns removed:",
    before_columns - after_columns
)


# ============================================================
# 9. CLEAN TEXT COLUMNS
# ============================================================

print("\nCleaning text columns...")

text_columns = df.select_dtypes(
    include=["object"]
).columns


for column in text_columns:

    # Convert to string
    df[column] = df[column].astype(str)

    # Remove leading/trailing spaces
    df[column] = df[column].str.strip()

    # Replace multiple spaces with one space
    df[column] = df[column].str.replace(
        r"\s+",
        " ",
        regex=True
    )

    # Convert fake missing values to NaN
    df[column] = df[column].replace(
        [
            "",
            "nan",
            "NaN",
            "none",
            "None",
            "null",
            "NULL",
            "N/A",
            "n/a",
            "NA",
            "na",
            "-"
        ],
        np.nan
    )


print("Text columns cleaned.")


# ============================================================
# 10. REMOVE DUPLICATE ROWS
# ============================================================

before_duplicates = len(df)

df.drop_duplicates(
    inplace=True
)

after_duplicates = len(df)

print(
    "\nDuplicate rows removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 11. REMOVE DUPLICATE BOOKS
# ============================================================

# If title and author columns exist,
# use them to identify duplicate books.

if "title" in df.columns:

    before_book_duplicates = len(df)

    if "author" in df.columns:

        df.drop_duplicates(
            subset=["title", "author"],
            keep="first",
            inplace=True
        )

    else:

        df.drop_duplicates(
            subset=["title"],
            keep="first",
            inplace=True
        )

    after_book_duplicates = len(df)

    print(
        "Duplicate books removed:",
        before_book_duplicates - after_book_duplicates
    )


# ============================================================
# 12. CONVERT NUMERIC COLUMNS
# ============================================================

print("\nConverting numeric columns...")


possible_numeric_columns = [
    "rank",
    "rating",
    "ratings_count",
    "score",
    "votes",
    "pages",
    "publication_year",
    "year"
]


for column in possible_numeric_columns:

    if column in df.columns:

        # Remove commas and other unwanted characters
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
        )

        # Convert to numeric
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


print("Numeric columns converted.")


# ============================================================
# 13. CLEAN RATING COLUMN
# ============================================================

if "rating" in df.columns:

    # Goodreads ratings normally range from 0 to 5

    invalid_rating = (
        (df["rating"] < 0) |
        (df["rating"] > 5)
    )

    invalid_count = invalid_rating.sum()

    df.loc[
        invalid_rating,
        "rating"
    ] = np.nan

    print(
        "Invalid ratings replaced with missing values:",
        invalid_count
    )


# ============================================================
# 14. CLEAN RANK COLUMN
# ============================================================

if "rank" in df.columns:

    invalid_rank = (
        df["rank"] <= 0
    )

    invalid_rank_count = invalid_rank.sum()

    df.loc[
        invalid_rank,
        "rank"
    ] = np.nan

    print(
        "Invalid ranks replaced:",
        invalid_rank_count
    )


# ============================================================
# 15. HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")


# Text columns
text_columns = df.select_dtypes(
    include=["object"]
).columns


for column in text_columns:

    # For text data, use "Unknown"
    df[column] = df[column].fillna(
        "Unknown"
    )


# Numeric columns
numeric_columns = df.select_dtypes(
    include=["number"]
).columns


for column in numeric_columns:

    # Use median for numeric missing values

    if df[column].isna().sum() > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )


print("Missing values handled.")


# ============================================================
# 16. CLEAN BOOK TITLES
# ============================================================

if "title" in df.columns:

    df["title"] = (
        df["title"]
        .astype(str)
        .str.strip()
    )

    # Remove unnecessary multiple spaces

    df["title"] = df["title"].str.replace(
        r"\s+",
        " ",
        regex=True
    )


# ============================================================
# 17. CLEAN AUTHOR NAMES
# ============================================================

if "author" in df.columns:

    df["author"] = (
        df["author"]
        .astype(str)
        .str.strip()
    )

    df["author"] = df["author"].str.replace(
        r"\s+",
        " ",
        regex=True
    )


# ============================================================
# 18. REMOVE UNNECESSARY INDEX COLUMN
# ============================================================

unnecessary_columns = [
    "unnamed:_0",
    "unnamed:_0"
]


for column in unnecessary_columns:

    if column in df.columns:

        df.drop(
            columns=[column],
            inplace=True
        )

        print(
            "Removed unnecessary column:",
            column
        )


# ============================================================
# 19. SORT BY RANK
# ============================================================

if "rank" in df.columns:

    df.sort_values(
        by="rank",
        inplace=True
    )


# ============================================================
# 20. RESET INDEX
# ============================================================

df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 21. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

missing_values = df.isnull().sum()

print(
    missing_values
)


# ============================================================
# 22. FINAL DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# 23. FINAL DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("AFTER CLEANING")
print("=" * 60)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# 24. DISPLAY CLEANED DATA
# ============================================================

print("\nCleaned data preview:")

print(
    df.head(10).to_string()
)


# ============================================================
# 25. SAVE CLEANED CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nCleaned CSV file:",
    OUTPUT_FILE
)

print(
    "\nLocation:"
)

print(
    os.path.abspath(OUTPUT_FILE)
)

print("\nFinal dataset shape:")

print(
    df.shape
)

print("\nDone!")