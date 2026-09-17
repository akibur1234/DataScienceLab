import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# GOODREADS DATA INSIGHT / DATA ANALYSIS
# ============================================================

INPUT_FILE = "goodreads_cleaned.csv"

OUTPUT_FOLDER = "data_insights"


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# ============================================================
# 2. CHECK CSV FILE
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("=" * 70)
    print("ERROR")
    print("=" * 70)

    print("\nCSV file was not found!")

    print("\nMake sure this file exists:")
    print(os.path.abspath(INPUT_FILE))

    print("\nExpected file:")
    print("goodreads_cleaned.csv")

    exit()


# ============================================================
# 3. READ CSV
# ============================================================

print("=" * 70)
print("GOODREADS DATA INSIGHT")
print("=" * 70)

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

    print("\nError reading CSV:")
    print(e)

    exit()


print("\nCSV loaded successfully!")


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nNumber of rows:", df.shape[0])

print("Number of columns:", df.shape[1])

print("\nColumns:")

for column in df.columns:
    print("-", column)


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n")
print("=" * 70)
print("2. DATA TYPES")
print("=" * 70)

print(
    df.dtypes
)


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n")
print("=" * 70)
print("3. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing_percentage = (
    df.isnull().sum()
    / len(df)
    * 100
)

missing_table = pd.DataFrame({

    "Missing Values": missing,

    "Missing Percentage": missing_percentage

})

print(
    missing_table
)


# Save missing value report

missing_table.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "missing_values.csv"
    )
)


# ============================================================
# 7. DUPLICATE DATA
# ============================================================

print("\n")
print("=" * 70)
print("4. DUPLICATE DATA")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(
    "Total duplicate rows:",
    duplicate_count
)


# ============================================================
# 8. NUMERICAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("5. NUMERICAL DATA SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns


if len(numeric_columns) > 0:

    numerical_summary = df[
        numeric_columns
    ].describe().T

    print(
        numerical_summary
    )

    numerical_summary.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "numerical_summary.csv"
        )
    )

else:

    print("No numerical columns found.")


# ============================================================
# 9. TEXT/CATEGORICAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("6. CATEGORICAL DATA SUMMARY")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns


for column in categorical_columns:

    print("\n--------------------------------")
    print("Column:", column)
    print("--------------------------------")

    print(
        "Unique values:",
        df[column].nunique()
    )

    print(
        df[column]
        .value_counts()
        .head(10)
    )


# ============================================================
# 10. RATING INSIGHTS
# ============================================================

if "rating" in df.columns:

    print("\n")
    print("=" * 70)
    print("7. RATING INSIGHTS")
    print("=" * 70)

    print(
        "Average rating:",
        round(df["rating"].mean(), 2)
    )

    print(
        "Highest rating:",
        round(df["rating"].max(), 2)
    )

    print(
        "Lowest rating:",
        round(df["rating"].min(), 2)
    )

    print(
        "Median rating:",
        round(df["rating"].median(), 2)
    )

    print(
        "Standard deviation:",
        round(df["rating"].std(), 2)
    )

    # Highest rated books

    highest_rated = df.sort_values(
        by="rating",
        ascending=False
    ).head(10)

    print("\nTop 10 highest-rated books:")

    columns_to_show = [
        column for column in
        ["rank", "title", "author", "rating"]
        if column in df.columns
    ]

    print(
        highest_rated[
            columns_to_show
        ].to_string(index=False)
    )

    highest_rated.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "top_10_highest_rated_books.csv"
        ),
        index=False
    )


# ============================================================
# 11. MOST POPULAR BOOKS
# ============================================================

if "ratings_count" in df.columns:

    print("\n")
    print("=" * 70)
    print("8. MOST POPULAR BOOKS")
    print("=" * 70)

    most_popular = df.sort_values(
        by="ratings_count",
        ascending=False
    ).head(10)

    columns_to_show = [
        column for column in
        [
            "rank",
            "title",
            "author",
            "rating",
            "ratings_count"
        ]
        if column in df.columns
    ]

    print(
        most_popular[
            columns_to_show
        ].to_string(index=False)
    )

    most_popular.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "top_10_most_popular_books.csv"
        ),
        index=False
    )


# ============================================================
# 12. HIGHEST SCORE BOOKS
# ============================================================

if "score" in df.columns:

    print("\n")
    print("=" * 70)
    print("9. HIGHEST SCORE BOOKS")
    print("=" * 70)

    highest_score = df.sort_values(
        by="score",
        ascending=False
    ).head(10)

    columns_to_show = [
        column for column in
        [
            "rank",
            "title",
            "author",
            "rating",
            "score"
        ]
        if column in df.columns
    ]

    print(
        highest_score[
            columns_to_show
        ].to_string(index=False)
    )

    highest_score.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "top_10_highest_score_books.csv"
        ),
        index=False
    )


# ============================================================
# 13. TOP AUTHORS
# ============================================================

if "author" in df.columns:

    print("\n")
    print("=" * 70)
    print("10. TOP AUTHORS")
    print("=" * 70)

    author_counts = (
        df["author"]
        .value_counts()
        .head(20)
    )

    print(
        author_counts
    )

    author_counts.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "top_authors.csv"
        )
    )


# ============================================================
# 14. RATING DISTRIBUTION
# ============================================================

if "rating" in df.columns:

    print("\n")
    print("=" * 70)
    print("11. RATING DISTRIBUTION")
    print("=" * 70)

    rating_distribution = (
        df["rating"]
        .round(1)
        .value_counts()
        .sort_index()
    )

    print(
        rating_distribution
    )

    rating_distribution.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "rating_distribution.csv"
        )
    )


# ============================================================
# 15. CORRELATION ANALYSIS
# ============================================================

if len(numeric_columns) >= 2:

    print("\n")
    print("=" * 70)
    print("12. CORRELATION ANALYSIS")
    print("=" * 70)

    correlation = df[
        numeric_columns
    ].corr()

    print(
        correlation
    )

    correlation.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "correlation_matrix.csv"
        )
    )


# ============================================================
# 16. GRAPH 1 - RATING DISTRIBUTION
# ============================================================

if "rating" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["rating"],
        bins=20,
        kde=True
    )

    plt.title(
        "Distribution of Goodreads Ratings"
    )

    plt.xlabel(
        "Rating"
    )

    plt.ylabel(
        "Number of Books"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "01_rating_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 17. GRAPH 2 - TOP 10 HIGHEST RATED BOOKS
# ============================================================

if "rating" in df.columns and "title" in df.columns:

    top_books = df.sort_values(
        by="rating",
        ascending=False
    ).head(10)

    plt.figure(figsize=(12, 7))

    plt.barh(
        top_books["title"].astype(str),
        top_books["rating"]
    )

    plt.title(
        "Top 10 Highest Rated Books"
    )

    plt.xlabel(
        "Rating"
    )

    plt.ylabel(
        "Book"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "02_top_10_highest_rated.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 18. GRAPH 3 - TOP AUTHORS
# ============================================================

if "author" in df.columns:

    top_authors = (
        df["author"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(12, 7))

    plt.bar(
        top_authors.index.astype(str),
        top_authors.values
    )

    plt.title(
        "Top 10 Authors by Number of Books"
    )

    plt.xlabel(
        "Author"
    )

    plt.ylabel(
        "Number of Books"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "03_top_authors.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 19. GRAPH 4 - RATINGS COUNT
# ============================================================

if (
    "title" in df.columns
    and "ratings_count" in df.columns
):

    popular_books = df.sort_values(
        by="ratings_count",
        ascending=False
    ).head(10)

    plt.figure(figsize=(12, 7))

    plt.barh(
        popular_books["title"].astype(str),
        popular_books["ratings_count"]
    )

    plt.title(
        "Top 10 Most Rated Books"
    )

    plt.xlabel(
        "Number of Ratings"
    )

    plt.ylabel(
        "Book"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "04_most_rated_books.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 20. GRAPH 5 - RATING VS RATINGS COUNT
# ============================================================

if (
    "rating" in df.columns
    and "ratings_count" in df.columns
):

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["ratings_count"],
        df["rating"],
        alpha=0.5
    )

    plt.title(
        "Rating vs Number of Ratings"
    )

    plt.xlabel(
        "Number of Ratings"
    )

    plt.ylabel(
        "Average Rating"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "05_rating_vs_ratings_count.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 21. GRAPH 6 - SCORE DISTRIBUTION
# ============================================================

if "score" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["score"],
        bins=20,
        kde=True
    )

    plt.title(
        "Distribution of Goodreads Scores"
    )

    plt.xlabel(
        "Score"
    )

    plt.ylabel(
        "Number of Books"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "06_score_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 22. GRAPH 7 - CORRELATION HEATMAP
# ============================================================

if len(numeric_columns) >= 2:

    plt.figure(
        figsize=(10, 7)
    )

    correlation = df[
        numeric_columns
    ].corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Between Numerical Variables"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "07_correlation_heatmap.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 23. CREATE OVERALL INSIGHT REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_FOLDER,
    "data_insight_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "GOODREADS DATA INSIGHT REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    # Dataset size

    report.write(
        f"Total number of books: {len(df)}\n"
    )

    report.write(
        f"Total number of columns: {len(df.columns)}\n\n"
    )

    # Rating

    if "rating" in df.columns:

        report.write(
            "RATING INSIGHTS\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            f"Average rating: "
            f"{df['rating'].mean():.2f}\n"
        )

        report.write(
            f"Highest rating: "
            f"{df['rating'].max():.2f}\n"
        )

        report.write(
            f"Lowest rating: "
            f"{df['rating'].min():.2f}\n"
        )

        report.write(
            f"Median rating: "
            f"{df['rating'].median():.2f}\n\n"
        )

    # Ratings count

    if "ratings_count" in df.columns:

        report.write(
            "POPULARITY INSIGHTS\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            f"Average ratings count: "
            f"{df['ratings_count'].mean():.2f}\n"
        )

        report.write(
            f"Maximum ratings count: "
            f"{df['ratings_count'].max()}\n"
        )

        report.write(
            f"Minimum ratings count: "
            f"{df['ratings_count'].min()}\n\n"
        )

    # Authors

    if "author" in df.columns:

        report.write(
            "AUTHOR INSIGHTS\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            f"Unique authors: "
            f"{df['author'].nunique()}\n\n"
        )

        report.write(
            "Top 10 authors:\n"
        )

        top_authors = (
            df["author"]
            .value_counts()
            .head(10)
        )

        for author, count in top_authors.items():

            report.write(
                f"{author}: {count} books\n"
            )

        report.write("\n")


# ============================================================
# 24. SAVE COMPLETE DATASET STATISTICS
# ============================================================

statistics = {}

statistics["Total Books"] = len(df)

statistics["Total Columns"] = len(df.columns)

statistics["Total Duplicate Rows"] = (
    df.duplicated().sum()
)

if "rating" in df.columns:

    statistics["Average Rating"] = (
        df["rating"].mean()
    )

    statistics["Maximum Rating"] = (
        df["rating"].max()
    )

    statistics["Minimum Rating"] = (
        df["rating"].min()
    )

if "ratings_count" in df.columns:

    statistics["Average Ratings Count"] = (
        df["ratings_count"].mean()
    )

if "author" in df.columns:

    statistics["Unique Authors"] = (
        df["author"].nunique()
    )


statistics_df = pd.DataFrame(
    statistics.items(),
    columns=["Metric", "Value"]
)


statistics_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "overall_statistics.csv"
    ),
    index=False
)


# ============================================================
# 25. FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("DATA INSIGHT COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nAll insight files were saved inside:")

print(
    os.path.abspath(OUTPUT_FOLDER)
)

print("\nGenerated files:")

for file in sorted(
    os.listdir(OUTPUT_FOLDER)
):

    print(
        "-",
        file
    )


print("\n")
print("=" * 70)
print("IMPORTANT INSIGHTS")
print("=" * 70)


if "rating" in df.columns:

    print(
        "\nAverage rating:",
        round(df["rating"].mean(), 2)
    )

    print(
        "Highest rating:",
        round(df["rating"].max(), 2)
    )

    print(
        "Lowest rating:",
        round(df["rating"].min(), 2)
    )


if "author" in df.columns:

    print(
        "Number of unique authors:",
        df["author"].nunique()
    )


if "ratings_count" in df.columns:

    print(
        "Most rated book:"
    )

    most_rated = df.loc[
        df["ratings_count"].idxmax()
    ]

    if "title" in df.columns:

        print(
            most_rated["title"]
        )

    print(
        "Ratings:",
        int(most_rated["ratings_count"])
    )
print("\nProgram finished.")