
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "goodreads_cleaned.csv"

FINAL_K = 4

NUMBER_OF_RECOMMENDATIONS = 5

RANDOM_STATE = 42


# ============================================================
# 2. LOAD CSV FILE
# ============================================================

print("=" * 65)
print("GOODREADS BOOK CLUSTERING AND RECOMMENDATION")
print("=" * 65)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"File not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nReading file:", INPUT_FILE)

print("Original shape:", df.shape)

print("\nOriginal column names:")

print(df.columns.tolist())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. CHECK REQUIRED ORIGINAL COLUMNS
# ============================================================

required_original_columns = [
    "title",
    "author",
    "rating",
    "ratings_count",
    "score",
    "votes"
]

missing_columns = [
    column
    for column in required_original_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns: "
        + str(missing_columns)
        + "\nAvailable columns: "
        + str(df.columns.tolist())
    )


# ============================================================
# 5. RENAME COLUMNS
# ============================================================

df = df.rename(columns={

    "title": "book_name",

    "ratings_count": "rating_count",

    "score": "meta_score_num",

    "votes": "number_of_voted_num",

    "rating": "rating_value"

})


print("\nRenamed columns:")

print(df.columns.tolist())


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

original_rows = len(df)

df = df.drop_duplicates()

print(
    "\nDuplicates removed:",
    original_rows - len(df)
)


# ============================================================
# 7. CLEAN TEXT COLUMNS
# ============================================================

df = df.dropna(
    subset=["book_name"]
)

df["book_name"] = (
    df["book_name"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["author"] = (
    df["author"]
    .fillna("Unknown")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ============================================================
# 8. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "rating_value",

    "rating_count",

    "meta_score_num",

    "number_of_voted_num"

]

for column in numeric_columns:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


# ============================================================
# 9. VALIDATE RATING VALUES
# ============================================================

invalid_rating = (

    (df["rating_value"] < 0)

    |

    (df["rating_value"] > 5)

)

df.loc[invalid_rating, "rating_value"] = np.nan


# ============================================================
# 10. SELECT CLUSTERING FEATURES
# ============================================================

feature_columns = [

    "rating_value",

    "rating_count",

    "meta_score_num",

    "number_of_voted_num"

]

features = df[feature_columns].copy()


# ============================================================
# 11. HANDLE INFINITE VALUES
# ============================================================

features = features.replace(

    [np.inf, -np.inf],

    np.nan

)


# ============================================================
# 12. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before filling:")

print(features.isnull().sum())


for column in feature_columns:

    median_value = features[column].median()

    if pd.isna(median_value):

        median_value = 0

    features[column] = (

        features[column]

        .fillna(median_value)

    )


# ============================================================
# 13. PREVENT NEGATIVE COUNTS
# ============================================================

count_columns = [

    "rating_count",

    "meta_score_num",

    "number_of_voted_num"

]

for column in count_columns:

    features[column] = features[column].clip(

        lower=0

    )


# ============================================================
# 14. LOG TRANSFORMATION
# ============================================================

for column in count_columns:

    features[column] = np.log1p(

        features[column]

    )


# ============================================================
# 15. CHECK DATA SIZE
# ============================================================

if len(df) < 2:

    raise ValueError(

        "At least 2 books are required for clustering."

    )


# ============================================================
# 16. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)

print("\nFeature scaling completed.")

print("Total books used:", len(df))


# ============================================================
# 17. DETERMINE SAFE K RANGE
# ============================================================

max_k = min(10, len(df) - 1)

k_values = range(2, max_k + 1)


# ============================================================
# 18. ELBOW METHOD
# ============================================================

print("\nCalculating elbow method...")

inertia_values = []

for k in k_values:

    model = KMeans(

        n_clusters=k,

        random_state=RANDOM_STATE,

        n_init=10

    )

    model.fit(X_scaled)

    inertia_values.append(

        model.inertia_

    )


plt.figure(figsize=(9, 6))

plt.plot(

    list(k_values),

    inertia_values,

    marker="o"

)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.xticks(list(k_values))

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(

    "elbow_method.png",

    dpi=300

)

plt.close()

print("Saved: elbow_method.png")


# ============================================================
# 19. SILHOUETTE SCORE
# ============================================================

print("\nCalculating silhouette scores...")

silhouette_values = []

for k in k_values:

    model = KMeans(

        n_clusters=k,

        random_state=RANDOM_STATE,

        n_init=10

    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(

        X_scaled,

        labels

    )

    silhouette_values.append(score)

    print(

        f"K = {k}, "

        f"Silhouette Score = {score:.4f}"

    )


plt.figure(figsize=(9, 6))

plt.plot(

    list(k_values),

    silhouette_values,

    marker="o"

)

plt.title("Silhouette Scores")

plt.xlabel("Number of Clusters")

plt.ylabel("Silhouette Score")

plt.xticks(list(k_values))

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(

    "silhouette_scores.png",

    dpi=300

)

plt.close()

print("Saved: silhouette_scores.png")


# ============================================================
# 20. TRAIN FINAL K-MEANS MODEL
# ============================================================

if FINAL_K > len(df):

    raise ValueError(

        "FINAL_K cannot be greater than total books."

    )

print("\nTraining final K-Means model...")

print("Selected K:", FINAL_K)

kmeans = KMeans(

    n_clusters=FINAL_K,

    random_state=RANDOM_STATE,

    n_init=10

)

cluster_labels = kmeans.fit_predict(

    X_scaled

)

df["Cluster"] = cluster_labels


# ============================================================
# 21. FINAL SILHOUETTE SCORE
# ============================================================

final_silhouette = silhouette_score(

    X_scaled,

    cluster_labels

)

print(

    "\nFinal Silhouette Score:",

    round(final_silhouette, 4)

)


# ============================================================
# 22. CLUSTER SUMMARY
# ============================================================

cluster_summary = (

    df.groupby("Cluster")

    .agg(

        Number_of_Books=(

            "book_name",

            "size"

        ),

        Average_Rating=(

            "rating_value",

            "mean"

        ),

        Average_Rating_Count=(

            "rating_count",

            "mean"

        ),

        Average_Meta_Score=(

            "meta_score_num",

            "mean"

        ),

        Average_Votes=(

            "number_of_voted_num",

            "mean"

        )

    )

    .reset_index()

)


print("\nCLUSTER SUMMARY")

print(

    cluster_summary

    .round(2)

    .to_string(index=False)

)


cluster_summary.to_csv(

    "cluster_summary.csv",

    index=False,

    encoding="utf-8-sig"

)

print("\nSaved: cluster_summary.csv")


# ============================================================
# 23. PCA VISUALIZATION
# ============================================================

print("\nCreating PCA visualization...")

pca = PCA(

    n_components=2,

    random_state=RANDOM_STATE

)

X_pca = pca.fit_transform(X_scaled)


plt.figure(figsize=(10, 7))

for cluster_number in sorted(

    df["Cluster"].unique()

):

    mask = (

        cluster_labels == cluster_number

    )

    plt.scatter(

        X_pca[mask, 0],

        X_pca[mask, 1],

        label=f"Cluster {cluster_number}",

        alpha=0.7

    )


plt.title(

    "Goodreads Book Clusters (PCA)"

)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(

    "book_clusters_pca.png",

    dpi=300

)

plt.close()

print("Saved: book_clusters_pca.png")


# ============================================================
# 24. SAVE CLUSTERED DATASET
# ============================================================

df.to_csv(

    "goodreads_clustered.csv",

    index=False,

    encoding="utf-8-sig"

)

print(

    "Saved: goodreads_clustered.csv"

)


# ============================================================
# 25. RECOMMENDATION FUNCTION
# ============================================================

def recommend_books(

    book_name,

    number_of_recommendations=5

):

    matches = df[

        df["book_name"]

        .str.casefold()

        == book_name.strip().casefold()

    ]

    if matches.empty:

        print(

            "\nBook not found:",

            book_name

        )

        print(

            "\nAvailable book examples:"

        )

        print(

            df["book_name"]

            .head(10)

            .to_string(index=False)

        )

        return pd.DataFrame()


    selected_index = matches.index[0]

    selected_position = df.index.get_loc(

        selected_index

    )

    selected_cluster = df.loc[

        selected_index,

        "Cluster"

    ]


    print(

        "\nSelected Book:",

        df.loc[

            selected_index,

            "book_name"

        ]

    )

    print(

        "Selected Cluster:",

        selected_cluster

    )


    # Select books from the same cluster.

    same_cluster_mask = (

        cluster_labels == selected_cluster

    )

    candidate_positions = np.where(

        same_cluster_mask

    )[0]

    candidate_df = df.iloc[

        candidate_positions

    ].copy()

    candidate_X = X_scaled[

        candidate_positions

    ]

    selected_vector = X_scaled[

        selected_position

    ].reshape(1, -1)


    # At least one other book is required.

    if len(candidate_df) <= 1:

        print(

            "Not enough books in this cluster."

        )

        return pd.DataFrame()


    n_neighbors = min(

        number_of_recommendations + 1,

        len(candidate_df)

    )


    neighbors = NearestNeighbors(

        n_neighbors=n_neighbors,

        metric="euclidean"

    )

    neighbors.fit(candidate_X)


    distances, positions = neighbors.kneighbors(

        selected_vector

    )


    recommendations = candidate_df.iloc[

        positions[0]

    ].copy()


    recommendations["Distance"] = distances[0]


    # Remove the selected book itself.

    recommendations = recommendations[

        recommendations.index != selected_index

    ]


    recommendations = recommendations.head(

        number_of_recommendations

    )


    result_columns = [

        "book_name",

        "author",

        "Cluster",

        "rating_value",

        "rating_count",

        "meta_score_num",

        "number_of_voted_num",

        "Distance"

    ]

    recommendations = recommendations[

        [

            column

            for column in result_columns

            if column in recommendations.columns

        ]

    ]


    return recommendations


# ============================================================
# 26. GENERATE RECOMMENDATIONS
# ============================================================

# Automatically select the first book in the dataset.
# This avoids errors when "Gone Girl" is not available.

selected_book = df["book_name"].iloc[0]

print(

    "\nGenerating recommendations for:",

    selected_book

)


recommendations = recommend_books(

    selected_book,

    NUMBER_OF_RECOMMENDATIONS

)


if not recommendations.empty:

    print("\nRECOMMENDED BOOKS")

    print(

        recommendations.to_string(

            index=False

        )

    )

    recommendations.to_csv(

        "book_recommendations.csv",

        index=False,

        encoding="utf-8-sig"

    )

    print(

        "\nSaved: book_recommendations.csv"

    )


# ============================================================
# 27. FINAL RESULT
# ============================================================

print("\n" + "=" * 65)

print("PROJECT COMPLETED")

print("=" * 65)

print(

    "\nTotal processed rows:",

    len(df)

)

print(

    "Number of clusters:",

    FINAL_K

)

print(

    "Final silhouette score:",

    round(final_silhouette, 4)

)

print("\nGenerated files:")

print("1. elbow_method.png")

print("2. silhouette_scores.png")

print("3. book_clusters_pca.png")

print("4. cluster_summary.csv")

print("5. goodreads_clustered.csv")

if not recommendations.empty:

    print("6. book_recommendations.csv")

print("\nAll tasks completed successfully!")