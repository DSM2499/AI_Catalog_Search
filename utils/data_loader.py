import pandas as pd
import re

def clean_text(text):
    #Lowercase, strip and remove non-alphanumeric characters from text
    if pd.isnull(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def load_clean_data(file_path = "data/styles.csv", output_path = "data/clean_products.csv"):
    #Load raw data
    df = pd.read_csv(file_path, on_bad_lines='skip')

    cols = ['id', 'productDisplayName', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'usage']
    df = df[cols]

    # Drop rows with missing critical values
    df.dropna(subset=['productDisplayName', 'baseColour', 'articleType'], inplace=True)

    # Clean text columns
    df['title'] = df['productDisplayName'].apply(clean_text)
    df['category'] = df['masterCategory'].fillna("Unknown").apply(clean_text)
    df['sub_category'] = df['subCategory'].fillna("Unknown").apply(clean_text)
    df['color'] = df['baseColour'].fillna("Unknown").apply(clean_text)
    df['gender'] = df['gender'].fillna("Unisex").apply(clean_text)
    df['usage'] = df['usage'].fillna("General").apply(clean_text)

    # Final dataframe
    cleaned_df = df[['id', 'title', 'category', 'sub_category', 'color', 'gender', 'usage']]
    cleaned_df.to_csv(output_path, index=False)

    print(f"[INFO] Cleaned dataset saved to: {output_path}")
    return cleaned_df


if __name__ == "__main__":
    load_clean_data()