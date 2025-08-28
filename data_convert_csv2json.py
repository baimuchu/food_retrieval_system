# Import necessary libraries
import pandas as pd
import json
import os

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv('prosusai_assignment_data/5k_items_curated.csv')

print(f"CSV loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Function to safely parse JSON strings
def safe_json_parse(json_str):
    try:
        return json.loads(json_str)
    except:
        return {}

# Parse the JSON columns to make them proper JSON objects
print("\nParsing JSON columns...")
df['itemMetadata_parsed'] = df['itemMetadata'].apply(safe_json_parse)
df['itemProfile_parsed'] = df['itemProfile'].apply(safe_json_parse)

# Create a cleaner structure for the JSON output
def create_clean_item(row):
    """Create a clean item structure for JSON output"""
    item = {
        "itemId": row['itemId'],
        "merchantId": row['merchantId'],
        "itemMetadata": row['itemMetadata_parsed'],
        "itemProfile": row['itemProfile_parsed']
    }
    return item

# Convert to list of dictionaries
print("Converting to JSON format...")
items_list = []
for idx, row in df.iterrows():
    if idx % 1000 == 0:  # Progress indicator
        print(f"Processing item {idx}/{len(df)}")
    
    clean_item = create_clean_item(row)
    items_list.append(clean_item)

# Create the final JSON structure
json_data = {
    "metadata": {
        "total_items": len(items_list),
        "total_merchants": df['merchantId'].nunique(),
        "description": "Food items dataset converted from CSV to JSON format"
    },
    "items": items_list
}

# Save as JSON file
output_file = 'prosusai_assignment_data/5k_items_curated.json'
print(f"\nSaving to JSON file: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"JSON file saved successfully!")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

# Also create a sample JSON file with just first 10 items for easier inspection
sample_file = 'prosusai_assignment_data/sample_items.json'
print(f"\nCreating sample file with first 10 items: {sample_file}")

sample_data = {
    "metadata": {
        "total_items": 10,
        "description": "Sample of first 10 items from the dataset"
    },
    "items": items_list[:10]
}

with open(sample_file, 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=2)

print(f"Sample file saved successfully!")
print(f"Sample file size: {os.path.getsize(sample_file) / 1024:.2f} KB")

# Show the structure of the first item
print(f"\n{'='*60}")
print(f"STRUCTURE OF FIRST ITEM:")
print(f"{'='*60}")

first_item = items_list[0]
print(json.dumps(first_item, indent=2, ensure_ascii=False))

# Show some statistics about the converted data
print(f"\n{'='*60}")
print(f"CONVERSION STATISTICS:")
print(f"{'='*60}")

# Count items with images
items_with_images = sum(1 for item in items_list if item['itemMetadata'].get('images'))
print(f"Items with images: {items_with_images}")

# Count items with complete taxonomy
items_with_taxonomy = sum(1 for item in items_list if item['itemMetadata'].get('taxonomy'))
print(f"Items with taxonomy: {items_with_taxonomy}")

# Count items with metrics
items_with_metrics = sum(1 for item in items_list if item['itemProfile'].get('metrics'))
print(f"Items with metrics: {items_with_metrics}")

print(f"\nConversion completed successfully!")
print(f"Main JSON file: {output_file}")
print(f"Sample file: {sample_file}")