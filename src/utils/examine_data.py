import csv
import json

# Examine queries.csv
print("=== QUERIES.CSV ===")
with open('prosusai_assignment_data/queries.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    queries = list(reader)
    print(f"Number of queries: {len(queries)}")
    print(f"Columns: {queries[0].keys() if queries else 'No data'}")
    print(f"Sample queries:")
    for i, query in enumerate(queries[:5]):
        print(f"  {i+1}. {query['search_term_pt']}")

print("\n=== 5K_ITEMS_CURATED.CSV ===")
with open('prosusai_assignment_data/5k_items_curated.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    items = list(reader)
    print(f"Number of items: {len(items)}")
    print(f"Columns: {items[0].keys() if items else 'No data'}")
    
    if items:
        print(f"\nSample item structure:")
        sample_item = items[0]
        for key, value in sample_item.items():
            if key == 'itemMetadata':
                try:
                    metadata = json.loads(value)
                    print(f"  {key}:")
                    for meta_key, meta_value in metadata.items():
                        if meta_key == 'images':
                            print(f"    {meta_key}: {len(meta_value)} images")
                        elif meta_key == 'taxonomy':
                            print(f"    {meta_key}: {meta_value}")
                        else:
                            print(f"    {meta_key}: {meta_value}")
                except:
                    print(f"  {key}: [JSON parsing error]")
            elif key == 'itemProfile':
                try:
                    profile = json.loads(value)
                    print(f"  {key}:")
                    for profile_key, profile_value in profile.items():
                        print(f"    {profile_key}: {profile_value}")
                except:
                    print(f"  {key}: [JSON parsing error]")
            else:
                print(f"  {key}: {value}")
        
        # Show a few more examples
        print(f"\nMore examples:")
        for i in range(1, min(4, len(items))):
            item = items[i]
            metadata = json.loads(item['itemMetadata'])
            print(f"  Item {i+1}: {metadata.get('name', 'N/A')} - {metadata.get('category_name', 'N/A')}") 