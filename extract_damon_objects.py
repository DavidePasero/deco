import numpy as np
import os
from common import constants
from collections import Counter

def extract_unique_objects():
    # Load DAMON dataset for all splits
    dataset = 'damon'
    modes = ['train', 'val', 'test']
    
    all_unique_objects = set()
    all_object_counts = Counter()
    split_unique_objects = {}
    
    for mode in modes:
        print(f"\nAnalyzing {mode} split:")
        try:
            print(f'Loading dataset: {constants.DATASET_FILES[mode][dataset]}')
            data = np.load(constants.DATASET_FILES[mode][dataset], allow_pickle=True)
            
            # Get object-wise contact labels
            try:
                contact_labels_objectwise = data['contact_label_objectwise']
                print(f"Found {len(contact_labels_objectwise)} samples with object-wise contacts")
                
                # Count objects in this split
                split_object_counts = Counter()
                split_unique_objects[mode] = set()
                
                for sample in contact_labels_objectwise:
                    if isinstance(sample, dict):
                        for obj_name in sample.keys():
                            split_object_counts[obj_name] += 1
                            split_unique_objects[mode].add(obj_name)
                            all_unique_objects.add(obj_name)
                
                # Update total counts
                all_object_counts.update(split_object_counts)
                
                # Print results for this split
                print(f"Found {len(split_object_counts)} unique object names in {mode} split:")
                for obj, count in sorted(split_object_counts.items()):
                    print(f"  - {obj}: {count} occurrences")
                    
            except KeyError:
                print(f"No 'contact_label_objectwise' found in the {mode} dataset")
                split_unique_objects[mode] = set()
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading {mode} split: {e}")
            split_unique_objects[mode] = set()
    
    # Print combined results
    print("\n=== COMBINED STATISTICS ===")
    print(f"Total unique objects across all splits: {len(all_unique_objects)}")
    print("Object counts (sorted by frequency):")
    for obj, count in sorted(all_object_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {obj}: {count} occurrences")
    
    # Find objects missing in val and test
    if 'train' in split_unique_objects and 'val' in split_unique_objects:
        missing_in_val = split_unique_objects['train'] - split_unique_objects['val']
        print(f"\nObjects in train but missing in val ({len(missing_in_val)}):")
        for obj in sorted(missing_in_val):
            print(f"  - {obj}")
    
    if 'train' in split_unique_objects and 'test' in split_unique_objects:
        missing_in_test = split_unique_objects['train'] - split_unique_objects['test']
        print(f"\nObjects in train but missing in test ({len(missing_in_test)}):")
        for obj in sorted(missing_in_test):
            print(f"  - {obj}")
            
    return all_unique_objects, all_object_counts, split_unique_objects

if __name__ == '__main__':
    unique_objects, object_counts, split_objects = extract_unique_objects()
    print(f"\nTotal unique objects: {len(unique_objects)}")
