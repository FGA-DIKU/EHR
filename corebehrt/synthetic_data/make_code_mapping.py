import os
import pandas as pd
from os.path import join
import torch

import sys
sys.path.append("../")
from azure_utils.setup import setup_azure

FOLDER_PATH = "BC_recreate"
FILE_PATH = "data_20240910/MEDS_old_v2/shard_events"
INPUT_TYPES = ["diagnosis", "medication", "procedure"]
MIN_COUNT = 15
MAX_ITERATIONS = 10
STRATEGY = 'individual'
SAVE_DIR = "code_mappings"

def get_count_of_codes(types, file_path):
    code_counts = {}
    for input_type in types:
        print('Handling', input_type)
        type_path = join(file_path, input_type)
        for file in os.listdir(type_path):
            if file.endswith(".parquet"):
                data = pd.read_parquet(join(type_path, file))
                summed_data = data.groupby("code").size().reset_index(name="count")
                
                for _, row in summed_data.iterrows():
                    code = row['code']
                    count = row['count']
                    if code in code_counts:
                        code_counts[code] += count
                    else:
                        code_counts[code] = count
    
    return code_counts

def get_codes_to_prune(code_counts, min_counts, strategy='individual'):
    if strategy == 'individual':
        # Individual strategy: only prune codes that are under threshold
        codes_to_prune = [code for code, count in code_counts.items() if count < min_counts and len(code) > 2]
    elif strategy == 'group':
        raise NotImplementedError("Group strategy is not implemented yet")
        
        # Group strategy: if any code of a certain length is under threshold, prune all codes of that length or longer
        prefixes_to_prune = [code[:-1] for code, count in code_counts.items() if count < min_counts and len(code) > 2]
        for prefix in prefixes_to_prune:
            codes_to_prune = [code for code, count in code_counts.items() if code.startswith(prefix) and len(code) > len(prefix)]

    return codes_to_prune

def prune_code_mappings(code_counts, code_mappings, min_counts, strategy='individual', iterations=0):
    codes_to_prune = get_codes_to_prune(code_counts, min_counts, strategy)
    
    if len(codes_to_prune) == 0 or iterations > MAX_ITERATIONS:
        return code_counts, code_mappings
        
    for code in codes_to_prune:
        prefix = code[:-1]
        code_mappings[code] = prefix
        
        # Update all codes that were previously mapped to this code
        for original_code, mapped_code in code_mappings.items():
            if mapped_code == code:
                code_mappings[original_code] = prefix
        
        if code.startswith('DC50'):
            prefix_count = code_counts.get(prefix, 0)
            print(f"Pruning {code} (count: {code_counts[code]}) to {prefix} (count: {prefix_count})")
        
        if prefix in code_counts:
            code_counts[prefix] = code_counts[code] + code_counts[prefix]
        else:
            code_counts[prefix] = code_counts[code]
        del code_counts[code]
    
    return prune_code_mappings(code_counts, code_mappings, min_counts, strategy, iterations + 1)

def get_code_mapping(code_counts, min_count, strategy='individual'):
    """
    Create code mapping with two strategies:
    
    strategy='group': If ANY code in a prefix group is under min_count, 
                     map ALL codes in that group to the prefix
    strategy='individual': Map each code individually if its prefix group 
                          meets the total threshold
    """
    code_mappings = {code: code for code in code_counts.keys()}
    code_counts, code_mappings = prune_code_mappings(code_counts, code_mappings, min_count, strategy)
    
    # Ensure no duplicates in code_counts (both as keys and values)
    final_code_counts = {}
    for code, count in code_counts.items():
        if code not in final_code_counts:
            final_code_counts[code] = count
    
    # Print lowest code counts
    print("Lowest code counts:")
    sorted_counts = sorted(final_code_counts.items(), key=lambda x: x[1])
    for code, count in sorted_counts[:10]:
        print(f"  {code}: {count}")

    print("Highest code counts:")
    sorted_counts = sorted(final_code_counts.items(), key=lambda x: x[1], reverse=True)
    for code, count in sorted_counts[:10]:
        print(f"  {code}: {count}")  

    return code_mappings

if __name__ == "__main__":
    mount_context = setup_azure(workspacename='researcher_data', folder_path=FOLDER_PATH)
    file_path = join(mount_context, FILE_PATH)
    code_counts = get_count_of_codes(INPUT_TYPES, file_path)
    
    print(f"Original unique codes: {len(code_counts)}")
    print(f"Total occurrences: {sum(code_counts.values())}")
    
    # Show some examples of codes and their counts
    print(f"\nSample codes and counts:")
    for i, (code, count) in enumerate(sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
        print(f"  {code}: {count}")
    
    # Test the mapping
    group_mapping = get_code_mapping(code_counts, MIN_COUNT, strategy=STRATEGY)
        
    # Print all mappings that start with DC50
    print("\nAll mappings starting with DC50:")
    dc50_mappings = {code: mapped for code, mapped in group_mapping.items() if code.startswith('DC50')}
    for code, mapped in sorted(dc50_mappings.items()):
        code_count = code_counts.get(code, "N/A")
        mapped_count = code_counts.get(mapped, "N/A")
        print(f"{code} -> {mapped} (count: {code_count} -> {mapped_count})")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(group_mapping, join(SAVE_DIR, f"{STRATEGY}_mapping.pt"))
        
