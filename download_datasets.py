import os
import sys
import json
import pickle
import requests
from tqdm import tqdm

# Ensure the datasets library is installed/loaded
try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package is not installed. Please run 'pip install datasets' first.")
    sys.exit(1)

# Base data directory
DATA_DIR = os.path.join(os.getcwd(), "data")

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def save_hf_to_jsonl(dataset_name, folder_name, rename_splits=None):
    """Downloads a HF dataset and saves splits to JSONL format."""
    print(f"\n--- Downloading Hugging Face Dataset: {dataset_name} ---")
    dest_dir = os.path.join(DATA_DIR, folder_name)
    check_dir(dest_dir)
    
    try:
        # Load dataset with trust_remote_code=True to bypass security restrictions on older scripts
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        
        # Save each split
        for split_name in dataset.keys():
            out_split_name = rename_splits.get(split_name, split_name) if rename_splits else split_name
            file_path = os.path.join(dest_dir, f"{out_split_name}.jsonl")
            
            # Check if file already exists and is non-empty
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"File '{file_path}' already exists and is valid. Skipping split '{split_name}'.")
                continue
                
            print(f"Saving '{split_name}' split to {file_path}...")
            split_data = dataset[split_name]
            
            # Write to a temporary file first, then rename it, to prevent corrupted partial files on crash
            temp_file_path = file_path + ".tmp"
            try:
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    for item in tqdm(split_data, desc=f"Writing {out_split_name}"):
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                if os.path.exists(file_path):
                    os.remove(file_path)
                os.rename(temp_file_path, file_path)
                print(f"Successfully saved {len(split_data)} records to {file_path}")
            except Exception as fe:
                print(f"Error writing file {file_path}: {fe}")
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
            
    except Exception as e:
        print(f"Error downloading/saving {dataset_name}: {e}")

def download_url_to_file(url, filename, folder_name):
    """Downloads a file from a URL with a progress bar."""
    print(f"\nChecking official ParlAI resource: {filename}")
    dest_dir = os.path.join(DATA_DIR, folder_name)
    check_dir(dest_dir)
    dest_path = os.path.join(dest_dir, filename)
    
    # Check if file exists and is valid
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        if filename.endswith(".pkl"):
            try:
                with open(dest_path, 'rb') as pf:
                    pickle.load(pf)
                print(f"File '{filename}' already exists and is verified. Skipping download.")
                return
            except Exception:
                print(f"File '{filename}' exists but fails validation. Re-downloading...")
        else:
            print(f"File '{filename}' already exists. Skipping download.")
            return

    print(f"Downloading from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        temp_dest_path = dest_path + ".tmp"
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
        with open(temp_dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        
        if total_size != 0 and t.n != total_size:
            print("WARNING: Might have failed to download complete file.")
            if os.path.exists(temp_dest_path):
                os.remove(temp_dest_path)
        else:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_dest_path, dest_path)
            print(f"Successfully downloaded {filename} to {dest_path}")
            
        # Verify if it is a pickle file
        if filename.endswith(".pkl") and os.path.exists(dest_path):
            try:
                with open(dest_path, 'rb') as pf:
                    data = pickle.load(pf)
                    print(f"Verified pickle file integrity. Type of loaded data: {type(data)}")
            except Exception as pe:
                print(f"Warning: downloaded file could not be loaded as pickle: {pe}")
                
    except Exception as e:
        print(f"Error downloading {url}: {e}")
def download_dailydialog_custom():
    print(f"\n--- Downloading Custom DailyDialog (ConvLab) ---")
    dest_dir = os.path.join(DATA_DIR, "dailydialog")
    check_dir(dest_dir)
    
    # Check if all splits already exist and are valid
    splits = ["train", "validation", "test"]
    all_exist = True
    for s in splits:
        file_path = os.path.join(dest_dir, f"{s}.jsonl")
        if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
            all_exist = False
            break
    
    if all_exist:
        print("DailyDialog dataset already exists and is valid. Skipping download.")
        return
        
    url = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip"
    print(f"Downloading from: {url}")
    
    try:
        import zipfile
        import io
        response = requests.get(url)
        response.raise_for_status()
        
        print("Extracting and parsing zip package...")
        z = zipfile.ZipFile(io.BytesIO(response.content))
        dialogues = json.loads(z.read('data/dialogues.json').decode('utf-8'))
        
        # Group dialogues by split
        grouped_data = {"train": [], "validation": [], "test": []}
        for d in dialogues:
            split = d['data_split']
            if split in grouped_data:
                grouped_data[split].append(d)
                
        # Write splits
        for split_name, records in grouped_data.items():
            file_path = os.path.join(dest_dir, f"{split_name}.jsonl")
            print(f"Saving '{split_name}' split to {file_path}...")
            temp_file_path = file_path + ".tmp"
            with open(temp_file_path, "w", encoding="utf-8") as f:
                for item in tqdm(records, desc=f"Writing {split_name}"):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rename(temp_file_path, file_path)
            print(f"Successfully saved {len(records)} records to {file_path}")
            
    except Exception as e:
        print(f"Error processing DailyDialog: {e}")

def download_personachat_custom():
    print(f"\n--- Downloading Custom PersonaChat (bavard) ---")
    dest_dir = os.path.join(DATA_DIR, "personachat")
    check_dir(dest_dir)
    
    splits = ["train", "validation"]
    all_exist = True
    for s in splits:
        file_path = os.path.join(dest_dir, f"{s}.jsonl")
        if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
            all_exist = False
            break
            
    if all_exist:
        print("PersonaChat dataset already exists and is valid. Skipping download.")
        return
        
    try:
        # Load dataset directly from json files on HF Hub to bypass python loading script issues
        data_files = {
            "train": "https://huggingface.co/datasets/bavard/personachat_truecased/resolve/main/personachat_truecased_full_train.json",
            "validation": "https://huggingface.co/datasets/bavard/personachat_truecased/resolve/main/personachat_truecased_full_valid.json"
        }
        dataset = load_dataset("json", data_files=data_files)
        
        # Save splits
        for split_name in dataset.keys():
            file_path = os.path.join(dest_dir, f"{split_name}.jsonl")
            print(f"Saving '{split_name}' split to {file_path}...")
            split_data = dataset[split_name]
            
            temp_file_path = file_path + ".tmp"
            with open(temp_file_path, "w", encoding="utf-8") as f:
                for item in tqdm(split_data, desc=f"Writing {split_name}"):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rename(temp_file_path, file_path)
            print(f"Successfully saved {len(split_data)} records to {file_path}")
            
    except Exception as e:
        print(f"Error processing PersonaChat: {e}")

def main():
    print("Starting dataset download process...")
    check_dir(DATA_DIR)
    
    # 1. DailyDialog
    download_dailydialog_custom()
    
    # 2. PersonaChat (true-cased)
    download_personachat_custom()
    
    # 3. EmpatheticDialogues
    save_hf_to_jsonl("facebook/empathetic_dialogues", "empathetic_dialogues")
    
    # 4. LIGHT (Hugging Face community version)
    save_hf_to_jsonl("dap-exp/light_dialog", "light", rename_splits={"valid": "validation"})
    
    # 5. LIGHT (Official ParlAI Pickles)
    parlai_light_urls = {
        "light-dialog-processed-small7.pkl": "http://parl.ai/downloads/light/light-dialog-processed-small7.pkl",
        "light-unseen-processed2.pkl": "http://parl.ai/downloads/light/light-unseen-processed2.pkl",
        "light-environment.pkl": "http://parl.ai/downloads/light/light-environment.pkl"
    }
    
    for filename, url in parlai_light_urls.items():
        # Rename destination filename slightly if needed
        dest_filename = filename.replace("-processed-small7", "_data").replace("-unseen-processed2", "_unseen_data").replace("-environment", "_environment")
        download_url_to_file(url, dest_filename, "light")

    print("\n=======================================================")
    print("All dataset downloads completed!")
    print(f"Datasets are saved under: {DATA_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    main()
