import pandas as pd
import os
import glob

# Configuration
SOURCE_FOLDER = 'src/main/java/data'

def normalize_by_max_gap(folder_path:str)->None:
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print("No CSV files found.")
        return

    for file_path in csv_files:
        try:
            # 1. Load data with high precision
            df = pd.read_csv(file_path, float_precision='high')
            
            # Ensure trade_time is numeric for the gap calculation
            df['trade_time'] = pd.to_numeric(df['trade_time'])
            
            # 2. Find the largest gap between adjacent rows
            # .diff() calculates: row[n] - row[n-1]
            max_gap = df['trade_time'].diff().max()
            
            if pd.isna(max_gap) or max_gap <= 0:
                print(f"Skipping {os.path.basename(file_path)}: Not enough data to find a gap.")
                continue

            interval_str = f"{int(max_gap)}ms"
            print(f"File: {os.path.basename(file_path)} | Max Gap Found: {interval_str}")

            # 3. Prepare for resampling
            df['trade_time'] = pd.to_datetime(df['trade_time'], unit='ms')
            df.set_index('trade_time', inplace=True)
            
            # 4. Resample using the max gap duration
            # This ensures segments are at least as large as the biggest jump in your data
            normalized_df = df.resample(interval_str).last()
            
            # 5. Clean up
            normalized_df = normalized_df.dropna(how='all').reset_index()
            
            # 6. Convert back to millisecond integers
            normalized_df['trade_time'] = (normalized_df['trade_time'].astype('int64') // 10**6)
            
            # 7. Reorder columns
            cols = ['trade_id', 'trade_time', 'price', 'size', 'side']
            normalized_df = normalized_df[cols]
            
            # 8. Save back to CSV
            normalized_df.to_csv(file_path, index=False)
            
            print(f"Successfully normalized {os.path.basename(file_path)} using {interval_str} intervals.")

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    normalize_by_max_gap(SOURCE_FOLDER)