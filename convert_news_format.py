import pandas as pd
import argparse
import os

def convert_format(input_file, output_file):
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if expected columns are present
    expected_columns = {'Title', 'Date', 'Link'}
    if not expected_columns.issubset(set(df.columns)):
        print(f"Error: Input file must contain columns {expected_columns}")
        return
        
    # Rename columns to match the target format
    df = df.rename(columns={'Title': 'title', 'Date': 'date', 'Link': 'url'})
    
    # Convert the date column to YYYY-MM-DD format
    print("Converting date format...")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Reorder columns to exactly match the target format: date, title, url
    df = df[['date', 'title', 'url']]
    
    # Save to the specified output file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Successfully saved formatted data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert investing.com news format to mubasher format")
    parser.add_argument("--input", type=str, default="data/news_investing.com/bac_news.csv", help="Input CSV file path")
    parser.add_argument("--output", type=str, default="data/news_investing.com/bac_news_formatted.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    convert_format(args.input, args.output)
