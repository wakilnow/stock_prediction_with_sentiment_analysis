import argparse
import pandas as pd
import nlpaug.augmenter.word as naw
import os
from tqdm import tqdm

import transformers
# Monkey-patch to fix nlpaug compatibility with newer transformers
if not hasattr(transformers.PreTrainedTokenizer, '_convert_token_to_id'):
    transformers.PreTrainedTokenizer._convert_token_to_id = lambda self, token: self.convert_tokens_to_ids(token)
if not hasattr(transformers.PreTrainedTokenizerFast, '_convert_token_to_id'):
    transformers.PreTrainedTokenizerFast._convert_token_to_id = lambda self, token: self.convert_tokens_to_ids(token)


def get_column_case_insensitive(df, col_name):
    for c in df.columns:
        if c.lower() == col_name.lower():
            return c
    return None

def main():
    parser = argparse.ArgumentParser(description="Augment Financial News Data")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output augmented CSV")
    parser.add_argument("--lang", choices=["en", "ar"], required=True, help="Language (en or ar)")
    parser.add_argument("--copies", type=int, default=2, help="Number of synthetic copies per headline")
    parser.add_argument("--aug_p", type=float, default=0.15, help="Percentage of words to replace (max 0.15 recommended for finance)")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    
    title_col = get_column_case_insensitive(df, "title")
    
    if not title_col:
        raise ValueError("Could not find 'title' or 'Title' column in CSV")

    print(f"Initializing contextual word embedder for {args.lang.upper()}...")
    if args.lang == 'en':
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="substitute", aug_p=args.aug_p, device='cpu'
        )
    else:
        aug = naw.ContextualWordEmbsAug(
            model_path='asafaya/bert-base-arabic', action="substitute", aug_p=args.aug_p, device='cpu'
        )

    augmented_rows = []
    
    print(f"Augmenting {len(df)} rows with {args.copies} copies each...")
    # Add original data
    for _, row in df.iterrows():
        new_row = row.to_dict()
        new_row["is_augmented"] = False
        augmented_rows.append(new_row)

    # Add augmented data
    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_text = str(row[title_col])
        if pd.isna(original_text) or len(original_text.strip()) < 5:
            continue
            
        try:
            # Generate n copies
            augmented_texts = aug.augment(original_text, n=args.copies)
            if isinstance(augmented_texts, str):
                augmented_texts = [augmented_texts]
                
            for text in augmented_texts:
                new_row = row.to_dict()
                new_row[title_col] = text
                new_row["is_augmented"] = True 
                augmented_rows.append(new_row)
        except Exception as e:
            # If augmentation fails for a specific row, skip
            pass

    df_out = pd.DataFrame(augmented_rows)
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved {len(df_out)} total rows (Original + Augmented) to {args.output}")

if __name__ == "__main__":
    main()
