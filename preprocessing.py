import pandas as pd
import re

def clean_and_normalize_dataset(file1_path, file2_path, output_name="preprocessed_dataset.csv"):
    # 1. Load and Merge CSVs
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df = pd.concat([df1, df2], ignore_index=True)

    # 2. Detailed Unicode Normalization Map
    # Maps redundant characters to their standard counterparts while preserving vowel orders
    normalization_map = {
        # 'Ha' variations (ሐ and ኀ to ሀ)
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
        'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
        # 'Ssa' variations (ሠ to ሰ)
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
        # 'Tsa' variations (ፀ to ጸ)
        'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ጺ': 'ጺ', 'ጻ': 'ጻ', 'ጼ': 'ጼ', 'ጽ': 'ጽ', 'ጾ': 'ጾ',
        # 'Aa' variations (ዐ to አ - Optional but recommended for consistency)
        'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'ኣ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ'
    }

    def refine_text(text):
        if not isinstance(text, str):
            text = str(text)
        
        # Requirement: Replace underscores and hyphens with a space
        # This prevents the tokenizer from seeing "word-word" as a single unknown token
        text = text.replace('_', ' ').replace('-', ' ')
        
        # Requirement: Remove noisy symbols ) ( " ! ] [ { }
        text = re.sub(r'[)(\"!\]\[{}]', '', text)
        
        # Methodology: Apply phonetic-aware Amharic Unicode Normalization
        # Iterates through text to replace specific characters found in the map
        normalized_text = ""
        for char in text:
            normalized_text += normalization_map.get(char, char)
            
        # Remove extra whitespace created during the cleaning process
        return re.sub(r'\s+', ' ', normalized_text).strip()

    # 3. Apply cleaning to both columns and ADD PREFIX to the input
    # We add the prefix AFTER the cleaning function so the cleaning rules don't accidentally alter the prefix
    df['input'] = "translate Amharic to English: " + df['input'].apply(refine_text)
    df['output'] = df['output'].apply(refine_text)
    
    # 4. Final Cleanup
    df.dropna(subset=['input', 'output'], inplace=True) # Remove rows with missing values
    df.drop_duplicates(inplace=True) # Remove identical sentence pairs
    
    # 5. Save the cleaned dataset
    df.to_csv(output_name, index=False)
    print(f"Cleaning Complete! {len(df)} rows saved to {output_name}")
    return df

# Run the function
clean_and_normalize_dataset('converted_train.csv', 'converted_train_1.csv')