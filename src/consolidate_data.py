import pandas as pd
import glob
import os
import re

# Paths
VAL_RESULTS_PATH = '/Users/lifudu/Desktop/FYP/GitHub/data/validation/validation_results.csv'
TXT_LOG_PATH = '/Users/lifudu/Desktop/FYP/GitHub/Shared with FYC students.txt'
NEXT_FORM_GLOB = '/Users/lifudu/Desktop/FYP/GitHub/results/next_formulations/*/next_formulations.csv'
OUTPUT_PATH = '/Users/lifudu/Desktop/FYP/GitHub/results/Final_Experiment_Validation_Summary.csv'

def parse_txt_log(path):
    """Parses the text log to extract EXP IDs and their predicted viability."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple regex to find blocks of experiments
    # Example:
    # 1. 304.7mM ectoin + 1.55M ethylene_glycol
    # a. Predicted viability: 36.6% ± 25.5%
    
    # We'll regex search for the predicted viability near the formula names
    # Mar 3 section starts after "Mar.3"
    # Mar 24 section starts after "Mar.24"
    
    # Actually, it's easier to map by the formula strings since they are unique-ish
    # We'll normalize the strings (lowercase, space stripping)
    
    predictions = {}
    
    # Pattern to match: [Formula Line] followed by [Predicted Line]
    # We'll look for lines starting with "a. Predicted viability:"
    lines = content.split('\n')
    current_formula = None
    for line in lines:
        line = line.strip()
        # Look for formula lines like "1. 33.0mM dmso + ..."
        if re.match(r'^\d+\.\s+', line):
            current_formula = re.sub(r'^\d+\.\s+', '', line).lower()
        elif "Predicted viability:" in line and current_formula:
            match = re.search(r'([\d\.]+)%', line)
            if match:
                val = float(match.group(1))
                predictions[current_formula] = val
                current_formula = None
    return predictions

def consolidate():
    # 1. Load Validation Results
    val_df = pd.read_csv(VAL_RESULTS_PATH)
    
    # 2. Load all CSV predictions
    csv_preds = []
    for f in glob.glob(NEXT_FORM_GLOB):
        df = pd.read_csv(f)
        csv_preds.append(df)
    all_csv_preds = pd.concat(csv_preds) if csv_preds else pd.DataFrame()
    
    # 3. Parse TXT log predictions
    txt_preds = parse_txt_log(TXT_LOG_PATH)
    
    final_rows = []
    for _, row in val_df.iterrows():
        exp_id = row['experiment_id']
        measured = row['viability_measured']
        notes = str(row['notes']).lower().strip()
        
        predicted = None
        
        # Try to find in CSVs first (more precise)
        if not all_csv_preds.empty:
            # Match by numeric overlap if possible, or string match
            # For simplicity, we'll use the 'formulation' column in next_formulations.csv
            match_csv = all_csv_preds[all_csv_preds['formulation'].str.lower().str.strip() == notes]
            if not match_csv.empty:
                predicted = match_csv.iloc[0]['predicted_viability']
        
        # If not found, try TXT log
        if predicted is None:
            # Try fuzzy matching or exact normalized match
            if notes in txt_preds:
                predicted = txt_preds[notes]
            else:
                # Try partial match (sometimes 'has' vs 'hsa' or '_' vs ' ')
                notes_clean = notes.replace('_', ' ')
                for pk, pv in txt_preds.items():
                    pk_clean = pk.replace('_', ' ')
                    if notes_clean == pk_clean or notes_clean in pk_clean or pk_clean in notes_clean:
                        predicted = pv
                        break
        
        error = measured - predicted if predicted is not None else None
        
        final_rows.append({
            'Experiment ID': exp_id,
            'Formulation Details': row['notes'],
            'Predicted Viability (%)': round(predicted, 2) if predicted is not None else "N/A",
            'Measured Viability (%)': round(measured, 2),
            'Error (%)': round(error, 2) if error is not None else "N/A"
        })
    
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Grand consolidation complete. File saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    consolidate()
