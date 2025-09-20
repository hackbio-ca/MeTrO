import pandas as pd
import os

excel_path = 'data/journal.pcbi.1009337.s005.xlsx'  # Change to your Excel file path
output_dir = 'data'

excel_file = pd.ExcelFile(excel_path)
for sheet_name in ['Transcriptomics data', 'Metabolomics data']:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    csv_path = os.path.join(output_dir, f"{sheet_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")