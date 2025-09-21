import polars as pl
import numpy as np
# read excel file
counts = pl.read_csv("data/CCLE_RNAseq_genes_counts_20180929.gct", separator="\t", skip_rows=2, null_values=["2e+05", "1e+05"])
rpkm = pl.read_csv("data/CCLE_RNAseq_genes_rpkm_20180929.gct", skip_rows=2, separator="\t")
metabolomics = pl.read_csv("data/CCLE_metabolomics_20190502.csv")

intersected_samples = np.intersect1d(metabolomics["CCLE_ID"], rpkm.columns)
metabolomics = metabolomics.filter(pl.col("CCLE_ID").is_in(intersected_samples))
rpkm = rpkm.select(["Name"] + intersected_samples.tolist()).rename({"Name": "gene_id"})

rpkm = rpkm.transpose(include_header=True, header_name="CCLE_ID", column_names=rpkm["gene_id"].to_list())\
    .filter(pl.col("CCLE_ID")!="gene_id")

merged_df = rpkm\
    .join(metabolomics, on="CCLE_ID")

merged_df = merged_df.drop("DepMap_ID")

merged_df.write_csv("data/merged_df.csv")

# randomly sample 1% of the rows
merged_df.sample(fraction=0.01, with_replacement=False, seed=42).write_csv("data/merged_df_debug.csv")

merged_df.write_csv("data/merged_df.csv")