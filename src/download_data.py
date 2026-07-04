from pathlib  import Path
import  pandas as pd
from sklearn.datasets import fetch_california_housing



def main() :
    print("Downloading California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)
    df : pd.DataFrame = housing.frame
    outdir = Path("../data/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    output_file = outdir / "california_housing.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print("Dataset Shape:", df.shape)

    print(("First Five Rows\n"))
    print(df.head(5))









if __name__ == "__main__":
    main()

