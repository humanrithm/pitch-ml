import pandas as pd

def write_mot_file(
        data: pd.DataFrame,
        output_file: str
    ) -> None:
    """
    Write a DataFrame to a .mot file format used by OpenSim.

    Parameters:
    data (pd.DataFrame): DataFrame containing motion data with columns for time and coordinates.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # ensure the DataFrame has the correct number of columns
    if data.shape[1] != 11:
        raise ValueError("DataFrame must have exactly 11 columns.")

    # Prepare the header
    header = [
            "Coordinates",
            "version=1",
            f"nRows={data.shape[0]}",
            "nColumns=11",
            "inDegrees=yes",
            "",
            "Units are S.I. units (second, meters, Newtons, ...)",
            "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).",
            "",
            "endheader"
        ]

    with open(output_file, "w") as f:
        for line in header:
            f.write(line + "\n")
        # Write column headers
        f.write('\t'.join(data.columns) + '\n')
        # Write data rows with required spacing
        for row in data.itertuples(index=False):
            row_str = f"{' '*6}{row[0]:.4f}"
            for val in row[1:]:
                row_str += f"\t{' '*6}{val:.6f}"
            f.write(row_str + "\n")



