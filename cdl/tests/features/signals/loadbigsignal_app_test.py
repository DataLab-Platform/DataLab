# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Load big signal application test:

- Generate a big signal (open an existing CSV file and create a dummy CSV file by
  duplicating the data several times)

- Load the big signal in the signal panel
"""

# guitest: show

import os.path as osp

import numpy as np
import pandas as pd
import pytest

from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.utils.tests import CDLTemporaryDirectory


def create_large_random_dataframe(nrows: int, ncols: int) -> pd.DataFrame:
    """Create a large DataFrame with random data.

    Args:
        nrows: The number of rows to generate.
        ncols: The number of columns to generate.
    """
    # The number of initial random rows to generate
    initial_rows = min(nrows, 1000)

    # Generate the initial "SignalXX" data with random values
    initial_signal_data = np.random.rand(initial_rows, ncols)

    # Duplicate the "SignalXX" data to fill the desired number of rows
    full_signal_data = np.tile(
        initial_signal_data,
        (nrows // initial_rows + (1 if nrows % initial_rows > 0 else 0), 1),
    )

    # If nrows is not an exact multiple of initial_rows, trim the excess rows
    full_signal_data = full_signal_data[:nrows, :]

    # Generate the "Time" column as a sequential array from 0 to nrows-1
    time_data = np.arange(nrows).reshape(-1, 1)

    # Combine the "Time" column with the "SignalXX" data
    full_data = np.hstack((time_data, full_signal_data))

    # Prepare the column names
    column_names = ["Time"] + [f"Signal{str(i).zfill(2)}" for i in range(1, ncols + 1)]

    # Create and return the DataFrame
    df = pd.DataFrame(full_data, columns=column_names)
    return df


@pytest.mark.parametrize("nrows, ncols", [(10000, 16)])
def test_loadbigsignal_app(nrows: int, ncols: int) -> None:
    """Load big signal application test"""
    with CDLTemporaryDirectory() as tmpdir:
        with cdltest_app_context() as win:
            execenv.print("Loading big signal application test:")
            execenv.print("  - Working in temporary directory:", tmpdir)
            execenv.print(f"  - Creating a big dataset ({nrows} rows, {ncols} columns)")
            df = create_large_random_dataframe(nrows, ncols)
            big_csv = osp.join(tmpdir, "big.csv")
            df.to_csv(big_csv, index=False)
            execenv.print("  - Loading the big signal in the signal panel")
            panel = win.signalpanel
            panel.load_from_files([big_csv])
            execenv.print("  - Saving the big signal")
            panel.save_to_files([big_csv.replace(".csv", "_copy.csv")])
            execenv.print("OK")


if __name__ == "__main__":
    test_loadbigsignal_app(1000000, 16)
