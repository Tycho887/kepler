from lib.simulation import run_sim
from lib.plotting import plot_3d, plot_distance
import pandas as pd
from lib.database import fetch_tle_data_multiple
from lib.utils import build_dataset, store_dataset


if __name__ == "__main__":

	N = 200
	days = 10
	step_size = 0.1  # Step size in minutes
	daily_error = 2.0  # Error in kilometers per day

	sats, pairs, indexes = run_sim(N, days, step_size, daily_error)

	df = build_dataset(sats, pairs, indexes)

	store_dataset(df)