import numpy as np
from sgp4.api import Satrec
from datetime import datetime, timedelta
from sgp4.api import WGS72
from abc import ABC, abstractmethod

class SatellitePropagator:
	"""
	A class to propagate a satellite's orbit using TLE data.
	"""
	def __init__(self, tle_data: tuple):
		"""
		Initializes the SatellitePropagator with TLE data.
		
		Args:
			tle_data (tuple): A tuple containing the two TLE lines.
		"""
		self.id = tle_data[0]
		self.name = tle_data[1]
		self.satellite = Satrec.twoline2rv(tle_data[2],tle_data[3], WGS72)
		self.start_time = datetime.fromisoformat(tle_data[4])
	
	def propagate(self, minutes_into_future: float, step_size: float):
		"""
		Propagates the satellite's position into the future at regular intervals.
		
		Args:
			start_time (datetime): The starting time for propagation.
			minutes_into_future (float): Minutes to propagate into the future.
			step_size (float): Time step size in minutes.
			
		Returns:
			dict: A dictionary containing arrays of time, position (x, y, z in km), and velocity (vx, vy, vz in km/s).
		"""
		# Initialize arrays to store results
		times = []
		positions = []
		velocities = []
		
		# Propagate the satellite at regular intervals
		current_time = self.start_time
		while (current_time - self.start_time).total_seconds() / 60 <= minutes_into_future:
			# Convert datetime to Julian Date (SGP4 requires Julian Date format)
			jd, fr = self._datetime_to_julian(current_time)
			
			# Propagate the satellite using SGP4
			error_code, position, velocity = self.satellite.sgp4(jd, fr)
			
			# Check for errors
			if error_code != 0:
				raise RuntimeError(f"SGP4 propagation error code: {error_code}")
			
			# Append results to arrays
			times.append(current_time)
			positions.append(position)
			velocities.append(velocity)
			
			# Update current time
			current_time += timedelta(minutes=step_size)
		
		# Return results as arrays
		return {
			"times": np.array(times),
			"positions_km": np.array(positions),  # (x, y, z) in kilometers
			"velocities_km_s": np.array(velocities)  # (vx, vy, vz) in kilometers per second
		}

	@staticmethod
	def _datetime_to_julian(date: datetime):
		"""
		Converts a datetime object to Julian Date and fractional day.
		
		Args:
			date (datetime): The datetime to convert.
		
		Returns:
			jd (float): Julian date integer part.
			fr (float): Fractional part of the day.
		"""
		jd_epoch = 2451545.0  # Julian Date at epoch 2000-01-01 12:00:00
		year = date.year
		month = date.month
		day = date.day
		hour = date.hour
		minute = date.minute
		second = date.second

		if month <= 2:
			year -= 1
			month += 12
		A = int(year / 100)
		B = 2 - A + int(A / 4)
		jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
		fractional_day = (hour + minute / 60 + second / 3600) / 24.0
		jd_int = int(jd)
		fr = jd - jd_int + fractional_day
		return jd_int, fr
