import numpy as np
from sgp4.api import Satrec
from datetime import datetime, timedelta
from sgp4.api import WGS72
from lib.utils import datetime_to_julian

class SatellitePropagator:
    """
    A class to propagate a satellite's orbit using TLE data.
    """

    def __init__(self, tle_data: tuple):
        """
        Initialize the SatellitePropagator with TLE data.

        Args:
            tle_data (tuple): A tuple containing the satellite ID, name, and two TLE lines, plus start time.
        """
        self.TLE = tle_data
        self.id = tle_data[0]
        self.name = tle_data[1]
        self.satellite = Satrec.twoline2rv(tle_data[2], tle_data[3], WGS72)
        self.start_time = datetime.fromisoformat(tle_data[4])

        self._position_data = None
        self.error_code = None

    def propagate(self, minutes_into_future: float, step_size: float):
        """
        Public method to propagate satellite position.

        Args:
            minutes_into_future (float): Time to propagate into the future in minutes.
            step_size (float): Step size for propagation in minutes.
        """
        try:
            self._propagate(minutes_into_future, step_size)
        except Exception as e:
            print(e)
            self.error_code = e

    def _propagate(self, minutes_into_future: float, step_size: float):
        """
        Internal method to propagate the satellite's orbit.

        Args:
            minutes_into_future (float): Minutes to propagate into the future.
            step_size (float): Time step size in minutes.
        """
        times, positions, velocities = [], [], []
        current_time = self.start_time

        while (current_time - self.start_time).total_seconds() / 60 <= minutes_into_future:
            jd, fr = self._datetime_to_julian(current_time)
            error_code, position, velocity = self.satellite.sgp4(jd, fr)

            if error_code != 0:
                raise RuntimeError(f"SGP4 propagation error code: {error_code}")

            times.append(current_time)
            positions.append(position)
            velocities.append(velocity)
            current_time += timedelta(minutes=step_size)

        self._position_data = {
            "times": np.array(times),
            "positions_km": np.array(positions,dtype=np.float32)
        }

    @property
    def positions(self):
        """
        Get the position data of the satellite.

        Returns:
            np.ndarray: Array of satellite positions in kilometers.
        """
        if self._position_data is None:
            raise RuntimeError("Satellite position data has not been propagated yet.")
        return self._position_data["positions_km"]

    @property
    def times(self):
        """
        Get the time steps of the satellite propagation.

        Returns:
            np.ndarray: Array of datetime objects corresponding to propagation steps.
        """
        if self._position_data is None:
            raise RuntimeError("Satellite position data has not been propagated yet.")
        return self._position_data["times"]

    @staticmethod
    def _datetime_to_julian(date: datetime):
        """
        Convert a datetime object to Julian Date and fractional day.

        Args:
            date (datetime): The datetime to convert.

        Returns:
            tuple: Julian date integer part and fractional part.
        """
        jd_epoch = 2451545.0  # Julian Date at epoch 2000-01-01 12:00:00
        year, month, day = date.year, date.month, date.day
        hour, minute, second = date.hour, date.minute, date.second

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
