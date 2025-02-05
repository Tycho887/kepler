import sqlite3

db_name = "database/satellites_tle.db"

db_structure = {
    "sqlite_sequence": {
        "name": "TEXT", 
        "seq": "INTEGER"
    },
    "tle_data": {
        "id": "INTEGER PRIMARY KEY",
        "name": "TEXT",
        "line1": "TEXT",
        "line2": "TEXT",
        "timestamp": "TEXT"},
	"orbital_data": {
        "id": "INTEGER PRIMARY KEY",
        "satellite_id": "INTEGER",
        "epoch": "TEXT",
        "mean_motion": "REAL",
        "eccentricity": "REAL",
        "inclination": "REAL",
		"perigee": "REAL",
		"apogee": "REAL",
		"period": "REAL",
        "argument_of_perigee": "REAL",
        "raan": "REAL",
        "mean_anomaly": "REAL",
        "mean_motion_derivative": "REAL",
        "mean_motion_sec_derivative": "REAL",
        "bstar": "REAL",
        "rev_number": "INTEGER"
    }
}

# We create a decorator function that handles the connection to the database and closes it after the function is executed.
def connect(function):
    def wrapper(*args, **kwargs):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        result = function(cursor, *args, **kwargs)
        conn.commit()
        conn.close()
        return result
    return wrapper


@connect
def fetch_tle_data_single(cursor, id):

	id = int(id)

	cursor.execute("SELECT * FROM tle_data WHERE id = ?", (id,))

	return cursor.fetchone()

@connect
def fetch_tle_data_multiple(cursor, ids: list):

	ids = tuple(map(int, ids))

	cursor.execute("SELECT * FROM tle_data WHERE id IN {}".format(ids))

	return cursor.fetchall()

@connect
def number_of_satellites(cursor):
	cursor.execute("SELECT COUNT(*) FROM tle_data")

	return cursor.fetchone()[0]