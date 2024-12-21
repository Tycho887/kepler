import sqlite3
from .db_structure import *
from .db_connect import connect


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