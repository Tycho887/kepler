import sqlite3

def fetch_tle_data(id = None, name = None, db="database/satellites_tle.db"):
	assert id or name, "Either id or name must be provided."

	conn = sqlite3.connect(db)

	cursor = conn.cursor()

	if id:
		cursor.execute("SELECT * FROM tle_data WHERE id = ?", (id,))
	else:
		cursor.execute("SELECT * FROM tle_data WHERE name = ?", (name,))

	data = cursor.fetchone()

	conn.close()

	return data