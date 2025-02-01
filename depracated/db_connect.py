import sqlite3
from .db_structure import db_name, db_structure

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

