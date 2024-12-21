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
        "timestamp": "TEXT"}
}
