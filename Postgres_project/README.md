Customer Problem : A startup called Sparkify wants to analyze the data they've been collecting on songs and user
activity on their new music streaming app. The analytics team is particularly interested in understanding what songs
users are listening to. Currently, they don't have an easy way to query their data, which resides in a directory of JSON
logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

Pipeline Aim:  transfers data from files in two local directories (song_data and log_data)
into a star database schema:
Fact Table:

1) songplays - records in log data associated with song plays i.e. records with page NextSong songplay_id, start_time,
   user_id, level, song_id, artist_id, session_id, location, user_agent

Dimension Tables:

1) users - users in the app user_id, first_name, last_name, gender, level
2) songs - songs in music database song_id, title, artist_id, year, duration
3) artists - artists in music database artist_id, name, location, latitude, longitude
4) time - timestamps of records in songplays broken down into specific units start_time, hour, day, week, month, year,
   weekday

Pipeline input:
song_data: json files log_data: log_files

Pipeline Description:

- create_tables.py :  drops and creates your tables. run this file to reset your tables before each time you run your
  ETL scripts.
- sql queries.py: collections of queries
- etl.py: main program. The script connects to the Sparkify database, extracts and processes the log_data and song_data,
  and loads data into the five tables.

program usage:
python create_table.py python etl.py

Additional notebooks:

- test.ipynb: notebook to test the results
- etl.ipynb: notebook to develop the pipeline program languages: Python and SQL
