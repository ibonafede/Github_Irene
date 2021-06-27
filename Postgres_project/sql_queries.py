# DROP TABLES

songplay_table = "songplays"
user_table = "users"
song_table = "songs"
artist_table = "artists"
time_table = "time"

tables_to_drop = [songplay_table,user_table,song_table,artist_table,time_table] 


# CREATE TABLES

song_table_create = ("""CREATE TABLE IF NOT EXISTS {table_name} (song_id varchar, title varchar, artist_id varchar, year int, duration numeric);
""".format(table_name = song_table))

artist_table_create = ("""CREATE TABLE IF NOT EXISTS {table_name} (artist_id varchar, name varchar, location varchar, latitude numeric, longitude numeric)
""".format(table_name = artist_table))

time_table_create = ("""CREATE TABLE IF NOT EXISTS {table_name} ( song varchar, length int, start_time varchar, hour int, day varchar, week_of_year int, month int, year int, weekday int)
""".format(table_name = time_table))

user_table_create = ("""CREATE TABLE IF NOT EXISTS {table_name} (user_id int, firstName varchar, lastName varchar, gender varchar, level varchar)
""".format(table_name = user_table))

songplay_table_create = ("""CREATE TABLE IF NOT EXISTS {table_name} (start_time varchar, user_id int, level varchar, song_id varchar,
artist_id varchar, session_id int, location varchar, user_agent varchar);
""".format(table_name = songplay_table))

# INSERT RECORDS

cols_song = ["song_id", "title", "artist_id", "year","duration"]
song_table_insert = """INSERT INTO {table} ({columns}) \
                 VALUES (%s, %s, %s, %s, %s)""".format(table = song_table,columns=",".join(cols_song))

cols_artist = ["artist_id", "name", "location", "latitude", "longitude"]
artist_table_insert = """INSERT INTO {table} ({columns}) \
VALUES (%s, %s, %s, %s, %s )""".format(table= artist_table,columns=",".join(cols_artist))

cols_time = ["song","length","start_time", "hour", "day", "week_of_year", "month","year","weekday"]
time_table_insert = ("""INSERT INTO {table} ({columns}) \
                 VALUES (%s, %s,%s, %s, %s, %s, %s,%s, %s )
""".format(table= time_table,columns=",".join(cols_time)))

cols_user = ["user_id", "firstName", "lastName", "gender","level"]
user_table_insert = ("""INSERT INTO {table} ({columns}) \
                 VALUES (%s, %s, %s, %s, %s )
""".format(table= user_table,columns=",".join(cols_user)))

cols_songplay = ["start_time", "user_id", "level", "song_id","artist_id", "session_id", "location", "user_agent"]
values = len(cols_songplay) * ["%s"]
songplay_table_insert = ("""INSERT INTO {table} ({columns}) \
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s) \
""".format(table = songplay_table,columns=",".join(cols_songplay)))

# FIND SONGS
#song ID and artist ID based on the title, artist name, and duration of a song.
#Select the timestamp, user ID, level, song ID, artist ID, session ID, location, and user agent and set to songplay_data
song_select = """select song_id, artist_id from \
(SELECT \
 artist_id, \
 song_id, \
 duration, \
 title as song, \
 name as artist \
 from artists \
 INNER JOIN songs using(artist_id) ) A
where artist = %s \
and song = %s
and duration = %s
"""
# QUERY LISTS
#create_table_queries = [songplay_table_create, user_table_create, song_table_create, artist_table_create, time_table_create]
#drop_table_queries = [songplay_table_drop, user_table_drop, song_table_drop, artist_table_drop, time_table_drop]

create_table_queries = [user_table_create, song_table_create, artist_table_create, time_table_create,songplay_table_create]
drop_table_queries = ["DROP table if exists {}".format(table) for table in tables_to_drop]