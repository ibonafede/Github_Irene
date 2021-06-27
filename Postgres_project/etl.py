import os
import glob
import psycopg2
import pandas as pd
from sql_queries import *
import datetime


def process_song_file(cur, filepath):

    # open song file
    df = pd.read_json(filepath, lines=True)
    #df = pd.concat(list_song_df).reset_index(drop=True)
    df.title = df.title.str.upper()
    df.artist_name = df.artist_name.str.upper()
    df["year"] = df["year"].values.astype(int)
    df["duration"] = df["duration"].values.astype(float)
    # insert song record
    song_data = df[["song_id","title","artist_id","year","duration"]]
    for row in song_data.itertuples(index=False):
        cur.execute(song_table_insert, tuple(row))
        #conn.commit()
    #cur.execute(song_table_insert, song_data)
    # insert artist record
    artist_data = df[["artist_id","artist_name","artist_location","artist_latitude","artist_longitude"]]
    for row in artist_data.itertuples(index=False):
        cur.execute(artist_table_insert, tuple(row))


def process_log_file(cur, filepath):
    # open log file
    list_log = []
    df = pd.read_json(filepath, lines=True)
    # convert in upper case
    df.artist = df.artist.str.upper()
    df.song = df.song.str.upper()
    # filter by NextSong action
    df = df[df["page"] == "NextSong"]

    # convert timestamp column to datetime
    df["ts"] = df["ts"].map(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
    
    time_dict = {}
    column_labels = ["song","length","start_time","hour", "day", "week_of_year", "month", "year","weekday"]
    for key in column_labels:
        time_dict[key] = []
    for index,item in df.iterrows():
        time_dict["song"].append(item["song"].upper())
        time_dict["length"].append(item["length"])
        row = item["ts"]
        time_dict["start_time"].append(row)
        time_dict["hour"].append(row.hour)
        time_dict["day"].append(row.day)
        time_dict["week_of_year"].append(row.isocalendar()[1])
        time_dict["month"].append(row.month)
        time_dict["year"].append(row.isocalendar()[0])
        time_dict["weekday"].append(row.isocalendar()[2])
    
    time_df = pd.DataFrame(time_dict)
    time_df["start_time"] = time_df["start_time"].map(lambda x:x.strftime('%Y%m%d_%H%M%S'))
    
    # insert time data records
    for i, row in time_df.iterrows():
        cur.execute(time_table_insert, list(row))

    # load user table
    user_df = df[["userId","firstName", "lastName","gender","level"]]

    # insert user records
    for row in user_df.itertuples(index=False):
        cur.execute(user_table_insert,tuple(list(row)) )
       

    # insert songplay records
    for index,row in df.iterrows():
        #print(row["artist"])
        cur.execute(song_select, (row["artist"], row["song"],row["length"]))
        results = cur.fetchone()

        if results:
            songid, artistid = results
            #print(results)
        else:
            songid, artistid = None, None
        #print(row)
        # insert songplay record
        if results and (row["userId"]!=""):
            songplay_data = row["ts"],row["userId"],row["level"],songid, artistid, row["sessionId"], row["location"],row["userAgent"]
            cur.execute(songplay_table_insert, (songplay_data))
            


def process_data(cur, conn, filepath, func):
    # get all files matching extension from directory
    all_files = []
    for root, dirs, files in os.walk(filepath):
        files = glob.glob(os.path.join(root,'*.json'))
        for f in files :
            all_files.append(os.path.abspath(f))

    # get total number of files found
    num_files = len(all_files)
    print('{} files found in {}'.format(num_files, filepath))

    # iterate over files and process
    for i, datafile in enumerate(all_files, 1):
        
        func(cur, datafile)
        conn.commit()
        print('{}/{} files processed.'.format(i, num_files))


def main():
    conn = psycopg2.connect("host=127.0.0.1 dbname=sparkifydb user=student password=student")
    cur = conn.cursor()

    process_data(cur, conn, filepath='data/song_data', func=process_song_file)
    process_data(cur, conn, filepath='data/log_data', func=process_log_file)

    conn.close()


if __name__ == "__main__":
    main()