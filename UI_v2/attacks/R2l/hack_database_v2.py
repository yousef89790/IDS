import csv

# open Excel ( Database )
import hashlib

# Encrept Password 
from urllib.request import urlopen

# open Url of Previous Password 

from plyer import notification
import time

def hash(password):

    result = hashlib.sha256(password.encode())

# convert Password into Hashed

    return result.hexdigest()

    # conver to Hexa digit


def get_wordlist(url):
    try:
        with urlopen(url) as f:
            wordlist = f.read().decode('utf-8').splitlines()

            # file Reader to download the file And split it into lines 

            return wordlist
    except Exception as e:

        # if  failed 
        
        print(f'failed to get wordlist: {e}')
        exit(1)


def get_users(path):
    try:

        result = []

        with open(path) as f:

            reader = csv.DictReader(f, delimiter=',')

            # Read the Path
            for row in reader:

                result.append(dict(row))

                #each Row With
            return result
    except Exception as e:
        print(f'failed to get users: {e}')
        exit(1)


def get_hashed_table(path):
    try:
        result = []
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                result.append(dict(row))
            return result
    except Exception as e:
        print(f'failed to get hashed table: {e}')
        exit(1)


def match_hash(users, hashed_table):
    for user in users:
        password_hash = hash(user['password'])
        for row in hashed_table:
            if password_hash == row['hash']:
                print(
                    f'username: {user["username"]}, password {row["password"]}')


if __name__ == '__main__':
    WORDLIST_URL = 'https://raw.githubusercontent.com/berzerk0/Probable-Wordlists/2df55facf06c7742f2038a8f6607ea9071596128/Real-Passwords/Top12Thousand-probable-v2.txt'
    DATABASE_PATH = 'database.csv'
    HASHED_TABLE_PATH = 'hashed_table.csv'

    notification.notify(
                title = "A data Breach ",
                message="Data Breach occuring" ,
            
                # displaying time
                timeout=4
            )
            # waiting time
    time.sleep(3)

    users = get_users(DATABASE_PATH)
    hashed_table = get_hashed_table(HASHED_TABLE_PATH)
    match_hash(users, hashed_table)
