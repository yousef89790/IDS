import csv
import hashlib
from urllib.request import urlopen


def get_wordlist(url):
    try:
        with urlopen(url) as f:
            wordlist = f.read().decode('utf-8').splitlines()
            return wordlist
    except Exception as e:
        print(f'failed to get wordlist: {e}')
        exit(1)


def hash(password):
    result = hashlib.sha256(password.encode())
    return result.hexdigest()


def create_hashed_table(wordlist_url, hashed_table_path):
    wordlist = get_wordlist(wordlist_url)
    try:
        with open(hashed_table_path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['password', 'hash'])
            for word in wordlist:
                writer.writerow([word, hash(word)])

    except Exception as e:
        print(f'failed to create hashed table: {e}')
        exit(1)


if __name__ == '__main__':
    WORDLIST_URL = 'https://raw.githubusercontent.com/berzerk0/Probable-Wordlists/2df55facf06c7742f2038a8f6607ea9071596128/Real-Passwords/Top1575-probable-v2.txt'
    HASHED_TABLE_PATH = 'hashed_table.csv'

    create_hashed_table(WORDLIST_URL, HASHED_TABLE_PATH)
