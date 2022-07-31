import requests

from os.path import join, isdir, abspath
from os import makedirs

from tqdm import tqdm

base_url = "https://www.football-data.co.uk/mmz4281/{SEASON}/{DIVISION}.csv"

first_league_suffix = "D1"
second_league_suffix = "D2"

include_second = False

year_start = 1993  # season 93/94
year_end = 2021  # season 21/22

path_store = abspath('./data')


def season_year_to_string(year_start):
    str_start = str(int(year_start) % 100).zfill(2)
    str_end = str(int(year_start + 1) % 100).zfill(2)
    return str_start+str_end


def get_url_for_season(season, division):
    return base_url.replace("{SEASON}", season).replace("{DIVISION}", division)


def get_path_to_save(season, division):
    return join(path_store, "{}_{}.csv".format(season, division))


def download_url_to_filename(url, filename):
    r = requests.get(url, allow_redirects=True)
    with open(filename, 'w+b') as file:
        file.write(r.content)


def download_past_year(season_year, division):
    season_string = season_year_to_string(season_year)
    season_url = get_url_for_season(season_string, division)
    filename = get_path_to_save(season_string, division)
    download_url_to_filename(season_url, filename)


def download_all_past_years():
    for year in tqdm(range(year_start, year_end)):
        download_past_year(year, first_league_suffix)
        if include_second:
            download_past_year(year, second_league_suffix)


def make_dir_if_not_exist():
    if not isdir(path_store):
        makedirs(path_store)


def main():
    make_dir_if_not_exist()
    download_all_past_years()


if __name__ == '__main__':
    main()
