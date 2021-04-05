import pandas as pd
import numpy as np
import requests
import string

from bs4 import BeautifulSoup


class StatsCrawler:
    """Crawls stats of all UFC Fighter from official homepage: http://ufcstats.com/statistics/fighters."""

    def __init__(self):
        chars = string.ascii_lowercase
        fighters = []
        for p in range(7):
            if p == 1:
                for c in chars:
                    url = "http://ufcstats.com/statistics/fighters?char=" + c
                    html = requests.get(url)
                    info = BeautifulSoup(html.text, "html.parser")
                    for i in info.find_all("td", {"class": "b-statistics__table-col"}):
                        links = i.find_all("a", href=True)
                        for link in links:
                            fighters.append(link["href"])
            else:
                for c in chars:
                    url = (
                        "http://ufcstats.com/statistics/fighters?char="
                        + c
                        + "&page="
                        + str(p)
                    )
                    html = requests.get(url)
                    info = BeautifulSoup(html.text, "html.parser")
                    for i in info.find_all("td", {"class": "b-statistics__table-col"}):
                        links = i.find_all("a", href=True)
                        for link in links:
                            fighters.append(link["href"])

        self.unique_fighters = list(set(fighters))
        self.fighter_stats = pd.DataFrame()

    def crawl_stats(self):
        """Crawls all relevant stats for each fighter."""
        for i, fighter in enumerate(self.unique_fighters):

            # Fighter‘s Bio
            fighter_homepage = requests.get(fighter)
            info_fighter = BeautifulSoup(fighter_homepage.text, "html.parser")

            self.fighter_stats.loc[i, "Name"] = info_fighter.find_all(
                ("span", {"class": "b-content__title-highlight"})
            )[0].text.strip()
            record = info_fighter.find_all(
                ("span", {"class": "b-content__title-highlight"})
            )[1].text.strip()
            self.fighter_stats.loc[i, "Record"] = record

            self.fighter_stats.loc[i, "Wins"] = int(record.split(":")[1].split("-")[0])
            self.fighter_stats.loc[i, "Losses"] = int(
                record.split(":")[1].split("-")[1]
            )
            if len(record.split(":")[1].split("-")[-1]) == 1:
                self.fighter_stats.loc[i, "Draws"] = int(
                    record.split(":")[1].split("-")[-1]
                )
            else:
                self.fighter_stats.loc[i, "Draws"] = int(
                    record.split(":")[1].split("-")[-1].split(" ")[0]
                )

            # Fighter‘s Stats
            stats = info_fighter.find_all(
                "li",
                {"class": "b-list__box-list-item b-list__box-list-item_type_block"},
            )

            height = stats[0].text.split()[1:]
            if len(height) != 1:
                cm_feets = int(height[0].split("'")[0]) / 0.032808
                cm_inches = (int(height[1].split('"')[0]) / 12) / 0.032808
                self.fighter_stats.loc[i, "Height_cm"] = cm_feets + cm_inches
            else:
                self.fighter_stats.loc[i, "Height_cm"] = np.nan

            weight = stats[1].text.strip().split(":")[-1].strip().split()[0]
            if weight == "--":
                self.fighter_stats.loc[i, "Weight_lbs"] = np.nan
            else:
                self.fighter_stats.loc[i, "Weight_lbs"] = int(weight)

            reach = stats[2].text.split(":")[1].strip().split('"')[0]
            if reach == "--":
                self.fighter_stats.loc[i, "Reach_inch"] = np.nan
            else:
                self.fighter_stats.loc[i, "Reach_inch"] = int(reach)

            self.fighter_stats.loc[i, "Stance"] = (
                stats[3].text.strip().split(":")[1].strip()
            )

            debut = stats[4].text.strip().split(",")
            if len(debut) == 1:
                self.fighter_stats.loc[i, "Debut"] = np.nan
            else:
                self.fighter_stats.loc[i, "Debut"] = int(debut[1])

            slpm = stats[5].text.split(":")[1].strip()
            if slpm == "--":
                self.fighter_stats.loc[i, "SLpM"] = np.nan
            else:
                self.fighter_stats.loc[i, "SLpM"] = float(slpm)

            str_acc = stats[6].text.split(":")[1].strip().split("%")[0]
            if str_acc == "--":
                self.fighter_stats.loc[i, "StrAcc"] = np.nan
            else:
                self.fighter_stats.loc[i, "StrAcc"] = int(str_acc) / 100

            sapm = stats[7].text.split(":")[1].strip()
            if sapm == "--":
                self.fighter_stats.loc[i, "SApM"] = np.nan
            else:
                self.fighter_stats.loc[i, "SApM"] = float(sapm)

            str_def = stats[8].text.split(":")[1].strip().split("%")[0]
            if str_def == "--":
                self.fighter_stats.loc[i, "StrDef"] = np.nan
            else:
                self.fighter_stats.loc[i, "StrDef"] = int(str_def) / 100

            tda = stats[10].text.split(":")[1].strip()
            if tda == "--":
                self.fighter_stats.loc[i, "TD_Avg"] = np.nan
            else:
                self.fighter_stats.loc[i, "TD_Avg"] = float(tda)

            tdac = stats[11].text.split(":")[1].strip().split("%")[0]
            if tdac == "--":
                self.fighter_stats.loc[i, "TD_Acc"] = np.nan
            else:
                self.fighter_stats.loc[i, "TD_Acc"] = int(tdac) / 100

            tdd = stats[12].text.split(":")[1].strip().split("%")[0]
            if tdd == "--":
                self.fighter_stats.loc[i, "TD_Def"] = np.nan
            else:
                self.fighter_stats.loc[i, "TD_Def"] = int(tdd) / 100

            suba = stats[13].text.split(":")[1].strip()
            if suba == "--":
                self.fighter_stats.loc[i, "Sub_Avg"] = np.nan
            else:
                self.fighter_stats.loc[i, "Sub_Avg"] = float(suba)

    def return_stats(self):
        return self.fighter_stats


class FightsCrawler(StatsCrawler):
    """Crawls fighting data for each fighter and their results.

    Args:
        StatsCrawler (object): Object that is used for inheritance.
    """

    def __init__(self):
        super().__init__()
        self.d = {
            "Fighter": [],
            "Opponent": [],
            "Result": [],
            "Fighters_Win": [],
            "Opponents_Win": [],
        }

    def crawl_fights(self):
        """Crawls all fights for each fighter and their results."""

        for fighter in self.unique_fighters:

            fighter_homepage = requests.get(fighter)
            info_fighter = BeautifulSoup(fighter_homepage.text, "html.parser")

            names = []
            for n in info_fighter.find_all("a", {"class": "b-link b-link_style_black"}):
                txt = n.text.strip()
                if not (
                    "vs." in txt
                    or any(char.isdigit() for char in txt)
                    or "UFC" in txt
                    or "Strikeforce" in txt
                    or ":" in txt
                    or "Affliction" in txt
                    or "WCFC" in txt
                    or "EliteXC" in txt
                    or "PRIDE" in txt
                    or "Destiny" in txt
                    or "WFA" in txt
                    or "Sengoku" in txt
                    or "IFL" in txt
                    or "DREAM" in txt
                    or "Ultimate" in txt
                    or "Bushido" in txt
                    or "BodogFight" in txt
                    or " - " in txt
                    or "UCC" in txt
                ):
                    names.append(txt)

            for enu, i in enumerate(names):
                if enu % 2 == 0:
                    self.d["Fighter"].append(i)
                else:
                    self.d["Opponent"].append(i)
            assert len(self.d["Fighter"]) == len(
                self.d["Opponent"]
            ), "Fighter and Opponent must have same length."

            for j in info_fighter.find_all("i", {"class": "b-flag__text"}):
                if j.text != "next":
                    self.d["Result"].append(j.text)
            assert len(self.d["Fighter"]) == len(
                self.d["Result"]
            ), "Fighter, Opponent and Result must have same length."

        for r in self.d["Result"]:
            if r == "loss":
                self.d["Fighters_Win"].append(0)
                self.d["Opponents_Win"].append(1)
            elif r == "win":
                self.d["Fighters_Win"].append(1)
                self.d["Opponents_Win"].append(0)
            elif r == "draw":
                self.d["Fighters_Win"].append(1)
                self.d["Opponents_Win"].append(1)
            elif r == "nc":
                self.d["Fighters_Win"].append(-1)
                self.d["Opponents_Win"].append(-1)

        assert len(self.d["Fighters_Win"]) == len(
            self.d["Result"]
        ), "Not all results were considered."

    def return_fights(self, frame=True):
        """Returns data as dataframe or dictionary. By default, a dataframe is used.

        Args:
            frame (bool, optional): Data criterium, if true returns dataframe. Defaults to True.

        Returns:
            pd.DataFrame: Fights' data as dataframe.
            dictionary: Fights' data as dictionary.
        """
        if frame:
            rslts = pd.DataFrame(self.d)
            return rslts
        else:
            return self.d


if __name__ == "__main__":
    # Execution presented in the Notebook "crawler"
    pass
