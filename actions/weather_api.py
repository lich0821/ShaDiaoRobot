#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import requests


class Weather():
    def __init__(self, key, uid="", unit="c", lang="zh-Hans") -> None:
        self.key = key
        self.uid = uid
        self.unit = unit
        self.lang = lang
        self.supported_dates = {"今天": 0, "明天": 1, "后天": 2}
        self.api = "https://api.seniverse.com/v3/weather/daily.json"

    def fetch(self, location, start=0, days=1):
        params = {"key": self.key, "location": location, "language": self.lang,
                  "unit": self.unit, "start": self.supported_dates.get(start, 0), "days": days}
        # TODO: Handle error codes
        result = requests.get(self.api, params=params, timeout=2)
        r0 = result.json()["results"][0]
        ret = {"location": r0["location"]["name"],
               "date": r0["daily"][0]["date"],
               "day": r0["daily"][0]["text_day"],
               "night": r0["daily"][0]["text_night"],
               "high": r0["daily"][0]["high"],
               "low": r0["daily"][0]["low"]
               }
        return ret


if __name__ == "__main__":
    w = Weather("your key")
    default_location = "北京"
    result = w.fetch(default_location)
    print(json.dumps(result, ensure_ascii=False))
