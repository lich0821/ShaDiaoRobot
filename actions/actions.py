# -*- coding: utf-8 -*-

import os
from typing import Dict, Text, Any, List, Union

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction

from .weather_api import Weather

# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


class ValidateWeatherForm(FormValidationAction):
    """校验天气查询表单"""

    def name(self) -> Text:
        return "validate_weather_form"

    @staticmethod
    def date_db() -> List[Text]:
        """支持的日期数据库"""
        return ["今天", "明天", "后天"]

    def validate_date(self, value: Text, dispatcher: CollectingDispatcher,
                      tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        """校验日期"""
        if value in self.date_db():
            return {"date": value}
        else:
            dispatcher.utter_message(response="utter_wrong_date")
            return {"date": None}

    def validate_city(self, value: Text, dispatcher: CollectingDispatcher,
                      tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        """校验城市"""
        if value:  # TODO: 使用城市列表
            return {"city": value}
        else:
            dispatcher.utter_message(response="utter_wrong_city")
            return {"city": None}


class ActionWeather(Action):
    def __init__(self) -> None:
        super().__init__()
        key = os.getenv('WEATHER_KEY')
        if not key:
            print("请配置天气API的KEY环境变量。（Linux/MacOS：export WEATHER_KEY=your_key）")
            exit(-1)

        self.weather = Weather(key)

    def name(self) -> Text:
        return "action_weather"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["date", "city"]

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
            ) -> List[Dict]:
        """Define what the form has to do after all required slots are filled"""
        city = tracker.get_slot("city")
        date = tracker.get_slot("date")

        try:
            data = self.weather.fetch(city, start=date)
        except Exception as e:
            print(e)    # TODO: log it
            msg = f"天气服务故障，请稍后再查。（免费的接口比较坑）"
            ss = [SlotSet("city", None), SlotSet("date", date)]
        else:
            msg = f"{data['location']}{date}({data['date']})的天气情况为: 白天{data['day']}, 夜晚{data['night']},气温:{data['low']}～{data['high']}℃"
            ss = [SlotSet("city", city), SlotSet("date", date)]

        dispatcher.utter_message(text=msg)

        return ss
