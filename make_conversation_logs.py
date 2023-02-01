import cohere
import numpy as np
import os
import openai
import json
from numpy.linalg import norm
import re
from time import time, sleep
from uuid import uuid4
import datetime
import os

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def cohere_completion(prompt, engine='command-xlarge-20221108', temp=0.7, top_k=0, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=[]):
    global co
    sleep(0.5)
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = co.generate(
                model=engine,
                prompt=prompt,
                max_tokens=tokens,
                temperature=temp,
                k=top_k,
                p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop_sequences=stop,
                return_likelihoods='NONE')
            text = response.generations[0].text.strip()
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "Cohere error: %s" % oops
            print('Error communicating with Cohere:', oops)
            sleep(1)

def create_dict_convo(convo):
    arr_convo=convo.split("\n\n")
    dict_convo={}
    cur_time=None
    prev_time=None
    for item in arr_convo:
        arr_find=re.findall("[A-Za-z]+:",item)
        cur_time=time()
        if len(arr_find)!=0:
            name=arr_find[0][:-1]
            message = item.split(": ")[1]
            dict_convo[cur_time]=[name,message]
        else:
            dict_convo[prev_time][1] += " "+item
            print(item)
        sleep(0.2)
        prev_time=cur_time
    return dict_convo

def create_database_conversation_logs(dict_convo):
    global counter
    counter +=1

    for time,dialog in dict_convo.items():
        name,message=(dialog[0],dialog[1])
        payload = {"User":name, "time":time,"timestring":timestamp_to_datetime(time), "message":message}
        filename='log_{}_{}.json'.format(time,name.upper())
        filepath="database_conversation_logs/{}".format(filename)
        print("SAVED: {}".format(filepath))
        save_json(filepath,payload)
    print("\n{} ======================================================\n".format(str(counter)))

if __name__ == "__main__":
    api_key = "zxjYWKfG5KYtuTwufTbJuPsmcjwt1j7psV44Pao1"
    co = cohere.Client(api_key)
    counter = 0
    file = open("topics-make_conversation_logs.txt","r")
    for topic in file:
        topic = topic.rstrip()
        prompt = open_file("prompt-make_conversation_logs.txt").replace('<<TOPIC>>', topic)
        response = cohere_completion(prompt)
        convo = "Hanif: "+response
        create_database_conversation_logs(create_dict_convo(convo))
