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

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

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

def cohere_embedding(content):
    global co
    response = co.embed([content]).embeddings
    vector = response[0]
    return vector

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

def load_convo():
    files = os.listdir('database_conversation_logs')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('database_conversation_logs/{}'.format(file))
        result.append(data)    
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered

def make_dialog(arr_convo):
    string_dialog=""
    for convo in arr_convo:
        string_dialog += "{}: {}\n".format(convo["User"],convo["message"])
    return string_dialog

def put_into_json_file(arr_convo,summary):
    global counter
    arr_user = list(set([i["User"] for i in arr_convo]))
    time_start = min([i["time"] for i in arr_convo])
    timestring_start = timestamp_to_datetime(time_start)
    time_end = max([i["time"] for i in arr_convo])
    timestring_end = timestamp_to_datetime(time_end)
    time_difference = time_end-time_start
    vector_embedding = cohere_embedding(summary)
    payload = {"User":arr_user,"summary":summary,"time_start":time_start,"timestring_start":timestring_start,"time_end":time_end,"timestring_end":timestring_end,"time_difference":time_difference,"vector":vector_embedding}
    filename='log_{}_summary_{}.json'.format(time_start,str(arr_user))
    filepath="database_summary_logs/{}".format(filename)
    print("{}. SAVED: {}".format(counter,filepath))
    save_json(filepath,payload)

if __name__ == "__main__":
    api_key = "zxjYWKfG5KYtuTwufTbJuPsmcjwt1j7psV44Pao1"
    co = cohere.Client(api_key)
    counter = 0
    convo_logs=load_convo()
    
    temp_convo_logs = list()
    for convo in convo_logs:
        if len(temp_convo_logs)==5:
            counter += 1
            pretty_string_dialog=make_dialog(temp_convo_logs)
            prompt = open_file("prompt-make_summary_logs.txt").replace('<<CONVO>>', pretty_string_dialog)
            summary = cohere_completion(prompt)
            put_into_json_file(temp_convo_logs,summary)
            temp_convo_logs = list()
        else:
            temp_convo_logs.append(convo)