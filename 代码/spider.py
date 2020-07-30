import requests
import urllib
import json
import os

with open("5-31.json", 'r') as load_f:
    load_dict = json.load(load_f)
    # print(load_dict)
    i = 0 
    for item in load_dict:
        url1 = "https://api.pubg.com/shards/steam/matches/" + item['id']
        payload = {}
        headers = {
            'Accept': 'application/vnd.api+json'
        }
        response = requests.request("GET", url1, headers=headers, data=payload)
        # print(json.loads(response.text))
        events = json.loads(response.text)
        # print(events)
        for item in events['included']:
            if (item['type'] == 'asset'):
                i = i + 1
                url2 = item['attributes']['URL']
                print(url2)
                file = 'match' + str(i) + '.json'
                LocalPath = os.path.join('./', file)
                # urllib.request.urlretrieve(url2, file)
                res = requests.get(url=url2, headers=headers)
                # json.loads(res.text)
                tmp_fp = open(LocalPath, 'w+', encoding='utf-8')
                json.dump(fp=tmp_fp, obj=json.loads(res.text))
                tmp_fp.close()
