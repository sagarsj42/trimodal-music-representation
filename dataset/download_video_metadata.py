import os
import re
import sys
import json
import pickle

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def youtube_authenticate(scopes):
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    api_service_name = 'youtube'
    api_version = 'v3'
    client_secrets_file = '/home2/sagarsj42/yt8m/youtube-data-api-creds.json'
    creds = None

    if os.path.exists('/home2/sagarsj42/yt8m/token.pickle'):
        with open('/home2/sagarsj42/yt8m/token.pickle', 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            creds = flow.run_local_server(port=0)
        with open('/home2/sagarsj42/yt8m/token.pickle', 'wb') as f:
            pickle.dump(creds, f)

    return build(api_service_name, api_version, credentials=creds)


def get_video_details(youtube, **kw_args):
    return youtube.videos().list(
        part='snippet,contentDetails,statistics',
        **kw_args
    ).execute()


def obtain_video_info(video_response):
    if not video_response.get('items'):
        print('no info')
        return dict()

    items = video_response.get('items')[0]
    vid = items['id'].strip()
    snippet = items['snippet']
    statistics = items['statistics']
    content_details = items['contentDetails']

    # get infos from the snippet
    title = snippet['title'].strip()
    description = snippet['description'].strip()
    publish_time = snippet['publishedAt'].strip()
    channel_id = snippet['channelId'].strip()
    channel_title = snippet['channelTitle'].strip()
    try:
        tags = ','.join(snippet['tags']).strip()
    except:
        print('tags n.a.', vid)
        tags = ''

    # get stats infos
    try:
        view_count = int(statistics['viewCount'].strip())
    except:
        print('views n.a.', vid)
        view_count = -1
    try:
        like_count = int(statistics['likeCount'].strip())
    except:
        print('likes n.a.', vid)
        like_count = -1
    try:
        comment_count = int(statistics['commentCount'].strip())
    except:
        print('comments n.a.', vid)
        comment_count = -1
    fav_count = int(statistics['favoriteCount'].strip())

    # get duration from content details
    try:
        duration = content_details['duration']
        # duration in the form of something like 'PT5H50M15S'
        # parsing it to be something like '5:50:15'
        parsed_duration = re.search(f'PT(\d+H)?(\d+M)?(\d+S)', duration).groups()
        duration_str = ''
        for d in parsed_duration:
            if d:
                duration_str += f'{d[:-1]}:'
        duration_str = duration_str.strip(':')
    except:
        print('duration n.a.', vid, duration)
        duration_str = ''

    # get content details
    definition = content_details['definition']
    licensed_content = content_details['licensedContent']
    content_rating = json.dumps(content_details['contentRating'])
    projection = content_details['projection']
    dimension = content_details['dimension']
    caption = content_details['caption']

    info_dict = {
        'vid': vid,
        'title': title,
        'description': description,
        'publish_time': publish_time,
        'channel_id': channel_id,
        'channel_title': channel_title,
        'tags': tags,

        'view_count': view_count,
        'like_count': like_count,
        'comment_count': comment_count,
        'fav_count': fav_count,

        'duration': duration_str,
        'definition': definition,
        'licensed_content': licensed_content,
        'content_rating': content_rating,
        'projection': projection,
        'dimension': dimension,
        'caption': caption
    }

    return info_dict


data_home_dir = sys.argv[1]
splits = sys.argv[2].split(',')

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
RAW_LABELS_EXT_DATA_PATH = os.path.join(data_home_dir, 'yt8m-label-extracted')
CLEAN_LABELS_EXT_DATA_PATH = os.path.join(data_home_dir, 'yt8m-labels')
INFO_DATA_PATH = os.path.join(data_home_dir, 'yt8m-info')
INFO_EXT_IDS_PATH = os.path.join(data_home_dir, 'yt8m-extracted-ids.txt')

youtube = youtube_authenticate(SCOPES)
os.makedirs(CLEAN_LABELS_EXT_DATA_PATH, exist_ok=True)
label_data = list()
corrections = 0

for split in splits:
    with open(os.path.join(CLEAN_LABELS_EXT_DATA_PATH, f'{split}.jsonl'), 'w') as f1:
        with open(os.path.join(RAW_LABELS_EXT_DATA_PATH, f'{split}.jsonl'), 'r') as f2:
            for line in f2:
                try:
                    item = json.loads(line)
                    label_data.append(item)
                except:
                    corrections += 1
                    break_ind = line.index('}') + 1
                    item = json.loads(line[:break_ind])
                    label_data.append(item)
                    item = json.loads(line[break_ind:])
                    label_data.append(item)
            f1.write('\n'.join([json.dumps(i) for i in label_data]))

    label_data = list()
    with open(os.path.join(CLEAN_LABELS_EXT_DATA_PATH, f'{split}.jsonl'), 'r') as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)
            label_data.append(item)

    os.makedirs(INFO_DATA_PATH, exist_ok=True)
    with open(os.path.join(INFO_DATA_PATH, f'{split}.jsonl'), 'a') as info_f:
        for item in label_data:
            vid = item['vid'].strip()

            try:
                with open(INFO_EXT_IDS_PATH, 'r') as list_f:
                    done_ids = set(list_f.read().split('\n'))
                    if len(done_ids) % 1000 == 0:
                        print(f'# infos downloaded so far: {len(done_ids)}')
                    if vid in done_ids:
                        continue
            except:
                print('Starting from all ids afresh.')

            video_response = get_video_details(youtube, id=vid)
            info = obtain_video_info(video_response)
            if not info:
                info['vid'] = vid
            assert info['vid'] == vid
            info_f.write(json.dumps(info) + '\n')

            with open(INFO_EXT_IDS_PATH, 'a') as list_f:
                list_f.write(vid + '\n')
