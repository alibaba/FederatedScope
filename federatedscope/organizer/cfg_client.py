broker_url = 'redis://172.17.138.149:6379/0'
result_backend = 'redis://172.17.138.149/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Oslo'
enable_utc = True

task_annotations = {'tasks.add': {'rate_limit': '10/m'}}
