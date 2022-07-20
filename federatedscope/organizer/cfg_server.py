result_backend = 'redis://'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Oslo'
enable_utc = True
task_annotations = {'tasks.add': {'rate_limit': '10/m'}}
