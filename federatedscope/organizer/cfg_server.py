server_ip = '172.23.27.236'
broker = f'redis://{server_ip}:6379/0',
backend = f'redis://{server_ip}'

result_backend = 'redis://'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Oslo'
enable_utc = True
task_annotations = {'tasks.add': {'rate_limit': '10/m'}}
