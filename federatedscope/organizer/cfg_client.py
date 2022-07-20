# ---------------------------------------------------------------------- #
# Lobby related (global variable stored in Redis)
# ---------------------------------------------------------------------- #
broker_url = 'redis://172.17.138.149:6379/0'
result_backend = 'redis://172.17.138.149/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Oslo'
enable_utc = True

task_annotations = {'tasks.add': {'rate_limit': '10/m'}}

# ---------------------------------------------------------------------- #
# ECS ssh related
# ---------------------------------------------------------------------- #
env_name = 'org'
root_path = 'test_org'
fs_version = '0.1.9'
