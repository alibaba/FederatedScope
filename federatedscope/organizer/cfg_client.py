# ---------------------------------------------------------------------- #
# Lobby related (global variable stored in Redis)
# ---------------------------------------------------------------------- #
broker_url = 'redis://172.23.27.236:6379/0'
result_backend = 'redis://172.23.27.236/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Oslo'
enable_utc = True

# ---------------------------------------------------------------------- #
# ECS ssh related
# ---------------------------------------------------------------------- #
env_name = 'test_org'
root_path = 'test_org'
fs_version = '0.1.9'
