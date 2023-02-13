# ---------------------------------------------------------------------- #
# Lobby related (global variable stored in Redis)
# ---------------------------------------------------------------------- #
server_ip = '127.0.0.1'
broker_url = f'redis://{server_ip}:6379/0'
result_backend = f'redis://{server_ip}/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Oslo'
enable_utc = True

# ---------------------------------------------------------------------- #
# ECS ssh related (To verify and launch the virtual environment)
# ---------------------------------------------------------------------- #
ENV_NAME = 'test_org'
ROOT_PATH = 'test_org'
FS_VERSION = '0.2.1'
TIMEOUT = 30
