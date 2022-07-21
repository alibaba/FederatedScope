# ---------------------------------------------------------------------- #
# Lobby related (global variable stored in Redis)
# ---------------------------------------------------------------------- #
server_ip = '172.23.27.236'
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
env_name = 'test_org'
root_path = 'test_org'
fs_version = '0.1.9'
