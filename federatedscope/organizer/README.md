# Organizer

Organizer is implemented by [Celery](https://docs.celeryq.dev/en/latest/) and [Redis](https://redis.io/) as Broker, so you should install it before using.

## Installation

```bash
# For Server
pip install -e .[org]
docker run -d -p 6379:6379 redis

# For Client
pip install -e .[org]
# Not full login ssh, will conduct `source ~/.bashrc` first.
# ** If not running interactively, `source ~/.bashrc` might fail **
# ** due to: `[ -z "$PS1" ] && return`, please comment this line **
```

## RUN

```bash
# For Server
cd federatedscope/organizer
celery -A server worker --loglevel=info

## For multi-worker
# celery multi start w1 -A server -l info
# celery multi start w2 -A server -l info
# ...

# For Client
# Modify `server_ip` in federatedscope/organizer/cfg_client.py
python federatedscope/organizer/client.py
help
```

