# Organizer

Organizer is implemented by [Celery](https://docs.celeryq.dev/en/latest/) and [Redis](https://redis.io/) as Broker, so you should install it before using.

## Installation Dependencies

```bash
# For Server
pip install -e .[dev]
python -m pip install celery[redis]
docker run -d -p 6379:6379 redis

# For Client
Not full login ssh, conduct `source ~/.bashrc` first.
** If not running interactively, `source ~/.bashrc` might fail **
** due to: `[ -z "$PS1" ] && return`, please comment this line **
```

## RUN
```bash
# For Server
celery -A server worker --loglevel=info

## For multi-worker
#celery multi start w1 -A server -l info
#celery multi start w2 -A server -l info
#...

# For Client
python federatedscope/organizer/client.py
```

