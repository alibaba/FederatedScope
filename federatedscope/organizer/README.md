# Organizer

Organizer is implemented by [Celery](https://docs.celeryq.dev/en/latest/) and [Redis](https://redis.io/) as Broker, so you should install it before using.

## Installation

```bash
# For FS
pip install -e .[dev]

# For Celery 
python -m pip install celery[redis]

# For Server
docker run -d -p 6379:6379 redis
```

## RUN
```bash
# For Server
celery -A server worker --loglevel=info

# For Client
python federatedscope/organizer/client.py
```

