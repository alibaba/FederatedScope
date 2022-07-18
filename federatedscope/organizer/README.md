# Organizer

Organizer is implemented by [Celery](https://docs.celeryq.dev/en/latest/) and [Redis](https://redis.io/) as Broker, so you should install it before using.

## Installation

```bash
python -m pip install celery[redis]

# For Server
docker run -d -p 6379:6379 redis
```

