def anonymize(info, psw):
    for key, value in info.items():
        if isinstance(value, dict):
            anonymize(info[key], psw)
        else:
            if key == psw:
                info[key] = "******"
    return info
