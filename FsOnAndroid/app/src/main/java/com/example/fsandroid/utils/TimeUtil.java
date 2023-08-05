package com.example.fsandroid.utils;

import java.text.SimpleDateFormat;
import java.util.Date;

public class TimeUtil {
    private TimeUtil() {}

    public static String getTimeStamp() {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        return format.format(new Date(System.currentTimeMillis()));
    }
}
