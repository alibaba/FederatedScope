package com.example.fsandroid.utils;

public class Log {
    private static final String TAG = "FS-DEVICE";

    private Log() {}

    public static void d(String log){
        android.util.Log.d(Log.TAG, log);
    }
}
