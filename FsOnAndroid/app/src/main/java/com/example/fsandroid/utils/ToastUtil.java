package com.example.fsandroid.utils;

import android.content.Context;
import android.widget.Toast;

public class ToastUtil {

    private ToastUtil() {}

    public synchronized static void showToast(Context context, String str, int duration) {
        Toast toast = Toast.makeText(context, str, duration);
        toast.show();
    }
}
