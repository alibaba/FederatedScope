// Refer to https://github.com/hello2mao/CPUMemDemo
package com.example.fsandroid.utils;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Debug;
import android.os.Handler;
import android.os.Message;

import com.example.fsandroid.UiMsgObject;
import com.example.fsandroid.enums.UiType;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Usage:
 *  AppMonitor.getInstance().init(getApplicationContext(), 100L);
 *  AppMonitor.getInstance().start();
 */
public class AppMonitor implements Runnable {

    private volatile static AppMonitor instance = null;
    private final ScheduledExecutorService scheduler;
    private ActivityManager activityManager;
    private long freq;
    private Long lastCpuTime;
    private Long lastAppCpuTime;
    private RandomAccessFile procStatFile;
    private RandomAccessFile appStatFile;
    private Handler uiHandler;

    private AppMonitor() {
        scheduler = Executors.newSingleThreadScheduledExecutor();
    }

    public static AppMonitor getInstance() {
        if (instance == null) {
            synchronized (AppMonitor.class) {
                if (instance == null) {
                    instance = new AppMonitor();
                }
            }
        }
        return instance;
    }

    // freq为采样周期
    public void init(Context context, long freq, Handler uiHandler) {
        activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        this.freq = freq;
        this.uiHandler = uiHandler;
    }

    public void start() {
        scheduler.scheduleWithFixedDelay(this, 0L, freq, TimeUnit.MILLISECONDS);
    }

    public boolean isAlive() {
        return !scheduler.isShutdown();
    }

    public void interpret() {
        scheduler.shutdownNow();
    }

    @Override
    public void run() {
        double cpu = sampleCPU();
        double mem = sampleMemory();
//        Log.d("CPU: " + cpu + "%" + "    Memory: " + mem + "MB");
        Message msg = Message.obtain();
        msg.obj = new UiMsgObject(UiType.TRAIN_LINE_CHART, (float)mem, (float)cpu);
        uiHandler.sendMessage(msg);
    }

    private double sampleCPU() {
        int rate = 0;

        try {
            String Result;
            Process p = Runtime.getRuntime().exec("top -n 1");
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            while ((Result = br.readLine()) !=null){
                if (Result.contains("com.example.fs")) {
                    String[] info = Result.trim().replaceAll(" +"," ").split(" ");
                    return Double.valueOf(info[9]);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return rate;
    }

    private double sampleMemory() {
        double mem = 0.0D;
        try {
            // 统计进程的内存信息 totalPss
            Debug.MemoryInfo memInfo = new Debug.MemoryInfo();
            Debug.getMemoryInfo(memInfo);

            mem = memInfo.getTotalPss() / 1024D;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return mem;
    }
}