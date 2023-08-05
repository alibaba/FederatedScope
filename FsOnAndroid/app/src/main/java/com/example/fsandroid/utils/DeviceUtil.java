package com.example.fsandroid.utils;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkCapabilities;
import android.net.NetworkInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.telephony.TelephonyManager;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DeviceUtil {
    private DeviceUtil() {}

    public static Map<String, Object> getDeviceInfo(Context context, String server_ip) {
        Map<String, Object> deviceInfo = new HashMap<>();
        try {
            deviceInfo.put("OS", System.getProperty("os.version")+"("+android.os.Build.VERSION.INCREMENTAL + ")");
            deviceInfo.put("OS API LEVEL", android.os.Build.VERSION.SDK_INT);
            deviceInfo.put("BRAND", Build.BRAND);
            deviceInfo.put("DISPLAY", Build.DISPLAY);
            deviceInfo.put("CPU_ABI", Build.CPU_ABI);
            deviceInfo.put("HARDWARE", Build.HARDWARE);
            deviceInfo.put("MANUFACTURER", Build.MANUFACTURER);
            deviceInfo.put("NUM_CPU_CORES", getNumberOfCores());
            deviceInfo.put("TOTAL_RAM", getTotalRAM());
            deviceInfo.put("NETWORK", getNetInfo(context));
            deviceInfo.put("LATENCY", getLatency(server_ip));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return deviceInfo;
    }

    public static int getLatency(String server_ip) throws IOException {
        int timeout = 10000;
        long beforeTime = System.currentTimeMillis();
        boolean flag = InetAddress.getByName(server_ip).isReachable(timeout);
        long afterTime = System.currentTimeMillis();
        int latency = (int) (afterTime - beforeTime);
        return latency;
    }

    public static String getNetworkType(Context context) {
        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo info = cm.getActiveNetworkInfo();
        if (info == null || !info.isConnected())
            return "-"; // not connected
        if (info.getType() == ConnectivityManager.TYPE_WIFI)
            return "WIFI";
        if (info.getType() == ConnectivityManager.TYPE_MOBILE) {
            int networkType = info.getSubtype();
            switch (networkType) {
                case TelephonyManager.NETWORK_TYPE_GPRS:
                    return "2G (GPRS)";
                case TelephonyManager.NETWORK_TYPE_EDGE:
                    return "2G (EDGE)";
                case TelephonyManager.NETWORK_TYPE_CDMA:
                    return "2G (CDMA)";
                case TelephonyManager.NETWORK_TYPE_1xRTT:
                    return "2G (1xRTT)";
                case TelephonyManager.NETWORK_TYPE_IDEN:     // api< 8: replace by 11
                    return "2G (IDEN)";
                case TelephonyManager.NETWORK_TYPE_GSM:      // api<25: replace by 16
                    return "2G (GSM)";
                case TelephonyManager.NETWORK_TYPE_UMTS:
                    return "3G (UMTS)";
                case TelephonyManager.NETWORK_TYPE_EVDO_0:
                    return "3G (EVDO-0)";
                case TelephonyManager.NETWORK_TYPE_EVDO_A:
                    return "3G (EVDO-A)";
                case TelephonyManager.NETWORK_TYPE_HSDPA:
                    return "3G (HSDPA)";
                case TelephonyManager.NETWORK_TYPE_HSUPA:
                    return "3G (HSUPA)";
                case TelephonyManager.NETWORK_TYPE_HSPA:
                    return "3G (HSPA)";
                case TelephonyManager.NETWORK_TYPE_EVDO_B:   // api< 9: replace by 12
                    return "3G (EVDO_B)";
                case TelephonyManager.NETWORK_TYPE_EHRPD:    // api<11: replace by 14
                    return "3G (EHRPD)";
                case TelephonyManager.NETWORK_TYPE_HSPAP:    // api<13: replace by 15
                    return "3G (HSPAP)";
                case TelephonyManager.NETWORK_TYPE_TD_SCDMA: // api<25: replace by 17
                    return "3G (TD_SCDMA)";
                case TelephonyManager.NETWORK_TYPE_LTE:      // api<11: replace by 13
                    return "4G (LTE)";
                case TelephonyManager.NETWORK_TYPE_IWLAN:    // api<25: replace by 18
                    return "4G (IWLAN)";
                case 19: // LTE_CA
                    return "4G (LTE_CA)";
                case TelephonyManager.NETWORK_TYPE_NR:       // api<29: replace by 20
                    return "5G (NR)";
                default:
                    return "?";
            }
        }
        return "?";
    }

    public static String getNetInfo(Context context) {

        if (android.os.Build.VERSION.SDK_INT >= 25) {

            ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
            //should check null because in airplane mode it will be null
            NetworkCapabilities nc = null;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                nc = cm.getNetworkCapabilities(cm.getActiveNetwork());
            }
            String net_type = getNetworkType(context);
            if (net_type.equals("WIFI")){
                // Note to allow permission in AndroidManifest.xml,
                // `<uses-permission android:name="android.permission.ACCESS_WIFI_STATE"/>`
                WifiManager wifiManager = (WifiManager) context.getSystemService(Context.WIFI_SERVICE);
                int linkSpeed = wifiManager.getConnectionInfo().getRssi();
                int level = WifiManager.calculateSignalLevel(linkSpeed, 5);
                return net_type + ", level " + level;
            }else{
                int downSpeed = nc.getLinkDownstreamBandwidthKbps();
                int upSpeed = nc.getLinkUpstreamBandwidthKbps();
                return net_type + ", upSpeed " + upSpeed + ", downSpeed " + downSpeed;
            }

        }
        else{
            return "Require minimum API 25 to get network info";
        }

    }


    public static String getTotalRAM() {
        RandomAccessFile reader = null;
        String load = null;
        DecimalFormat twoDecimalForm = new DecimalFormat("#.##");
        double totRam = 0;
        String lastValue = "";
        try {
            reader = new RandomAccessFile("/proc/meminfo", "r");
            load = reader.readLine();

            // Get the Number value from the string
            Pattern p = Pattern.compile("(\\d+)");
            Matcher m = p.matcher(load);
            String value = "";
            while (m.find()) {
                value = m.group(1);
                // System.out.println("Ram : " + value);
            }
            reader.close();

            totRam = Double.parseDouble(value);
            // totRam = totRam / 1024;

            double mb = totRam / 1024.0;
            double gb = totRam / 1048576.0;
            double tb = totRam / 1073741824.0;

            if (tb > 1) {
                lastValue = twoDecimalForm.format(tb).concat(" TB");
            } else if (gb > 1) {
                lastValue = twoDecimalForm.format(gb).concat(" GB");
            } else if (mb > 1) {
                lastValue = twoDecimalForm.format(mb).concat(" MB");
            } else {
                lastValue = twoDecimalForm.format(totRam).concat(" KB");
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Streams.close(reader);
        }

        return lastValue;
    }


    private static int getNumberOfCores() {
        if (android.os.Build.VERSION.SDK_INT >= 17) {
            return Runtime.getRuntime().availableProcessors();
        } else {
            return getNumCoresOldPhones();
        }
    }

    /**
     * Gets the number of cores available in this device, across all processors.
     * Requires: Ability to peruse the filesystem at "/sys/devices/system/cpu"
     *
     * @return The number of cores, or 1 if failed to get result
     */
    private static int getNumCoresOldPhones() {
        //Private Class to display only CPU devices in the directory listing
        class CpuFilter implements FileFilter {
            @Override
            public boolean accept(File pathname) {
                //Check if filename is "cpu", followed by a single digit number
                if (Pattern.matches("cpu[0-9]+", pathname.getName())) {
                    return true;
                }
                return false;
            }
        }

        try {
            //Get directory containing CPU info
            File dir = new File("/sys/devices/system/cpu/");
            //Filter to only list the devices we care about
            File[] files = dir.listFiles(new CpuFilter());
            //Return the number of cores (virtual CPU devices)
            return files.length;
        } catch (Exception e) {
            //Default to return 1 core
            return 1;
        }
    }
}
