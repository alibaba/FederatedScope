package com.example.fsandroid.utils;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;


// https://blog.csdn.net/pbm863521/article/details/78811250
public class AssetsUtil {
    private AssetsUtil() {
    }

    public static void copyAssetsDir2Phone(Context activity, String dataType) {
        String zipName = dataType + ".zip";

        InputStream is;
        ZipInputStream zis;
        try {
            String filename;
            is = activity.getAssets().open(zipName);
            zis = new ZipInputStream(new BufferedInputStream(is));
            ZipEntry ze;
            byte[] buffer = new byte[1024];
            int count;

            while ((ze = zis.getNextEntry()) != null) {
                filename = ze.getName();

                if (filename.startsWith("__")) {
                    continue;
                }

                // Need to create directories if not exists, or
                // it will generate an Exception...
                if (ze.isDirectory()) {
                    File fmd = new File(activity.getFilesDir().getAbsolutePath() + File.separator + filename);
                    fmd.mkdirs();
                    continue;
                }

                FileOutputStream fout = new FileOutputStream(activity.getFilesDir().getAbsolutePath() + File.separator + filename);
                while ((count = zis.read(buffer)) != -1) {
                    fout.write(buffer, 0, count);
                }

                fout.close();
                zis.closeEntry();
            }
            zis.close();
            Log.d("Finished unzip file " + zipName);
        }
        catch(IOException e) {
            e.printStackTrace();
            Log.d("Fail to unzip file " + zipName);
        }
    }
}
