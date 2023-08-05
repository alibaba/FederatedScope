package com.example.fsandroid.utils;

import android.content.Context;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class FilesUtil {
    public static String storagePath;

    private FilesUtil() {}

    public static void initFileManager(Context context) {
        if (storagePath == null) {
            storagePath = context.getFilesDir().getPath();
        }
    }

    public static boolean fileExists(String path) {
        File file = new File(path);
        return fileExists(file);
    }

    public static boolean fileExists(Path path) {
        return fileExists(path.toString());
    }

    public static boolean fileExists(File file) {
        return file.exists();
    }

    public static void deleteFileIfExists(String path) {
        File file = new File(path);
        if (fileExists(file) && file.isFile()) {
            file.delete();
        }
    }

    public static void copyFile(String pFilePath, Path pTargetPath) {
        copyFile(Paths.get(pFilePath), pTargetPath);
    }

    public static void copyFile(Path pFilePath, Path pTargetPath) {
        if (fileExists(pFilePath)) {
            try {
                Files.copy(pFilePath, pTargetPath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            Log.d("File " + pFilePath + " doesn't exist");
        }
    }
}
