package com.example.fsandroid.utils;

import com.example.fsandroid.configs.Config;

import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class YamlUtil {
    private final static String fileName4Config = "config.yaml";

    private volatile static Config instance = null;

    private YamlUtil() {}

    public static Config getConfig() {
        if (instance == null) {
            synchronized (Config.class) {
                if (instance == null) {
                    Path filePath = Paths.get(FilesUtil.storagePath, fileName4Config);
                    if (FilesUtil.fileExists(filePath)) {
                        try {
                            Yaml yaml = new Yaml(new Constructor(Config.class));
                            InputStream inputStream = Files.newInputStream(filePath);
                            instance = yaml.load(inputStream);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    } else {
                        // user default yaml
                        instance = new Config();
                    }
                }
            }
        }
        return instance;
    }
}
