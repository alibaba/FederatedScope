package com.example.fsandroid;

import com.example.fsandroid.enums.UiType;

public class UiMsgObject {
    public UiType type;
    public Object obj;
    public Object obj2;

    public UiMsgObject(UiType type, Object obj) {
        this.type = type;
        this.obj = obj;
    }

    public UiMsgObject(UiType type, Object obj, Object obj2) {
        this.type = type;
        this.obj = obj;
        this.obj2 = obj2;
    }
}
