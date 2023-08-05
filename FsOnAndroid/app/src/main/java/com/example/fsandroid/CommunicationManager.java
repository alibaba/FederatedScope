package com.example.fsandroid;

import android.content.Context;

import com.example.fsandroid.enums.CompressionType;
import com.example.fsandroid.protos.FileInfo;
import com.example.fsandroid.protos.FileRequest;
import com.example.fsandroid.protos.MessageRequest;
import com.example.fsandroid.protos.MessageResponse;
import com.example.fsandroid.protos.MsgValue;
import com.example.fsandroid.protos.gRPCComServeFuncGrpc;
import com.example.fsandroid.protos.mDict_keyIsString;
import com.example.fsandroid.protos.mSingle;
import com.example.fsandroid.utils.DeviceUtil;
import com.example.fsandroid.utils.Log;
import com.example.fsandroid.utils.TimeUtil;
import com.google.protobuf.ByteString;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import io.grpc.ManagedChannel;
import io.grpc.android.AndroidChannelBuilder;
import io.grpc.stub.StreamObserver;
import kotlin.TypeCastException;

public class CommunicationManager {
    private gRPCComServeFuncGrpc.gRPCComServeFuncStub asyncStub;
    private gRPCComServeFuncGrpc.gRPCComServeFuncBlockingStub blockingStub;
    private Context mContext;
    private String mHost;
    private String mCompression;

    public CommunicationManager(Context context, String host, String port, String compression) {

        ManagedChannel mchannel = AndroidChannelBuilder.forAddress(host, Integer.valueOf(port))
                .context(context)
                .usePlaintext()
                .build();

        // TODO: since the trained model could be huge, maybe use newAsyncStub rather than blocking stub
        asyncStub = gRPCComServeFuncGrpc.newStub(mchannel);
        blockingStub = gRPCComServeFuncGrpc.newBlockingStub(mchannel);
        mContext = context;
        mHost = host;
        mCompression = compression;

        // Assert compression method
        if (compression.equals(CompressionType.DEFLATE) || compression.equals(CompressionType.GZIP)) {
            Log.d("Activate compression method " + compression);
        } else {
            Log.d("No compression method is found with " + compression);
        }
    }

    public boolean joinIN(int clientId, String report_host, int report_port, Map<String, Object> deviceInfo) {
        Map content = new HashMap<String, Object>();
        content.put("host", report_host); // 手机自己的ip，虚拟机的话是0.0.0.0，加上转发；手机的话就是获取ip
        content.put("port", Integer.toString(report_port)); // 手机自己的port，和什么端口无关

        // device information
        content.putAll(deviceInfo);

        return sendMessage("join_in", clientId, 0, 0, packageMsgValue(content));
    }

    public boolean uploadMetrics(int clientId, int state, Map<String, Object> metrics) {
        return sendMessage("metrics", clientId, 0, state, packageMsgValue(metrics));
    }

    public boolean uploadMnnModel(int clientId, int state, int nSample, String mpath) throws IOException {
        // input file
        File file = new File(mpath);
        long fileSize = file.length();
        FileInputStream fileInputStream = new FileInputStream(file);
        byte[] buffer = new byte[(int) fileSize];

        int offset = 0;
        int numRead = 0;
        while (offset < buffer.length
                && (numRead = fileInputStream.read(buffer, offset, buffer.length - offset)) >= 0) {
            offset += numRead;
        }

        // 确保所有数据均被读取
        if (offset != buffer.length) {
            throw new IOException("Could not completely read file "
                    + file.getName());
        }
        fileInputStream.close();

        // Fill info
        FileRequest fileRequest = FileRequest.newBuilder()
                .setInfo(FileInfo.newBuilder()
                        .setSender(clientId)
                        .setNSample(nSample)
                        .setState(state)
                        .setTimestamp(TimeUtil.getTimeStamp()))
                .setChunk(ByteString.copyFrom(buffer))
                .build();
        try {
            if (mCompression.equals(CompressionType.DEFLATE) || mCompression.equals(CompressionType.GZIP)) {
                blockingStub.withCompression(mCompression).uploadMnnModel(fileRequest);
            } else {
                blockingStub.uploadMnnModel(fileRequest);
            }
            return true;
        }catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    private boolean sendMessage(String pmsg_type, int psender, int preceiver, int pstate, MsgValue pcontent) {
        MsgValue msg_type = packageMsgValue(pmsg_type);
        MsgValue sender = packageMsgValue(psender);
        MsgValue receiver = packageMsgValue(preceiver);
        MsgValue state = packageMsgValue(pstate);

        String ptimestamp = TimeUtil.getTimeStamp();
        MsgValue timestamp = packageMsgValue(ptimestamp);

        MessageRequest msgRequest = MessageRequest.newBuilder()
                .putMsg("msg_type", msg_type)
                .putMsg("sender", sender)
                .putMsg("receiver", receiver)
                .putMsg("state", state)
                .putMsg("content", pcontent)
                .putMsg("timestamp", timestamp)
                .build();

        // Try to send, if failed, show Toast
        try {
            MessageResponse response;
            if (mCompression.equals(CompressionType.DEFLATE) || mCompression.equals(CompressionType.GZIP)) {
                response = blockingStub.withCompression(mCompression).sendMessage(msgRequest);
            } else {
                response = blockingStub.sendMessage(msgRequest);
            }

            if (response.getMsg().equals("ACK")) {
                return true;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    private <T> MsgValue packageMsgValue(T msgValue) {
        Class msgCls = msgValue.getClass();
        if (msgCls.equals(String.class) || msgCls.equals(Integer.class) || msgCls.equals(Float.class) || msgCls.equals(float.class) || msgCls.equals(int.class)) {
            mSingle singleValue = packageMSingle(msgValue);
            return MsgValue.newBuilder().setSingleMsg(singleValue).build();
        } else if (msgCls.equals(HashMap.class) || msgCls.equals(Map.class)) {
            mDict_keyIsString.Builder dictBuilder = mDict_keyIsString.newBuilder();
            for (Map.Entry<String, Object> entry: ((HashMap<String, Object>) msgValue).entrySet()){
                dictBuilder.putDictValue(entry.getKey(), packageMsgValue(entry.getValue()));
            }
            return MsgValue.newBuilder().setDictMsgStringkey(dictBuilder).build();
        } else {
            throw new TypeCastException();
        }
    }

    private <T> mSingle packageMSingle(T value) {
        Object valueClass = value.getClass();
        if (valueClass.equals(String.class)) {
            return mSingle.newBuilder().setStrValue((String) value).build();
        } else if (valueClass.equals(Integer.class) || valueClass.equals(int.class)) {
            return mSingle.newBuilder().setIntValue((int) value).build();
        } else if (valueClass.equals(float.class) || valueClass.equals(Float.class)) {
            return mSingle.newBuilder().setFloatValue((float) value).build();
        }
        return null;
    }
}
