package com.example.fsandroid;

import com.example.fsandroid.enums.CompressionType;
import com.example.fsandroid.protos.FileRequest;
import com.example.fsandroid.protos.MessageRequest;
import com.example.fsandroid.protos.MessageResponse;
import com.example.fsandroid.protos.MsgValue;
import com.example.fsandroid.protos.gRPCComServeFuncGrpc;
import com.example.fsandroid.utils.FilesUtil;
import com.example.fsandroid.utils.Log;
import com.google.protobuf.Empty;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.concurrent.TimeUnit;

import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;

/***
 * Transfer message/file from server to this android device
 */
public class S2DCommManager {
    private Server server;
    private final ClientService callback;
    private final String mnnPath2Train = Paths.get(FilesUtil.storagePath, "model2train.mnn").toString();
    private final String mnnPath2Test = Paths.get(FilesUtil.storagePath, "model2test.mnn").toString();


    public S2DCommManager(int port,  ClientService callback, String compression_method){
        this.callback = callback;
        try {
            if (compression_method.equals(CompressionType.GZIP) || compression_method.equals(CompressionType.DEFLATE)) {
                Log.d("Server started, listening on " + port + " with " + compression_method + " compression");
                server = NettyServerBuilder.forPort(port)
                        .intercept(new ServerInterceptor() {
                            @Override
                            public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
                                call.setCompression(compression_method);
                                return next.startCall(call, headers);
                            }
                        })
                        .addService(new gRPCComServeFuncImpl())
                        .maxInboundMessageSize(1024 * 1024 * 100)
                        .build()
                        .start();
            } else {
                Log.d("Server started, listening on " + port + " without compression");
                server = NettyServerBuilder.forPort(port)
                        .addService(new gRPCComServeFuncImpl())
                        .maxInboundMessageSize(1024 * 1024 * 100)
                        .build()
                        .start();
            }
        } catch (IOException e) {
            e.printStackTrace();
            Log.d(e.getMessage());
        }

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                try {
                    S2DCommManager.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
            }
        });


    }

    private void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }

    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    public void blockUntilShutdown(){
        if (server != null) {
            try {
                server.awaitTermination();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    class gRPCComServeFuncImpl extends gRPCComServeFuncGrpc.gRPCComServeFuncImplBase {
        @Override
        public void sendMessage(MessageRequest request, StreamObserver<MessageResponse> responseObserver) {
            // Parse the request
            String msg_type = request.getMsgOrThrow("msg_type").getSingleMsg().getStrValue();
            int state = request.getMsgOrThrow("state").getSingleMsg().getIntValue();
            int timeStamp = request.getMsgOrThrow("timestamp").getSingleMsg().getIntValue();
            int sender = request.getMsgOrThrow("sender").getSingleMsg().getIntValue();
            int receiver = request.getMsgOrThrow("receiver").getSingleMsg().getIntValue();
            // Parse according to msg type
            MsgValue content = request.getMsgOrThrow("content");

            Log.d("Receive message " + msg_type + "!");

            // Feedback
            MessageResponse msgResponse = MessageResponse.newBuilder().setMsg("ACK").build();
            responseObserver.onNext(msgResponse);
            responseObserver.onCompleted();

            // Dispatch to different services
            switch (msg_type) {
                case "assign_executor_id":
                    int assignId = Integer.valueOf(content.getSingleMsg().getStrValue());
                    callback.assignExecutorId(assignId);
                    break;
                case "finish":
                    // TODO:
                    callback.finish();
                    break;
            }
        }

        @Override
        public void sendMnnModel4Train(FileRequest request, StreamObserver<Empty> responseObserver) {
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
            try {
                // Delete the file if it exists
                FilesUtil.deleteFileIfExists(mnnPath2Train);
                // Write file
                OutputStream writer = Files.newOutputStream(Paths.get(mnnPath2Train), StandardOpenOption.CREATE);
                writer.write(request.getChunk().toByteArray());
                writer.close();
                Log.d("Finish receiving training model in " + mnnPath2Train + " for " + request.getInfo().getClientIdCount() + " clients!");
                int targetClientId = request.getInfo().getClientId(0);
                // Load model and start to train
                callback.localTrain(request.getInfo().getState(), mnnPath2Train, targetClientId);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void sendMnnModel4Test(FileRequest request, StreamObserver<Empty> responseObserver) {
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
            try {
                // Delete the file if it exists
                FilesUtil.deleteFileIfExists(mnnPath2Test);
                // Write file
                OutputStream writer = Files.newOutputStream(Paths.get(mnnPath2Test), StandardOpenOption.CREATE);
                writer.write(request.getChunk().toByteArray());
                writer.close();
                Log.d("Finish receiving model in " + mnnPath2Test + "!");
                List<Integer> clientIdList = request.getInfo().getClientIdList();
                callback.localEvaluate(request.getInfo().getState(), mnnPath2Test, clientIdList);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
