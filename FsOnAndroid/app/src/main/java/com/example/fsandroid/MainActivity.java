package com.example.fsandroid;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.SimpleAdapter;
import android.widget.TextView;
import android.widget.Toast;

import com.example.fsandroid.configs.Config;
import com.example.fsandroid.databinding.ActivityMainBinding;
import com.example.fsandroid.enums.UiType;
import com.example.fsandroid.enums.EventType;
import com.example.fsandroid.utils.AppMonitor;
import com.example.fsandroid.utils.AssetsUtil;
import com.example.fsandroid.utils.DeviceUtil;
import com.example.fsandroid.utils.FilesUtil;
import com.example.fsandroid.utils.Log;
import com.example.fsandroid.utils.TimeUtil;
import com.example.fsandroid.utils.ToastUtil;
import com.example.fsandroid.utils.YamlUtil;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class MainActivity extends AppCompatActivity {

    // Used to load the 'fsmnn' library on application startup.
    static {
        System.loadLibrary("fsmnn");
    }

    // Load custom config and merge with the default values
    private Config mConfig;

    private int executorId = -1;

    private ActivityMainBinding binding;
    private Handler uiHandler;

    // gRPC communication
    private S2DCommManager s2DCommManager;
    private CommunicationManager commManager;

    // Config panel
    private Button startBtn;

    // Fed log panel
    private ListView fedLogList;
    private List<HashMap<String, String>> logItems;
    private SimpleAdapter logAdapter;

    // Training panel
    private ProgressBar trainProgressBar;
    private TextView epochText;
    private TextView trainLog;
    private MyLineChartView trainLineChart;

    // Test panel
    private TextView testLog;

    // Action bar
    private ActionBar actionBar;

    private boolean isLogined = false;

    // Mnn thread
    private MnnThread mnnThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        actionBar = getSupportActionBar();

        // Init path for storage
        FilesUtil.initFileManager(getApplicationContext());

        mConfig = YamlUtil.getConfig();

        startBtn = binding.startButton;
        fedLogList = binding.fedLog;
        epochText = binding.epochId;
        trainProgressBar = binding.trainProgressBar;
        trainLog = binding.lossLog;
        testLog = binding.accuracyLog;
        trainLineChart = binding.lineChart;

        trainProgressBar.setMax(100);

        startBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String host = mConfig.distribute.server_host;
                String port = Integer.toString(mConfig.distribute.server_port);
                // Check host and port
                if (checkHostAndPort(host, port)) {
                    commManager = new CommunicationManager(MainActivity.this, host, port, mConfig.distribute.grpc_compression);
                    new Thread(new LoginRunnable()).start();
                } else {
                    // TODO: Report error
                    ToastUtil.showToast(MainActivity.this, "Please check the input ip and port!", Toast.LENGTH_SHORT);
                }
            }
        });

        // Fed logs
        logItems = new ArrayList<>();
        String[] from = new String[]{"timePrefix", "logContent"};
        int[] to = new int[]{R.id.time_prefix, R.id.log_content};
        logAdapter = new SimpleAdapter(MainActivity.this, logItems, R.layout.row_item_log, from, to);
        fedLogList.setAdapter(logAdapter);

        uiHandler = new Handler() {
            public void handleMessage(Message msg) {
                UiMsgObject uiMsgObject = (UiMsgObject) msg.obj;
                switch (uiMsgObject.type) {
                    case FED_LOG:
                        // Fill a new item
                        HashMap<String, String> logItem = new HashMap<>();
                        SimpleDateFormat timesStamp = new SimpleDateFormat("yyyy/MM/dd HH:mm - ");
                        logItem.put("timePrefix", timesStamp.format(new Date()));
                        logItem.put("logContent", (String) uiMsgObject.obj);
                        logItems.add(logItem);
                        // update fed log
                        logAdapter.notifyDataSetChanged();
                        // sync to logcat
                        Log.d((String) uiMsgObject.obj);
                        break;
                    case TRAIN_LOSS:
                        Map trainInfo = (HashMap<String, Object>)uiMsgObject.obj;
                        epochText.setText("Epoch " + (int) trainInfo.get("epoch") + "/" + mConfig.train.local_update_steps);
                        trainLog.setText("Loss: " + (float) trainInfo.get("loss"));
                        break;
                    case TEST_ACCU:
                        Map testInfo = (HashMap<String, Object>)uiMsgObject.obj;
                        testLog.setText("Accuracy: " + (float) testInfo.get("accuracy") * 100f + "%");
                        break;
                    case TRAIN_PROGRESS:
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                            trainProgressBar.setProgress((int)uiMsgObject.obj, true);
                        } else {
                            trainProgressBar.setProgress((int)uiMsgObject.obj);
                        }
                        break;
                    case TRAIN_LINE_CHART:
                        // TODO: add actions here
                        trainLineChart.addData((float)uiMsgObject.obj, (float)uiMsgObject.obj2);
                        break;
                }

            }
        };

        // gRPC communication from android device to the server
        s2DCommManager = new S2DCommManager(mConfig.distribute.device_port, new ClientService() {

            @Override
            public void assignExecutorId(int id) {
                login();
                executorId = id;
                uptUi(UiType.FED_LOG, "Receive executor ID as " + id);
//                startBtn.setEnabled(false);
            }

            @Override
            public void localTrain(int pState, String pPath, int pClientId) {
                // update global state
                Log.d("Put train event into handler");
                Message msg = new Message();
                msg.what = EventType.TRAIN_EVENT;
                msg.obj = new MnnTrainEvent(pPath, pState, pClientId);
                mnnThread.mHandler.sendMessage(msg);
                Log.d("Finish putting train handler");
            }

            @Override
            public void localEvaluate(int pState, String pPath, List<Integer> pClientIdList) {
                Log.d("Put test event into handler");
                Message msg = new Message();
                msg.what = EventType.TEST_EVENT;
                msg.obj = new MnnTestEvent(pPath, pState, pClientIdList);
                mnnThread.mHandler.sendMessage(msg);
                Log.d("Finish putting test event");
            }

            @Override
            public void finish() {
                // TODO: record time and report to the remote server

            }
        }, mConfig.distribute.grpc_compression);

        // Start threads
        // TODO: Thread management
        new Thread(new GrpcRunnable()).start();

        mnnThread = new MnnThread();
        mnnThread.start();

        Log.d("Finish init the MainActivity!");

        // Start training automatically if config says yes
        if (mConfig.train.auto_start) {
            Log.d("Start to train with TRAIN.AUTO_TRAIN=TRUE");
            ToastUtil.showToast(getApplicationContext(), "Start to train with TRAIN.AUTO_TRAIN=TRUE", Toast.LENGTH_SHORT);
            startBtn.performClick();
        } else {
            AppMonitor.getInstance().init(getApplicationContext(), 1000L, uiHandler);
            AppMonitor.getInstance().start();
        }
    }

    private synchronized void login() {
        isLogined = true;
        Log.d("Login is successful!");
    }

    /**
     * A native method that is implemented by the 'fsmnn' native library,
     * which is packaged with this application.
     */
    public native int generalTrainModelInMNN(String pPathModel, Config pConfig, String pPathData);

    public native int fedBabuTrainModelInMNN(String pPathWholeModel, String pPathBodyModel, Config pConfig, String pPathData);

    public native HashMap<String, Object> generalTestModelInMNN(String pPathModel, Config pConfig, String pPathData);
    
    public native HashMap<String, Object> fedBabuTestModelInMNN(String pPathWholeModel, String pPathBodyModel, Config pConfig, String pPathData);
    
    public void updateTrainInfoFromCpp(int epoch, float loss) {
        Message msg = Message.obtain();
        Map obj = new HashMap<String, Object>();
        obj.put("epoch", epoch);
        obj.put("loss", loss);
        msg.obj = new UiMsgObject(UiType.TRAIN_LOSS, obj);
        uiHandler.sendMessage(msg);
    }

    public void updateTestInfoFromCpp(int epoch, float accuracy) {
        Message msg = Message.obtain();
        Map obj = new HashMap<String, Object>();
        obj.put("epoch", epoch);
        obj.put("accuracy", accuracy);
        msg.obj = new UiMsgObject(UiType.TEST_ACCU, obj);
        uiHandler.sendMessage(msg);
    }

    private void uptProgressBar(int progress) {
        uptUi(UiType.TRAIN_PROGRESS, progress);
    }

    private boolean checkHostAndPort(String host, String port) {
        // TODO
        return true;
    }

    private <T> void uptUi(UiType type, T obj) {
        Message msg = Message.obtain();
        msg.obj = new UiMsgObject(type, obj);
        uiHandler.sendMessage(msg);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_activity_main, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()){
            case R.id.action_bar_setting:
                Intent intent = new Intent();
                intent.setClass(MainActivity.this, SettingActivity.class);
                startActivity(intent);
                overridePendingTransition(R.anim.activity_in_from_right, R.anim.activity_out_static);
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onDestroy() {
        // Stop all threads
        Log.d("onDestroy: stop all sub-threads.");
        mnnThread.interrupt();

        super.onDestroy();
    }

    private class LoginRunnable implements Runnable {
        private int mRepeatTimes = 1;
        @Override
        public void run() {
            if (mConfig.train.auto_start) {
                // Transfer all data into files
                uptUi(UiType.FED_LOG, "Begin to unzip " + mConfig.data.type+ ".zip");
                AssetsUtil.copyAssetsDir2Phone(getApplicationContext(), mConfig.data.type);
                uptUi(UiType.FED_LOG, "Finish!");
            }

            uptUi(UiType.FED_LOG, "Connect to Server " + mConfig.distribute.server_host + ":" + mConfig.distribute.server_port + "...");

            try {
                 Map<String, Object> deviceInfo = DeviceUtil.getDeviceInfo(getApplicationContext(), mConfig.distribute.report_host);
                while (!isLogined && mRepeatTimes <= 10) {
                    Log.d("Login for "+ mRepeatTimes + "-th time" );
                    commManager.joinIN(executorId, mConfig.distribute.report_host, mConfig.distribute.report_port, deviceInfo);
                    Thread.sleep(500);
                    mRepeatTimes++;
                }
                uptUi(UiType.FED_LOG, "Login is succeeded!");
//                startBtn.setEnabled(false);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private class GrpcRunnable implements Runnable {
        @Override
        public void run() {
            s2DCommManager.blockUntilShutdown();
        }
    }

    private class MnnThread extends Thread {
        private Handler mHandler = null;

        @Override
        public void run() {
            Looper.prepare();
            mHandler = new Handler() {
                @Override
                public void handleMessage(@NonNull Message msg) {
                    String pathClientData;
                    switch (msg.what) {
                        case EventType.TRAIN_EVENT:
                            Log.d("Start train event in MnnThread");
                            MnnTrainEvent trainEvent = (MnnTrainEvent) msg.obj;
                            uptUi(UiType.FED_LOG, "Start local training for CLIENT #" + trainEvent.mClientId + " in ROUND #" + trainEvent.mState);
                            // Update current state
                            Log.d("Begin to train model in " + trainEvent.mModelPath);
                            pathClientData = mConfig.data.root + "/" + mConfig.data.type + "/" + "task_" + Integer.toString(trainEvent.mClientId);

                            int n4Samples;
                            if (mConfig.train.fedbabu) {
                                Path fedBabuModelPath = Paths.get(FilesUtil.storagePath, "fedBabuModel.mnn");
                                if (trainEvent.mState == 0) {
                                    // Copy the model to specific location
                                    FilesUtil.copyFile(trainEvent.mModelPath, fedBabuModelPath);
                                    n4Samples = fedBabuTrainModelInMNN(trainEvent.mModelPath, "xxx", mConfig, pathClientData);
                                } else {
                                    n4Samples = fedBabuTrainModelInMNN(fedBabuModelPath.toString(), trainEvent.mModelPath, mConfig, pathClientData);
                                }
                            } else {
                                n4Samples = generalTrainModelInMNN(trainEvent.mModelPath, mConfig, pathClientData);
                            }

                            uptUi(UiType.FED_LOG, "Finish local training for ROUND #" + trainEvent.mState);
                            // upload to model
                            try {
                                uptUi(UiType.FED_LOG, "Upload model to server (" + mConfig.distribute.server_host + ": " + Integer.toString(mConfig.distribute.server_port) + ") in Round #" + trainEvent.mState);
                                commManager.uploadMnnModel(trainEvent.mClientId, trainEvent.mState, n4Samples,
                                        "/data/user/0/com.example.fsandroid/files/localTrainedModel.mnn");
                                uptUi(UiType.FED_LOG, "Finish upload in Round #" + trainEvent.mState);
                            } catch (IOException e) {
                                e.printStackTrace();
                                uptUi(UiType.FED_LOG, "Upload failed in Round #" + trainEvent.mState);
                            }
                            Log.d("End train event in MnnThread");
                            break;

                        case EventType.TEST_EVENT:
                            Log.d("Start test event in MnnThread");
                            MnnTestEvent testEvent = (MnnTestEvent) msg.obj;
                            for (int clientId: testEvent.mClientIdList) {
                                uptUi(UiType.FED_LOG, "Start local evaluation for Client #" + clientId + " in Round #" + testEvent.mState);
                                // Execute evaluation
                                pathClientData = mConfig.data.root + "/" + mConfig.data.type + "/task_" + Integer.toString(clientId);

                                Map<String, Object> metrics;
                                if (mConfig.train.fedbabu) {
                                    Path fedBabuModelPath = Paths.get(FilesUtil.storagePath, "fedBabuModel.mnn");
                                    metrics = fedBabuTestModelInMNN(fedBabuModelPath.toString(), testEvent.mModelPath, mConfig, pathClientData);
                                } else {
                                    metrics = generalTestModelInMNN(testEvent.mModelPath, mConfig, pathClientData);
                                }

                                uptUi(UiType.FED_LOG, "Finish local evaluation for Client #" + clientId + " in Round #" + testEvent.mState);
                                // Upload evaluation results
                                commManager.uploadMetrics(clientId, testEvent.mState, metrics);
                            }
                            Log.d("End test event in MnnThread");
                            break;

                        default:
                            Log.d("Unknow event type " + msg.what);
                    }

                }
            };
            Looper.loop();
        }
    }

    private class MnnTrainEvent{
        public String mModelPath;
        public int mState;
        public int mClientId;
        public MnnTrainEvent(String pModelPath, int pState, int pClientId ) {
            this.mModelPath = pModelPath;
            this.mState = pState;
            this.mClientId = pClientId;
        }
    }

    private class MnnTestEvent{
        public String mModelPath;
        public int mState;
        public List<Integer> mClientIdList;
        public MnnTestEvent(String pModelPath, int pState, List<Integer> pClientIdList) {
            this.mModelPath = pModelPath;
            this.mState = pState;
            this.mClientIdList = pClientIdList;
        }
    }
}
