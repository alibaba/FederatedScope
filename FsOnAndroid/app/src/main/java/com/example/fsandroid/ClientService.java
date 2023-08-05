package com.example.fsandroid;

import java.util.List;

public interface ClientService {
    void assignExecutorId(int id);
    void localTrain(int pState, String pPath, int pClientId);
    void localEvaluate(int pState, String pPath, List<Integer> clientIdList);
    void finish();
}
