# Message compression for efficient communication

We provide plugins of message compression for efficient communication.

## Lossless compression based on gRPC
When running with distributed mode of FederatedScope, the shared messages can be compressed using the compression module provided by gRPC (More details can be found [here](https://chromium.googlesource.com/external/github.com/grpc/grpc/+/HEAD/examples/python/compression/)).

Users can turn on the message compression by adding the following configuration:
```yaml
distribute:
  grpc_compression: 'deflate' # or 'gzip'
```

The compression of training ConvNet-2 on FEMNIST  is shown as below:

| | NoCompression | Deflate | Gzip |
| :---: | :---: | :---: | :---: |
| Communication bytes per round (in gRPC channel) | 4.021MB | 1.888MB | 1.890MB | 


## Model quantization
We provide a symmetric uniform quantization to transform the model parameters (32-bit float) to 8/16-bit int (note that it might bring model performance drop).

To apply model quantization, users need to add the following configurations:
```yaml
quantization:
    method: 'uniform'
    nbits: 16 # or 8
```

We conduct experiments based on the scripts provided in `federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml` and report the results as:

| | 32-bit float (vanilla) | 16-bit int | 8-bit int |
| :---: | :---: | :---: | :---: |
| Shared model size (in memory) | 25.20MB | 12.61MB | 6.31MB | 
| Model performance (acc) | 0.7856 | 0.7854 | 0.6807 | 

More fancy compression techniques are coming soon!  We greatly appreciate contribution to FederatedScope!
