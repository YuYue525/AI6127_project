# AI6127_project

## Siamese Recurrent Neural Networks

Please train the corresponding models with commands:

```
# train Siamese model with GRU unit
python siamese_rnn/gru.py
```
```
# train Siamese model with bidirectional GRU unit
python siamese_rnn/bigru.py
```
```
# train Siamese model with LSTM unit
python siamese_rnn/lstm.py
```
```
# train Siamese model with bidirectional LSTM unit
python siamese_rnn/bilstm.py
```
```
# train Siamese model with Elman RNN unit
python siamese_rnn/rnn.py
```
```
# train Siamese model with bidirectional Elman RNN unit
python siamese_rnn/birnn.py
```

And then test the different model with commands:

```
python siamese_rnn/test.py --model_type {} --model_path
