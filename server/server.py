from flask import Flask, request, redirect, url_for, flash, jsonify
# import numpy as np
# import pickle as p
import json
import itertools

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer
from allennlp.data.instance import Instance

# EN_EMBEDDING_DIM = 256
# ZH_EMBEDDING_DIM = 256
# HIDDEN_DIM = 256
EN_EMBEDDING_DIM = 512
ZH_EMBEDDING_DIM = 512
HIDDEN_DIM = 512
CUDA_DEVICE = -1


app = Flask(__name__)


@app.route('/api', methods=['POST'])
def makecalc():

    data = request.json
    ask=data['ask']
    predictor = SimpleSeq2SeqPredictor(model, reader)
    ask = reader.text_to_instance("", ask)

    answer=''.join(predictor.predict_instance(ask)['predicted_tokens'])

    return jsonify(answer=answer)

if __name__ == '__main__':

    reader = Seq2SeqDatasetReader(
    source_tokenizer=WordTokenizer(),
    target_tokenizer=CharacterTokenizer(),
    source_token_indexers={'tokens': SingleIdTokenIndexer()},
    target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    # train_dataset = reader.read('data/mt/tatoeba.eng_fin.train.tsv')
    # validation_dataset = reader.read('data/mt/tatoeba.eng_fin.dev.tsv')

    # vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
    #                                   min_count={'tokens': 3, 'target_tokens': 3})
    vocab = Vocabulary.from_files("vocabulary")
    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    # encoder = PytorchSeq2SeqWrapper(
    #     torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                                          projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    max_decoding_steps = 30   # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8,
                          use_bleu=True)

    with open("finbotallen/best.th", 'rb') as f:
        model.load_state_dict(torch.load(
            f, map_location=lambda storage, loc: storage))

    # modelfile = 'model.th'
    # model = p.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')