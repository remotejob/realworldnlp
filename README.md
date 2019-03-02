# Real-World Natural Language Processing

This repository contains example code for the book "Real-World Natural Language Processing."

AllenNLP (0.7.2 or above)is required to run the example code in this repository.


conda create -n allennlp python=3.7.2
conda activate allennlp


cp /tmp/seq2seqalien.txt data/mt/tatoeba.eng_fin.tsv

cat data/mt/tatoeba.eng_fin.tsv | awk 'NR%10==1' > tatoeba.eng_fin.test.tsv
cat data/mt/tatoeba.eng_fin.tsv | awk 'NR%10==2' > data/mt/tatoeba.eng_fin.dev.tsv
cat data/mt/tatoeba.eng_fin.tsv | awk 'NR%10!=1&&NR%10!=2' > data/mt/tatoeba.eng_fin.train.tsv
 