# Real-World Natural Language Processing

This repository contains example code for the book "Real-World Natural Language Processing."

AllenNLP (0.7.2 or above)is required to run the example code in this repository.


conda create -n allennlp python=3.6
conda activate allennlp36


cp /tmp/seq2seqalien.txt data/mt/tatoeba.eng_fin.tsv

cat data/mt/tatoeba.eng_fin.tsv | awk 'NR%10==1' > tatoeba.eng_fin.test.tsv
cat data/mt/tatoeba.eng_fin.tsv | awk 'NR%10==2' > data/mt/tatoeba.eng_fin.dev.tsv
cat data/mt/tatoeba.eng_fin.tsv | awk 'NR%10!=1&&NR%10!=2' > data/mt/tatoeba.eng_fin.train.tsv


python examples/mt/mt.py

tar zcfv /tmp/allennlpmodel.tar.gz model.th vocabulary/

rm -f model.th vocabulary/


curl --header "Content-Type: application/json" --request POST --data '{"ask":"missa sinä asua"}' http://localhost:5000/api
curl --header "Content-Type: application/json" --request POST --data '{"ask":"milloin voida tavata"}' http://localhost:5000/api
curl --header "Content-Type: application/json" -H "Content-Type: application/x-www-form-urlencoded; charset=utf-8" --request POST --data '{"ask":"olla sinä iso tissi ja nänni"}' http://localhost:5000/api
curl -H "Content-Type: application/json; Content-Type: application/x-www-form-urlencoded; charset=utf-8" --request POST --data '{"ask":"se olla ihana"}' http://localhost:5000/api
curl -H "Content-Type: application/json; Content-Type: application/x-www-form-urlencoded; charset=utf-8" --request POST --data '{"ask":"minkalainen sinä olla"}' http://localhost:5000/api