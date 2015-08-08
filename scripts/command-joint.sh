
for i in `seq 1 10`;
do
	echo "MultiLabelerBasic $i "		
	./MultiLabelerBasic -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.jb.crf  -word emb/word2vec.tin.emb
	echo "MultiLabelerNewBasic $i"
 	./MultiLabelerNewBasic -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.jnb.crf -word emb/word2vec.tin.emb 
	echo "MultiLabelerTanh $i"	     
	./MultiLabelerTanh -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.jlt.crf -word emb/word2vec.tin.emb 
	echo "MultiLabelerNewTanh $i"
	./MultiLabelerNewTanh -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.jnlt.crf -word emb/word2vec.tin.emb 
	echo "MultiLabelerNNTanh $i"
	./MultiLabelerNNTanh -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.jt.crf -word emb/word2vec.tin.emb 
done
