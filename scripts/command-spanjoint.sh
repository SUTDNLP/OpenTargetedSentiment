
for i in `seq 1 10`;
do
	echo "MultiLabelerBasic $i "		
	./MultiLabelerBasic -l -train spanjoint/train$i.spannn -dev spanjoint/dev$i.spannn -test spanjoint/test$i.spannn -model test.model -option option.jb.crf  -word emb/word2vec.tin.emb
	echo "MultiLabelerNewBasic $i"
 	./MultiLabelerNewBasic -l -train spanjoint/train$i.spannn -dev spanjoint/dev$i.spannn -test spanjoint/test$i.spannn -model test.model -option option.jnb.crf -word emb/word2vec.tin.emb 
	echo "MultiLabelerTanh $i"	     
	./MultiLabelerTanh -l -train spanjoint/train$i.spannn -dev spanjoint/dev$i.spannn -test spanjoint/test$i.spannn -model test.model -option option.jlt.crf -word emb/word2vec.tin.emb 
	echo "MultiLabelerNewTanh $i"
	./MultiLabelerNewTanh -l -train spanjoint/train$i.spannn -dev spanjoint/dev$i.spannn -test spanjoint/test$i.spannn -model test.model -option option.jnlt.crf -word emb/word2vec.tin.emb 
	echo "MultiLabelerNNTanh $i"
	./MultiLabelerNNTanh -l -train spanjoint/train$i.spannn -dev spanjoint/dev$i.spannn -test spanjoint/test$i.spannn -model test.model -option option.jt.crf -word emb/word2vec.tin.emb 
done
