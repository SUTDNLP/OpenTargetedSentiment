
for i in `seq 1 10`;
do
	echo "SparseCRFMMLabeler  $i "		
	./SparseCRFMMLabeler  -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.pipener.sparse	

	echo "TNNCRFMMLabeler  $i"
 	./TNNCRFMMLabeler  -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.pipener.dense -word emb/word2vec.tin.emb 

	echo "SparseTNNCRFMMLabeler $i"	     
	./SparseTNNCRFMMLabeler -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.pipener.combine -word emb/word2vec.tin.emb 
	
done
