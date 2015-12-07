
for i in `seq 1 10`;
do
	echo "SparseCRFMMLabeler  $i "		
	./SparseCRFMMLabeler  -l -train sen/train$i.sen -dev sen/dev$i.sen -test sen/test$i.sen -model test.model -option option.collapsed.sparse	

	echo "TNNCRFMMLabeler  $i"
 	./TNNCRFMMLabeler  -l -train sen/train$i.sen -dev sen/dev$i.sen -test sen/test$i.sen -model test.model -option option.collapsed.dense -word emb/word2vec.tin.emb 

	echo "SparseTNNCRFMMLabeler $i"	     
	./SparseTNNCRFMMLabeler -l -train sen/train$i.sen -dev sen/dev$i.sen -test sen/test$i.sen -model test.model -option option.collapsed.combine -word emb/word2vec.tin.emb 
	
done
