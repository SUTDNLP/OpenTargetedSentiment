
for i in `seq 1 10`;
do
	echo "MultiDiscreteCRFMMLabeler $i "		
	./MultiDiscreteCRFMMLabeler -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.joint.sparse
	echo "MultiDenseCRFMMLabeler $i"
 	./MultiDenseCRFMMLabeler -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.joint.dense -word emb/word2vec.tin.emb 
	echo "MultiDcombCRFMMLabeler $i"	     
	./MultiDcombCRFMMLabeler -l -train joint/train$i.nn -dev joint/dev$i.nn -test joint/test$i.nn -model test.model -option option.joint.combine -word emb/word2vec.tin.emb 
done
