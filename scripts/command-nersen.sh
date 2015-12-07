
for i in `seq 1 10`;
do
    # get the training file for the second step of the pipeline model
    cp joint/train$i.nn nersen/train$i.ner.sen

	echo "DiscreteCRFMMLabeler $i "
	# replace the ner results by the predicted results from the first step of the pipeline model for the development and test corpus
	# java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipeline.sparse nersen/test$i.pipeline.sparse.ner.sen
    # java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipeline.sparse nersen/dev$i.pipeline.sparse.ner.sen
	./DiscreteCRFMMLabeler -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipeline.sparse.ner.sen -test nersen/test$i.pipeline.sparse.ner.sen -model test.model -option option.pipesen.sparse
	
	echo "DenseCRFMMLabeler  $i"
	# replace the ner results by the predicted results from the first step of the pipeline model for the development and test corpus
 	# java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipeline.dense nersen/test$i.pipeline.dense.ner.sen
    # java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipeline.dense nersen/dev$i.pipeline.dense.ner.sen
 	./DenseCRFMMLabeler  -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipeline.dense.ner.sen -test nersen/test$i.pipeline.dense.ner.sen -model test.model -option option.pipesen.dense  -word emb/word2vec.tin.emb
 	
	echo "DcombCRFMMLabeler  $i"	
    # replace the ner results by the predicted results from the first step of the pipeline model for the development and test corpus	
	# java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipeline.combine nersen/test$i.pipeline.combine.ner.sen
    # java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipeline.combine nersen/dev$i.pipeline.combine.ner.sen
	./DcombCRFMMLabeler  -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipeline.combine.ner.sen -test nersen/test$i.pipeline.combine.ner.sen -model test.model -option option.pipesen.combine  -word emb/word2vec.tin.emb
	
done
