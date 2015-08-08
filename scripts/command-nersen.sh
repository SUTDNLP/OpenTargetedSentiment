
for i in `seq 1 10`;
do
        echo "folder $i"
        cp joint/train$i.nn nersen/train$i.ner.sen

	echo "LabelerBasic $i "
	java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipe.basic nersen/test$i.pipe.basic.ner.sen
        java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipe.basic nersen/dev$i.pipe.basic.ner.sen
	./LabelerBasic -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipe.basic.ner.sen -test nersen/test$i.pipe.basic.ner.sen -model test.model -option option.pb.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNewBasic $i"
 	java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipe.newbasic nersen/test$i.pipe.newbasic.ner.sen
        java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipe.newbasic nersen/dev$i.pipe.newbasic.ner.sen
 	./LabelerNewBasic -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipe.newbasic.ner.sen -test nersen/test$i.pipe.newbasic.ner.sen -model test.model -option option.pnb.crf  -word emb/word2vec.tin.emb
 	
	echo "LabelerTanh $i"	      
	java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipe.ltanh nersen/test$i.pipe.ltanh.ner.sen
        java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipe.ltanh nersen/dev$i.pipe.ltanh.ner.sen
	./LabelerTanh -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipe.ltanh.ner.sen -test nersen/test$i.pipe.ltanh.ner.sen -model test.model -option option.plt.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNewTanh $i" 
	java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipe.newltanh nersen/test$i.pipe.newltanh.ner.sen
        java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipe.newltanh nersen/dev$i.pipe.newltanh.ner.sen
	./LabelerNewTanh -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipe.newltanh.ner.sen -test nersen/test$i.pipe.newltanh.ner.sen -model test.model -option option.pnlt.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNNTanh $i"
	java -Xmx1G -jar AddFirstOutput.jar joint/test$i.nn ner/test$i.ner.pipe.tanh nersen/test$i.pipe.tanh.ner.sen
        java -Xmx1G -jar AddFirstOutput.jar joint/dev$i.nn ner/dev$i.ner.pipe.tanh nersen/dev$i.pipe.tanh.ner.sen
	./LabelerNNTanh -l -train nersen/train$i.ner.sen -dev nersen/dev$i.pipe.tanh.ner.sen -test nersen/test$i.pipe.tanh.ner.sen -model test.model -option option.pt.crf  -word emb/word2vec.tin.emb
	
done
