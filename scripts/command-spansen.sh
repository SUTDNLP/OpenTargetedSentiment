
for i in `seq 1 10`;
do
        echo "folder $i"
        cp spanjoint/train$i.spannn spansen/train$i.span.sen

	echo "LabelerBasic $i "
	java -Xmx1G -jar AddFirstOutput.jar spanjoint/test$i.spannn span/test$i.span.pipe.basic spansen/test$i.pipe.basic.span.sen
        java -Xmx1G -jar AddFirstOutput.jar spanjoint/dev$i.spannn span/dev$i.span.pipe.basic spansen/dev$i.pipe.basic.span.sen
	./LabelerBasic -l -train spansen/train$i.span.sen -dev spansen/dev$i.pipe.basic.span.sen -test spansen/test$i.pipe.basic.span.sen -model test.model -option option.pb.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNewBasic $i"
 	java -Xmx1G -jar AddFirstOutput.jar spanjoint/test$i.spannn span/test$i.span.pipe.newbasic spansen/test$i.pipe.newbasic.span.sen
        java -Xmx1G -jar AddFirstOutput.jar spanjoint/dev$i.spannn span/dev$i.span.pipe.newbasic spansen/dev$i.pipe.newbasic.span.sen
 	./LabelerNewBasic -l -train spansen/train$i.span.sen -dev spansen/dev$i.pipe.newbasic.span.sen -test spansen/test$i.pipe.newbasic.span.sen -model test.model -option option.pnb.crf  -word emb/word2vec.tin.emb
 	
	echo "LabelerTanh $i"	      
	java -Xmx1G -jar AddFirstOutput.jar spanjoint/test$i.spannn span/test$i.span.pipe.ltanh spansen/test$i.pipe.ltanh.span.sen
        java -Xmx1G -jar AddFirstOutput.jar spanjoint/dev$i.spannn span/dev$i.span.pipe.ltanh spansen/dev$i.pipe.ltanh.span.sen
	./LabelerTanh -l -train spansen/train$i.span.sen -dev spansen/dev$i.pipe.ltanh.span.sen -test spansen/test$i.pipe.ltanh.span.sen -model test.model -option option.plt.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNewTanh $i" 
	java -Xmx1G -jar AddFirstOutput.jar spanjoint/test$i.spannn span/test$i.span.pipe.newltanh spansen/test$i.pipe.newltanh.span.sen
        java -Xmx1G -jar AddFirstOutput.jar spanjoint/dev$i.spannn span/dev$i.span.pipe.newltanh spansen/dev$i.pipe.newltanh.span.sen
	./LabelerNewTanh -l -train spansen/train$i.span.sen -dev spansen/dev$i.pipe.newltanh.span.sen -test spansen/test$i.pipe.newltanh.span.sen -model test.model -option option.pnlt.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNNTanh $i"
	java -Xmx1G -jar AddFirstOutput.jar spanjoint/test$i.spannn span/test$i.span.pipe.tanh spansen/test$i.pipe.tanh.span.sen
        java -Xmx1G -jar AddFirstOutput.jar spanjoint/dev$i.spannn span/dev$i.span.pipe.tanh spansen/dev$i.pipe.tanh.span.sen
	./LabelerNNTanh -l -train spansen/train$i.span.sen -dev spansen/dev$i.pipe.tanh.span.sen -test spansen/test$i.pipe.tanh.span.sen -model test.model -option option.pt.crf  -word emb/word2vec.tin.emb
	
done
