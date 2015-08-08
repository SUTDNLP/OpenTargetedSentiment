
for i in `seq 1 10`;
do
	echo "LabelerBasic $i "		
	./LabelerBasic -l -train span/train$i.span -dev span/dev$i.span -test span/test$i.span -model test.model -option option.pb.crf  -word emb/word2vec.tin.emb	

	echo "LabelerNewBasic $i"
 	./LabelerNewBasic -l -train span/train$i.span -dev span/dev$i.span -test span/test$i.span -model test.model -option option.pnb.crf -word emb/word2vec.tin.emb 

	echo "LabelerTanh $i"	     
	./LabelerTanh -l -train span/train$i.span -dev span/dev$i.span -test span/test$i.span -model test.model -option option.plt.crf -word emb/word2vec.tin.emb 
			
	echo "LabelerNewTanh $i"
	./LabelerNewTanh -l -train span/train$i.span -dev span/dev$i.span -test span/test$i.span -model test.model -option option.pnlt.crf -word emb/word2vec.tin.emb 
		
	echo "LabelerNNTanh $i"
	./LabelerNNTanh -l -train span/train$i.span -dev span/dev$i.span -test span/test$i.span -model test.model -option option.pt.crf -word emb/word2vec.tin.emb
	
done
