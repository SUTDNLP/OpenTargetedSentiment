
for i in `seq 1 10`;
do
	echo "LabelerBasic $i "		
	./LabelerBasic -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.pb.crf  -word emb/word2vec.tin.emb	

	echo "LabelerNewBasic $i"
 	./LabelerNewBasic -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.pnb.crf -word emb/word2vec.tin.emb 

	echo "LabelerTanh $i"	     
	./LabelerTanh -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.plt.crf -word emb/word2vec.tin.emb 
			
	echo "LabelerNewTanh $i"
	./LabelerNewTanh -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.pnlt.crf -word emb/word2vec.tin.emb 
		
	echo "LabelerNNTanh $i"
	./LabelerNNTanh -l -train ner/train$i.ner -dev ner/dev$i.ner -test ner/test$i.ner -model test.model -option option.pt.crf -word emb/word2vec.tin.emb
	
done
