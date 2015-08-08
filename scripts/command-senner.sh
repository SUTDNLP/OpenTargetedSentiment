
for i in `seq 1 10`;
do
	echo "LabelerBasic $i "		
	java -Xmx1G -jar AddFirstOutFeats.jar temp/test$i.exchange.nn sen/test$i.sen.pipe.basic senner/test$i.pipe.basic.sen.ner
        java -Xmx1G -jar AddFirstOutFeats.jar temp/dev$i.exchange.nn sen/dev$i.sen.pipe.basic senner/dev$i.pipe.basic.sen.ner
	./LabelerBasic -l -train senner/train$i.sen.ner -dev senner/dev$i.pipe.basic.sen.ner -test senner/test$i.pipe.basic.sen.ner -model test.model -option option.pb.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNewBasic $i"
 	java -Xmx1G -jar AddFirstOutFeats.jar temp/test$i.exchange.nn sen/test$i.sen.pipe.newbasic senner/test$i.pipe.newbasic.sen.ner
        java -Xmx1G -jar AddFirstOutFeats.jar temp/dev$i.exchange.nn sen/dev$i.sen.pipe.newbasic senner/dev$i.pipe.newbasic.sen.ner
 	./LabelerNewBasic -l -train senner/train$i.sen.ner -dev senner/dev$i.pipe.newbasic.sen.ner -test senner/test$i.pipe.newbasic.sen.ner -model test.model -option option.pnb.crf  -word emb/word2vec.tin.emb
 	
	echo "LabelerTanh $i"	      
	java -Xmx1G -jar AddFirstOutFeats.jar temp/test$i.exchange.nn sen/test$i.sen.pipe.ltanh senner/test$i.pipe.ltanh.sen.ner
        java -Xmx1G -jar AddFirstOutFeats.jar temp/dev$i.exchange.nn sen/dev$i.sen.pipe.ltanh senner/dev$i.pipe.ltanh.sen.ner
	./LabelerTanh -l -train senner/train$i.sen.ner -dev senner/dev$i.pipe.ltanh.sen.ner -test senner/test$i.pipe.ltanh.sen.ner -model test.model -option option.plt.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNewTanh $i" 
	java -Xmx1G -jar AddFirstOutFeats.jar temp/test$i.exchange.nn sen/test$i.sen.pipe.newltanh senner/test$i.pipe.newltanh.sen.ner
        java -Xmx1G -jar AddFirstOutFeats.jar temp/dev$i.exchange.nn sen/dev$i.sen.pipe.newltanh senner/dev$i.pipe.newltanh.sen.ner
	./LabelerNewTanh -l -train senner/train$i.sen.ner -dev senner/dev$i.pipe.newltanh.sen.ner -test senner/test$i.pipe.newltanh.sen.ner -model test.model -option option.pnlt.crf  -word emb/word2vec.tin.emb
	
	echo "LabelerNNTanh $i"
	java -Xmx1G -jar AddFirstOutFeats.jar temp/test$i.exchange.nn sen/test$i.sen.pipe.tanh senner/test$i.pipe.tanh.sen.ner
        java -Xmx1G -jar AddFirstOutFeats.jar temp/dev$i.exchange.nn sen/dev$i.sen.pipe.tanh senner/dev$i.pipe.tanh.sen.ner
	./LabelerNNTanh -l -train senner/train$i.sen.ner -dev senner/dev$i.pipe.tanh.sen.ner -test senner/test$i.pipe.tanh.sen.ner -model test.model -option option.pt.crf  -word emb/word2vec.tin.emb
	
done
