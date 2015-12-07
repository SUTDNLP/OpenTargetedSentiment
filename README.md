OpenTargetedSentiment
======
Data and Codes of [Neural Networks for Open Domain Targeted Sentiment](http://www.aclweb.org/anthology/D/D15/D15-1073.pdf), EMNLP 2015  
To run it, please first compile the codes in the sub-folders of OSentMultiInputFormat and OSentSingleInputFormat.  
The codes are a implemented based on our neural library [LibN3L](https://github.com/SUTDNLP/LibN3L)  
You should download LibN3L first, and modify the path in CMakeLists.txt, and get it ready for compilation.  
You will get nine binaries after compiling, and you can follow the commands in script folder to run them. 
Please notice the comments in the command files, there are some simple scripts which you should write by yourself, 
such as split the name entity and  sentiment results from one file, or replacing gold-standard name entity results by automatic results in pipeline models.  
If you have any question, please email to mason.zms@gmail.com 
