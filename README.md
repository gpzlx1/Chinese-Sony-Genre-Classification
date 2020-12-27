# Chinese-Sony-Genre-Classification
Predict Music Sony From Lyrics For Chinese Sonys

### Dataset
In this repo, we offer the **chinese** songs dataset crawled from the Netease Music, and label them into 4 classes:
* Rock & Roll (摇滚)
* Rap (说唱)
* Ballad (民谣)
* Ancient (古风)
Check details in `$PROJECT_ROOT/data`

### implementation
We also implemented several NLP models to classify Chinese songs. 
* Based on the Neural Networks(details in `$PROJECT_ROOT/src/NN`): TextCNN, FaastText
* Machine learning methods(details in `$PROJECT_ROOT/src/NotNN`): Naive Bayes, Random Forests, Logistic Regression
