# digital_breakthrough_2022_vladi

Code for digital breakthrough hackathon https://hacks-ai.ru/

General: 
- Task is to predict Jira Issue completion time 
- Using text models like BOW, TF-IDF and W2V to create text features
- Using normal features like assigne_id and project_id
- Combining normal & text features using Gradient Boosting (Catboost)
- Cross-Validation (10-fold, time sensitive) to check overfitting
- Full model train & Inference on separate test dataset

To launch:
- Run translate_text.ipynb to translate summary text from Russian / Vietnamese to English
- Run prepare_text.ipynb to create different text encodings (bag-of-word, tf-idf, w2v, nearest issues features) on different texts (summary and comments, translated and original, lemmatized and stemmed)
- Run main.ipynb to collect features and train Boosting model. Also creates result in ./result folder

