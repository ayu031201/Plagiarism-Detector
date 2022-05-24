# Plagiarism-Detector
Built a Plagiarism Detector using some Machine Learning Techniques in Python, which detects the similarity between any two textudocumentsent or any other Code/Essays.

The technique that I used to check the similarity between the files is "Cosine-Similarity".

Here are the steps involed in the code :-

1 -> At first we have to update all of the text documents that we want to compare and take input of all of the ".txt" ending files from our local directory.

2-> After that we have to change all of the textual content from the files to an array of numbers and that can be done with "TfidVectorizer" {from scikit-learn} which   Transforms text to feature vectors that can be used as input to the estimator.

3 -> After converting those texts into numbers we will apply cosine similarity between each pair of unique files and store the scores/detected similarity in a triplet of vector.

4 -> Similarity between files is calculated by the formula: cosine = | A * B | / |norm(A) * norm(B) |.

5 -> We have to run our programme and the output will show (n*(n-1)) / 2 { where n is total number of ".txt" files }, lines of scores which consists of similarity between each pair of files present in that local directory which ends with ".txt" extension.

 *Output.png shows the following similarity/detected plag between the files {Code1.txt, Code2.txt, Code3.txt}.*
