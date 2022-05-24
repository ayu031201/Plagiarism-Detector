# we first have to convert our textual data into numbers and we'll do that by a process known as word embedding.. It follows some algorithms that we will take built in from scikit.
# and words are being represented as positions in space and for detection in similiarity we will use vector dot product to determine how closely the two
# texts are similar by computing the value of cosine similiarity between vector representation of text assignments.

import os  # used for creating,removing,fetching directory.  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      

files = [doc for doc in os.listdir() if doc.endswith('.txt')]  # files that ends with '.txt' in the local directory.
notes = [open(_file,encoding='utf-8').read() for _file in files]

def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()  # transform the text to array.
def c_similarity(doc1, doc2): return cosine_similarity([doc1,doc2])

vectors = vectorize(notes)
pair_of_files_and_vector = list(zip(files,vectors)) # list zip creates a pair of student_files and vectors.

# print(s_vectors)
def check_plagiarism(): # O(n*n) complexity of checking for each pair similarity.
    global pair_of_files_and_vector
    results_triplets = set()
    for first_file, first_vector in pair_of_files_and_vector:   # a pair exists in pair_of_files_and_vector
        new_vectors = pair_of_files_and_vector.copy()
        current_index = new_vectors.index((first_file, first_vector))
        del new_vectors[current_index]
        for second_file,second_vector in new_vectors:  # for each element we will take rest of the other elements for it to be compared and delete the current element.
            score = c_similarity(first_vector,second_vector)[0][1]  # passing a 2-D vector for cosine similarity.
            first_pair = sorted((first_file, second_file))
            score = (first_pair[0], first_pair[1],score)
            results_triplets.add(score)
    return results_triplets

for scores in check_plagiarism():  # total pairs will be (n * (n-1)) / 2 . { n is number of .txt files } 
    print(scores)
