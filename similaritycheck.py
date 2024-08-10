import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

user_files = [file for file in os.listdir() if file.endswith('.txt')]
user_notes = [open(_file, encoding='utf-8').read() for _file in user_files]

def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity(file1, file2):
    return cosine_similarity([file1],[file2])[0][0]

vectors = vectorize(user_notes)
s_vectors = list(zip(user_files, vectors)) #t
plagiarism_results = set()

def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            similarity_score = similarity(text_vector_a, text_vector_b)
            student_pair = sorted((student_a, student_b)) #t--nodplicates
            score = (student_pair[0], student_pair[1], similarity_score)
            plagiarism_results.add(score)

    return plagiarism_results

if __name__ == "__main__":
    for data in check_plagiarism():
        print(data)
