import pickle
books = pickle.load(open("books.pkl", "rb"))
print(books.columns)
