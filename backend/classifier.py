from sklearn.neighbors import KNeighborsClassifier

class ClassifierFactory:
    @staticmethod
    def create_with_fit(data, target):
        model = KNeighborsClassifier(n_neighbors=15)
        model.fit(data, target)
        return model