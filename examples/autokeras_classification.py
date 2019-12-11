import autokeras as ak
from factnn.utils.cross_validate import get_chunk_of_data

directory = "/home/jacob/"
gamma_dir = [directory + "gammaFeature/no_clean/"]
proton_dir = [directory + "protonFeature/no_clean/"]

x_train, y_train = get_chunk_of_data(directory=gamma_dir, proton_directory=proton_dir, indicies=(60, 100, 1), rebin=75,
                                    chunk_size=1500, as_channels=True)
print(x_train.shape)
print(y_train.shape)
y_train = y_train.argmax(axis=1)
print(y_train.shape)
clf = ak.ImageClassifier(verbose=1, augment=False)
clf.fit(x_train, y_train)
results = clf.evaluate(x_train, y_train)
print(results)